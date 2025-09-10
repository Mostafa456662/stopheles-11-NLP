import os
import json
import dotenv
import ollama
import shutil
import pickle
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()


FOLDER_PATH = os.getenv("FOLDER_PATH")
DBS_PATH = os.getenv("DBS_PATH")
PAPERS_PATH = os.getenv("PAPERS_PATH")
EMBEDDING_MODEL = "nomic-embed-text"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD"))


def extract_text(file_path: str, max_chars: int = 2048) -> str:
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "
            if len(full_text) >= max_chars:
                break
    return full_text


def get_embedding_from_ollama(text, model=EMBEDDING_MODEL):
    """Get embedding from Ollama"""
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return np.array(response["embedding"])
    except Exception as e:
        print(f"Error getting embedding from Ollama: {e}")
        return None


def get_paper_paths_by_folder(root_folder):
    """Get all paper paths organized by folder"""
    papers_by_folder = {}

    for folder_name in os.listdir(root_folder):
        FOLDER_PATH = os.path.join(root_folder, folder_name)
        if os.path.isdir(FOLDER_PATH):
            papers = []
            for file_name in os.listdir(FOLDER_PATH):
                if file_name.endswith(".pdf"):
                    papers.append(os.path.join(FOLDER_PATH, file_name))
            if papers:
                papers_by_folder[folder_name] = papers

    return papers_by_folder


def create_embeddings_dict(papers_by_folder, extract_text):
    """Create embeddings dictionary with folder indices"""
    embeddings_dict = {}

    for folder_name, paper_paths in papers_by_folder.items():
        folder_embeddings = []

        print(f"Processing folder: {folder_name}")
        for paper_path in paper_paths:
            try:
                text = extract_text(paper_path)
                if text and text.strip():
                    embedding = get_embedding_from_ollama(text)
                    if embedding is not None:
                        folder_embeddings.append(
                            {"path": paper_path, "embedding": embedding, "text": text}
                        )
                        print(f"  Processed: {os.path.basename(paper_path)}")
                    else:
                        print(
                            f"  Failed to get embedding: {os.path.basename(paper_path)}"
                        )
                else:
                    print(f"  Skipped (no text): {os.path.basename(paper_path)}")
            except Exception as e:
                print(f"  Error processing {paper_path}: {e}")

        if folder_embeddings:
            embeddings_dict[folder_name] = folder_embeddings

    return embeddings_dict


def calculate_folder_similarity(target_embedding, folder_embeddings):
    """Calculate average cosine similarity between target and folder papers"""
    if not folder_embeddings:
        return 0.0

    similarities = []
    for paper_data in folder_embeddings:
        similarity = cosine_similarity(
            target_embedding.reshape(1, -1), paper_data["embedding"].reshape(1, -1)
        )[0][0]
        similarities.append(similarity)

    return np.mean(similarities)


def find_best_folder(target_embedding, embeddings_dict):
    """Find the folder with highest average similarity"""
    best_folder = None
    best_similarity = -1.0

    folder_similarities = {}

    for folder_name, folder_embeddings in embeddings_dict.items():
        avg_similarity = calculate_folder_similarity(
            target_embedding, folder_embeddings
        )
        folder_similarities[folder_name] = avg_similarity

        if avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_folder = folder_name

    return best_folder, best_similarity, folder_similarities


def create_new_folder(root_folder, base_name="cluster"):
    """Create a new folder with unique name"""
    counter = 1
    while True:
        new_folder_name = f"{base_name}_{counter}"
        new_FOLDER_PATH = os.path.join(root_folder, new_folder_name)
        if not os.path.exists(new_FOLDER_PATH):
            os.makedirs(new_FOLDER_PATH)
            return new_folder_name, new_FOLDER_PATH
        counter += 1


def organize_paper(
    paper_path,
    root_folder,
    embeddings_dict,
    extract_text,
    threshold=SIMILARITY_THRESHOLD,
):
    """Organize a single paper based on text similarity"""
    try:
        # Extract text and create embedding
        text = extract_text(paper_path)
        if not text or not text.strip():
            print(f"No text found for {os.path.basename(paper_path)}")
            return None

        target_embedding = get_embedding_from_ollama(text)
        if target_embedding is None:
            print(f"Failed to get embedding for {os.path.basename(paper_path)}")
            return None

        # Find best matching folder
        best_folder, best_similarity, all_similarities = find_best_folder(
            target_embedding, embeddings_dict
        )

        print(f"\nAnalyzing: {os.path.basename(paper_path)}")
        print(f"Folder similarities:")
        for folder, sim in all_similarities.items():
            print(f"  {folder}: {sim:.3f}")

        # Decide on placement
        if best_folder and best_similarity >= threshold:
            # Move to existing folder
            target_FOLDER_PATH = os.path.join(root_folder, best_folder)
            target_path = os.path.join(target_FOLDER_PATH, os.path.basename(paper_path))

            shutil.move(paper_path, target_path)
            print(
                f"Moved to existing folder '{best_folder}' (similarity: {best_similarity:.3f})"
            )

            # Update embeddings dictionary
            embeddings_dict[best_folder].append(
                {"path": target_path, "embedding": target_embedding, "text": text}
            )

            return best_folder
        else:
            # Create new folder
            new_folder_name, new_FOLDER_PATH = create_new_folder(root_folder)
            target_path = os.path.join(new_FOLDER_PATH, os.path.basename(paper_path))

            shutil.move(paper_path, target_path)
            print(
                f"Created new folder '{new_folder_name}' (max similarity: {best_similarity:.3f})"
            )

            # Update embeddings dictionary
            embeddings_dict[new_folder_name] = [
                {"path": target_path, "embedding": target_embedding, "text": text}
            ]

            return new_folder_name

    except Exception as e:
        print(f"Error organizing {paper_path}: {e}")
        return None


def save_embeddings_cache(embeddings_dict, cache_path):
    """Save embeddings dictionary to NPZ and JSON files"""

    # Prepare data for NPZ (embeddings only)
    embeddings_data = {}
    metadata = {}

    for folder_name, papers in embeddings_dict.items():
        folder_embeddings = []
        folder_metadata = []

        for i, paper_data in enumerate(papers):
            folder_embeddings.append(paper_data["embedding"])
            folder_metadata.append(
                {"path": paper_data["path"], "text": paper_data["text"]}
            )

        if folder_embeddings:
            # Stack embeddings into a 2D array
            embeddings_data[folder_name] = np.stack(folder_embeddings)
            metadata[folder_name] = folder_metadata

    # Save embeddings as NPZ
    np.savez_compressed(f"{cache_path}.npz", **embeddings_data)

    # Save metadata as JSON
    with open(f"{cache_path}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Saved embeddings to {cache_path}.npz and metadata to {cache_path}_metadata.json"
    )


def load_embeddings_cache(cache_path):
    """Load embeddings dictionary from NPZ and JSON files"""
    try:
        # Load embeddings
        embeddings_file = f"{cache_path}.npz"
        metadata_file = f"{cache_path}_metadata.json"

        if not (os.path.exists(embeddings_file) and os.path.exists(metadata_file)):
            return {}

        embeddings_data = np.load(embeddings_file)

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Reconstruct embeddings_dict
        embeddings_dict = {}

        for folder_name in embeddings_data.files:
            folder_embeddings = embeddings_data[folder_name]
            folder_metadata = metadata[folder_name]

            papers_list = []
            for i, embedding in enumerate(folder_embeddings):
                papers_list.append(
                    {
                        "path": folder_metadata[i]["path"],
                        "embedding": embedding,
                        "text": folder_metadata[i]["text"],
                    }
                )

            embeddings_dict[folder_name] = papers_list

        embeddings_data.close()
        print(f"Loaded embeddings from {embeddings_file}")
        return embeddings_dict

    except Exception as e:
        print(f"Error loading cache: {e}")
        return {}


def classify(
    new_paper,
    extract_text=extract_text,
    root_folder=FOLDER_PATH,
    threshold=SIMILARITY_THRESHOLD,
    use_cache=False,
):
    """Main function to organize papers"""
    cache_path = os.path.join(DBS_PATH, "embeddings_cache")

    new_paper_path = PAPERS_PATH + new_paper
    # Load or create embeddings dictionary

    if use_cache:
        print("Loading embeddings from cache...")
        embeddings_dict = load_embeddings_cache(cache_path)
    else:
        print("Creating embeddings for existing papers...")
        papers_by_folder = get_paper_paths_by_folder(root_folder)
        embeddings_dict = create_embeddings_dict(papers_by_folder, extract_text)
        if use_cache:
            save_embeddings_cache(embeddings_dict, cache_path)

    print(f"\nFound {len(embeddings_dict)} folders with papers")
    for folder_name, papers in embeddings_dict.items():
        print(f"  {folder_name}: {len(papers)} papers")

    # Organize the new paper
    result_folder = organize_paper(
        new_paper_path, root_folder, embeddings_dict, extract_text, threshold
    )

    # Save updated cache
    if use_cache:
        save_embeddings_cache(embeddings_dict, cache_path)

    return result_folder, embeddings_dict


def main():
    """Example of how to use the paper organizer"""

    root_folder = os.getenv("FOLDER_PATH")
    new_paper = "1503.02531v1.pdf"

    # Organize the paper
    result_folder, embeddings_dict = classify(
        root_folder=root_folder,
        new_paper=new_paper,
        threshold=SIMILARITY_THRESHOLD,
        use_cache=False,
    )

    if result_folder:
        print(f"\nPaper organized into folder: {result_folder}")
    else:
        print("\nFailed to organize paper")


if __name__ == "__main__":
    main()
