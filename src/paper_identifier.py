import os
import dotenv
import numpy as np
from pathlib import Path
from pypdf import PdfReader
import re
from typing import List, Tuple, Optional

# local imports
from tasks.functions import get_embedding_from_ollama

dotenv.load_dotenv()


def extract_text(file_path: str) -> str:
    """Extract complete text from PDF"""
    reader = PdfReader(file_path)
    text = reader.pages[0].extract_text()
    return text.strip()


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace and normalize
    text = re.sub(r"\s+", " ", text.strip())
    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s\.\,\;\:\!\?\-]", " ", text)
    return text


def get_paper_files(folder_path: str) -> List[str]:
    """
    Recursively get all paper files from the folder.

    Args:
        folder_path: Root folder path to search

    Returns:
        List of file paths
    """
    paper_extensions = {".pdf", ".docx", ".txt", ".md"}
    paper_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in paper_extensions:
                paper_files.append(os.path.join(root, file))

    return paper_files


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def identify_paper(
    paper_name: str,
    folder_path: str = os.getenv("FOLDER_PATH"),
    verbose=False,
) -> Tuple[str, float]:
    """
    Identify which paper in the folder best matches the given paper name using embeddings.

    Args:
        paper_name: Name or description of the paper to find
        folder_path: Path to the folder containing papers

    Returns:
        Tuple of (best_match_file_path, similarity_score)
    """
    if not folder_path or not os.path.exists(folder_path):
        raise ValueError(f"Invalid folder path: {folder_path}")

    # Get embedding for the query paper name

    query_embedding = get_embedding_from_ollama(paper_name)

    # Get all paper files
    paper_files = get_paper_files(folder_path)

    if not paper_files:
        raise ValueError(f"No paper files found in {folder_path}")

    best_match = None
    best_similarity = -1
    similarities = []

    for file_path in paper_files:
        try:

            # Extract text from file
            text_content = extract_text(file_path)

            if not text_content.strip():

                continue

            # Create a combined string with filename and content for better matching
            filename = os.path.splitext(os.path.basename(file_path))[0]
            combined_text = f"{filename}. {text_content}"

            # Get embedding for the file content
            file_embedding = get_embedding_from_ollama(combined_text)

            # Calculate similarity
            similarity = cosine_similarity(query_embedding, file_embedding)
            similarities.append((file_path, similarity))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = file_path

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if best_match is None:
        raise ValueError("No valid papers found or processed")

    # Sort and display top matches
    similarities.sort(key=lambda x: x[1], reverse=True)
    if verbose:
        print(f"\nTop 5 matches:")
        for i, (path, sim) in enumerate(similarities[:5], 1):
            print(f"{i}. {os.path.basename(path)}: {sim:.4f}")

    return best_match, best_similarity


if __name__ == "__main__":
    # Example usage
    paper_to_find = "segment anything model"
    result = identify_paper(paper_to_find, verbose=True)[0]

    if result:
        print(f"\nFound paper: {result}")
    else:
        print("No suitable paper found.")
