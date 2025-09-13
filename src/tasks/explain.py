import os
import dotenv
import numpy as np
from typing import List, Dict
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


# Local imports
from tasks.gemma import generate
from tasks.functions import (
    extract_text,
    get_embedding_from_ollama,
    create_overlapping_chunks,
)


dotenv.load_dotenv()


class ChunkIndex:
    def __init__(self):
        self.chunks = []  # List of chunk texts
        self.embeddings = []  # List of embeddings
        self.metadata = []  # List of {paper_path, chunk_idx, paper_title}

    def add_paper(self, paper_path: str, paper_title: str = None):
        """Add all chunks from a paper to the index"""

        if not paper_title:
            paper_title = os.path.splitext(os.path.basename(paper_path))[0]

        full_text = extract_text(paper_path)
        if not full_text:
            return

        chunks = create_overlapping_chunks(full_text)

        for i, chunk in enumerate(chunks):
            embedding = get_embedding_from_ollama(chunk)
            if embedding is not None:
                self.chunks.append(chunk)
                self.embeddings.append(embedding)
                self.metadata.append(
                    {
                        "paper_path": paper_path,
                        "paper_title": paper_title,
                        "chunk_idx": i,
                        "total_chunks": len(chunks),
                    }
                )

    def search(
        self, query: str, top_papers: int = 5, chunks_per_paper: int = 3
    ) -> List[Dict]:
        """Search prioritize papers with multiple relevant chunks"""
        if not self.embeddings:
            return []

        query_embedding = get_embedding_from_ollama(query)
        if query_embedding is None:
            return []

        # Calculate similarities
        embeddings_matrix = np.array(self.embeddings)
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), embeddings_matrix
        )[0]

        # Group by paper
        paper_scores = defaultdict(list)
        for i, sim in enumerate(similarities):
            paper_path = self.metadata[i]["paper_path"]
            paper_scores[paper_path].append(
                {
                    "similarity": sim,
                    "chunk_idx": i,
                    "chunk": self.chunks[i],
                    "metadata": self.metadata[i],
                }
            )

        # Compute paper-level score (mean of top 3 chunk similarities)
        paper_chunks = []
        for paper_path, chunks in paper_scores.items():
            # Sort chunks by similarity (descending)
            chunks.sort(key=lambda x: x["similarity"], reverse=True)

            # score = mean of top 3 similarities
            top_similarities = [c["similarity"] for c in chunks[:3]]
            score = np.mean(top_similarities)

            paper_chunks.append(
                {
                    "paper_path": paper_path,
                    "paper_title": chunks[0]["metadata"]["paper_title"],
                    "score": float(score),
                    "chunks": chunks,
                }
            )

        # Sort papers by score
        paper_chunks.sort(key=lambda x: x["score"], reverse=True)

        # Select top papers
        selected_papers = paper_chunks[:top_papers]

        # For each selected paper, take top chunks
        final_chunks = []
        for paper in selected_papers:
            top_chunks = paper["chunks"][:chunks_per_paper]
            for chunk_data in top_chunks:
                final_chunks.append(
                    {
                        "chunk": chunk_data["chunk"],
                        "similarity": chunk_data["similarity"],
                        "metadata": chunk_data["metadata"],
                        "paper_chunks_score": paper["score"],
                    }
                )

        return final_chunks


def construct_passage(chunks: List[Dict], paper_title: str) -> str:
    """Construct coherent passage from overlapping chunks using LLM"""
    if len(chunks) == 1:
        return chunks[0]["chunk"]

    # Combine chunk texts
    combined_text = "\n\n".join([chunk["chunk"] for chunk in chunks])

    prompt = f"""The following are overlapping sections from the paper "{paper_title}". 
Combine them into a single coherent passage, removing redundancy while preserving all important information: include only the passage
and no additional text

{combined_text}

Create a coherent passage:"""

    return generate(prompt=prompt)


def explain_passage(query: str, passage: str):
    prompt = f"""your job is to explain this passage: {passage} according to this query {query} provide the explanation only"""
    return generate(prompt=prompt)


def explain(
    query: str,
    papers_dir: str = os.getenv("PAPERS_PATH"),
    top_papers: int = 5,
    chunks_per_paper: int = 3,
) -> str:
    """Main function to query papers and return the explanation of found passages from the best-represented paper."""

    # Initialize index
    index = ChunkIndex()

    # Index all papers recursively
    for root, dirs, files in os.walk(papers_dir):
        for file in files:
            if file.endswith(".pdf"):
                paper_path = os.path.join(root, file)
                index.add_paper(paper_path)

    if not index.chunks:
        return "No papers found or indexed."

    search_results = index.search(
        query, top_papers=top_papers, chunks_per_paper=chunks_per_paper
    )

    if not search_results:
        return "No relevant information found."

    # Group by paper
    paper_groups = defaultdict(list)
    for result in search_results:
        paper_path = result["metadata"]["paper_path"]
        paper_groups[paper_path].append(result)

    # Find paper with the MOST retrieved chunks
    best_paper_path = max(paper_groups, key=lambda p: len(paper_groups[p]))
    best_paper_chunks = paper_groups[best_paper_path]
    paper_title = best_paper_chunks[0]["metadata"]["paper_title"]

    print(f"Paper: {paper_title}")

    # Construct coherent passage using LLM
    passage = construct_passage(best_paper_chunks, paper_title)

    answer = explain_passage(query=query, passage=passage)

    return f"Source: {paper_title}\n\n{answer}"


def main():
    """Example usage"""
    papers_folder = os.getenv("PAPERS_PATH")

    query = """Can you explain this passage,  Model Variants. We base ViT configurations on those used for BERT (Devlin et al., 2019), as
 summarized in Table 1. The “Base” and “Large” models are directly adopted from BERT and we
 add the larger “Huge” model. In what follows we use brief notation to indicate the model size and
 the input patch size: for instance, ViT-L/16 means the “Large” variant with 16 16 input patch size.
 Note that the Transformer’s sequence length is inversely proportional to the square of the patch size,
 thus models with smaller patch size are computationally more expensive."""

    answer = explain(
        query=query,
        papers_dir=papers_folder,
        top_papers=5,
        chunks_per_paper=3,
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
