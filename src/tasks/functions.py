import os
import ollama
import numpy as np
from typing import List
from pypdf import PdfReader


EMBEDDING_MODEL = "nomic-embed-text"


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


def create_overlapping_chunks(
    text: str, chunk_size: int = 1800, overlap_size: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters (leaving room for prompt)
        overlap_size: Number of characters to overlap between chunks

    Returns:
        List of text chunks with overlaps
    """
    chunks = []
    start = 0

    while start < len(text):
        # Define chunk end
        end = start + chunk_size

        # If this isn't the last chunk, try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings in the last 200 characters
            search_start = max(end - 200, start + chunk_size // 2)
            sentence_end = -1

            for i in range(end, search_start, -1):
                if (
                    text[i : i + 1] in ".!?"
                    and i + 1 < len(text)
                    and text[i + 1].isspace()
                ):
                    sentence_end = i + 1
                    break

            if sentence_end != -1:
                end = sentence_end

        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position (with overlap)
        if end >= len(text):
            break
        start = end - overlap_size

        # Ensure we're making progress
        if start <= len(chunks) * (chunk_size - overlap_size):
            start = len(chunks) * (chunk_size - overlap_size)

    return chunks
