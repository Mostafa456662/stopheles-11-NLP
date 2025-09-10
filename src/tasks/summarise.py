import re
from pypdf import PdfReader
from typing import List, Tuple
from gemma import generate


def extract_full_text(file_path: str) -> str:
    """Extract complete text from PDF"""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "
    return full_text.strip()


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters for English)"""
    return len(text) // 4


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


def summarise_chunk(chunk: str, generate, chunk_index: int, total_chunks: int) -> str:
    """
    summarise a single chunk of text

    Args:
        chunk: Text chunk to summarise
        generate: LLM generation function
        chunk_index: Current chunk index (0-based)
        total_chunks: Total number of chunks

    Returns:
        Summary of the chunk
    """
    if chunk_index == 0 and total_chunks == 1:
        # Single chunk - comprehensive summary
        prompt = f"""summarise this research paper excerpt comprehensively. Focus on:
- Main research question and objectives
- Key methodology and approach
- Important findings and results  
- Conclusions and implications

Text to summarise:
{chunk}

Provide a clear, structured summary:"""

    elif chunk_index == 0:
        # First chunk - likely contains abstract/intro
        prompt = f"""summarise this opening section of a research paper (part 1 of {total_chunks}). Focus on:
- Research problem and objectives
- Background context
- Methodology overview
- Key hypotheses or questions

Text to summarise:
{chunk}

Summary:"""

    elif chunk_index == total_chunks - 1:
        # Last chunk - likely contains conclusions
        prompt = f"""summarise this concluding section of a research paper (part {chunk_index + 1} of {total_chunks}). Focus on:
- Main findings and results
- Key conclusions
- Implications and future work
- Limitations if mentioned

Text to summarise:
{chunk}

Summary:"""

    else:
        # Middle chunks - methodology, results, discussion
        prompt = f"""summarise this middle section of a research paper (part {chunk_index + 1} of {total_chunks}). Focus on:
- Key methods, experiments, or analysis
- Important findings or results
- Critical insights or observations
- Relevant data or evidence

Text to summarise:
{chunk}

Summary:"""

    return generate(prompt=prompt, verbose=False)


def combine_summaries(summaries: List[str], generate) -> str:
    """
    Combine individual chunk summaries into a final comprehensive summary

    Args:
        summaries: List of chunk summaries
        generate: LLM generation function

    Returns:
        Final combined summary
    """
    combined_text = "\n\n".join(
        [f"Section {i+1}:\n{summary}" for i, summary in enumerate(summaries)]
    )

    prompt = f"""Below are summaries of different sections of a research paper. Create a cohesive, comprehensive summary that integrates all sections:

{combined_text}

Create a unified summary covering:
1. Research objective and problem
2. Methodology and approach
3. Key findings and results
4. Main conclusions and implications

Final integrated summary:"""

    return generate(prompt=prompt, verbose=False)


def summarise_paper(file_path: str, generate=generate) -> dict:
    """
    summarise a research paper using overlapping chunks and multiple LLM calls

    Args:
        file_path: Path to the PDF paper
        generate: Function that takes a prompt and returns LLM response

    Returns:
        Dictionary containing:
        - 'final_summary': The comprehensive final summary
        - 'chunk_summaries': List of individual chunk summaries
        - 'total_chunks': Number of chunks processed
        - 'original_length': Original text length in characters
    """
    print(f"Processing paper: {file_path}")

    # Extract full text
    full_text = extract_full_text(file_path)
    if not full_text:
        return {"error": "Could not extract text from PDF"}

    print(
        f"Extracted text length: {len(full_text)} characters (~{estimate_tokens(full_text)} tokens)"
    )

    # Create overlapping chunks
    chunks = create_overlapping_chunks(full_text)
    print(f"Created {len(chunks)} overlapping chunks")

    # summarise each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        summary = summarise_chunk(chunk, generate, i, len(chunks))
        chunk_summaries.append(summary)
        print(f"Generated summary {i+1}: {len(summary)} characters")

    # Combine summaries into final summary
    print("Combining summaries into final comprehensive summary...")
    final_summary = combine_summaries(chunk_summaries, generate)

    return {
        "final_summary": final_summary,
        "chunk_summaries": chunk_summaries,
        "total_chunks": len(chunks),
        "original_length": len(full_text),
    }


def main():
    result = summarise_paper(
        "doc/papers/CV/Fine_Tune_LViT_for_zero_shot_classifiction[1].pdf",
    )

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Processed {result['total_chunks']} chunks")
        print(f"Final Summary:\n{result['final_summary']}")


if __name__ == "__main__":
    main()