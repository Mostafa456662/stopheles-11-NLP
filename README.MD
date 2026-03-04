# 📄 Research Paper Analysis Tool

A CLI-based conversational tool for analyzing ML research papers. Supports **summarization**, **explanation via semantic search**, and **automatic classification** of papers into folders — all powered locally via **Ollama (Gemma 3 + nomic-embed-text)**.

---

## 🧠 Overview

The tool runs as an interactive chat loop. Each user message is passed through a **confidence-based router** that decides whether to call a specialized function or answer the query directly as an ML assistant.

- **Summarize** — chunk a paper and generate a comprehensive multi-part summary
- **Explain** — semantically search across all papers to find and explain relevant passages
- **Classify** — embed a new paper and move it into the most similar existing folder (or create a new one)

---

## 🏗️ Architecture

```
main.py (Chat Loop)
   └── router.py
         ├── get_confidence()       → LLM scores how likely this is a function call (0–1)
         │
         ├── [confidence >= 0.8] → select_and_call_function()
         │     ├── identify_paper()         → cosine similarity search to find the paper file
         │     ├── summarise_paper()        → chunked summarization pipeline
         │     ├── explain()                → semantic chunk search + passage synthesis
         │     └── classify()               → embedding-based folder classification
         │
         └── [confidence < 0.8]  → generate()   (direct LLM response)
```

---

## 🔄 Processing Pipelines

### 1. 📝 Summarization (`tasks/summarise.py`)

1. Extracts full text from the PDF via `pypdf`
2. Splits text into **overlapping chunks** (1800 chars, 200 char overlap, sentence-boundary aware)
3. Summarizes each chunk individually with context-aware prompts (intro / middle / conclusion)
4. Combines all chunk summaries into one final **integrated summary** via a second LLM call

### 2. 🔍 Explanation (`tasks/explain.py`)

1. Recursively indexes all PDFs in `PAPERS_PATH` into a `ChunkIndex`
2. Each chunk is embedded with `nomic-embed-text` via Ollama
3. A query embedding is compared against all chunk embeddings using **cosine similarity**
4. Papers are ranked by their **mean top-3 chunk similarity score**
5. The best paper's top chunks are merged into a coherent passage via LLM
6. A final explanation is generated against the query

### 3. 🗂️ Classification (`tasks/classify.py`)

1. Scans `FOLDER_PATH` and embeds existing papers organized by subfolder
2. Embeds the new paper and calculates **average cosine similarity** against each folder
3. If best similarity ≥ `SIMILARITY_THRESHOLD` → moves paper into that folder
4. Otherwise → creates a new `cluster_N` folder and moves the paper there
5. Supports an optional **NPZ + JSON embedding cache** to skip re-embedding on repeat runs

### 4. 🔎 Paper Identification (`paper_identifier.py`)

Used by summarization and classification to resolve a paper name from natural language:

1. Embeds the user's description of the paper
2. Scans all files in `FOLDER_PATH` (PDF, DOCX, TXT, MD)
3. Extracts the first ~200 chars + filename from each file and embeds it
4. Returns the file path with the **highest cosine similarity** to the query

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language Model | [Ollama](https://ollama.com/) — `gemma3:4b` |
| Embedding Model | [Ollama](https://ollama.com/) — `nomic-embed-text` |
| PDF Parsing | `pypdf` |
| Vector Similarity | `numpy` + `sklearn` cosine similarity |
| Embedding Cache | NumPy `.npz` + JSON metadata |
| Config | `.env` via `python-dotenv` |

---

## 🚀 Getting Started

### Prerequisites

- [Ollama](https://ollama.com/) installed and running
- Required models pulled:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

- Python 3.9+

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
FOLDER_PATH=/path/to/your/organized/paper/folders
PAPERS_PATH=/path/to/all/papers
DBS_PATH=/path/to/store/embedding/cache
SIMILARITY_THRESHOLD=0.6
```

| Variable | Description |
|---|---|
| `FOLDER_PATH` | Root folder containing subfolders of categorized papers |
| `PAPERS_PATH` | Flat or nested folder of all papers (used by explain) |
| `DBS_PATH` | Where embedding cache (`.npz` + `.json`) is stored |
| `SIMILARITY_THRESHOLD` | Float (0–1). Papers below this go into a new folder |

---

## ▶️ Running

```bash
python main.py
```

The tool starts an interactive chat loop:

```
You: summarise the ViT paper
Assistant: [finds paper → chunks → summarizes → combines]

You: explain how patch embeddings work in the LViT paper
Assistant: [indexes papers → searches → constructs passage → explains]

You: classify 1503.02531v1.pdf
Assistant: [embeds paper → compares folders → moves file]

You: what is knowledge distillation?
Assistant: [answers directly without calling a function]
```

Type `quit`, `exit`, or `bye` to stop.



## ⚙️ Key Configuration

These values are hardcoded or set via `.env`:

| Setting | Default | Location |
|---|---|---|
| LLM model | `gemma3:4b` | `tasks/gemma.py` |
| Embedding model | `nomic-embed-text` | `tasks/functions.py` |
| Max tokens per response | `500` | `tasks/gemma.py` |
| Temperature | `0.3` | `tasks/gemma.py` |
| Chunk size | `1800` chars | `tasks/functions.py` |
| Chunk overlap | `200` chars | `tasks/functions.py` |
| Similarity threshold | `.env` | `tasks/classify.py` |
| Routing confidence cutoff | `0.8` | `router.py` |
| Explain: top papers | `5` | `tasks/explain.py` |
| Explain: chunks per paper | `3` | `tasks/explain.py` |

---

## 📦 Requirements

```
ollama
pypdf
numpy
scikit-learn
python-dotenv
```
