# AmbedkarGPT
  
A simple command-line Q&A system that ingests a short speech by Dr. B.R. Ambedkar and answers questions based solely on that content.

## What this project does
- Loads `speech.txt`
- Splits the text into chunks
- Creates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores embeddings in a local ChromaDB
- Retrieves relevant chunks for a user question
- Answers using Ollama (Mistral 7B) LLM via LangChain

## Requirements
- Python 3.10+
- Ollama installed and `mistral` model pulled
- Internet for first-time embedding model download

## Setup (step-by-step)
1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd AmbedkarGPT

