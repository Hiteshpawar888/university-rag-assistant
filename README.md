# University AI Assistant using RAG

## Project Overview

This project is a Retrieval-Augmented Generation chatbot that answers questions from university PDF documents.

It extracts text from PDF files, splits the text into chunks, converts the chunks into semantic embeddings using Sentence Transformers, stores them in ChromaDB, and generates source-grounded answers using OpenAI GPT-4o-mini.

## Features

- PDF text extraction using PyPDF
- Text chunking with overlap
- Semantic embeddings using Sentence Transformers
- Vector storage using ChromaDB
- Similarity-based document retrieval
- OpenAI GPT-4o-mini answer generation
- Source PDF and page number citation
- Streamlit web interface

## Tech Stack

Python, Streamlit, PyPDF, Sentence Transformers, ChromaDB, OpenAI API, python-dotenv

## How It Works

1. University PDF documents are stored in `data/pdfs`.
2. Text is extracted from each PDF page.
3. Text is split into overlapping chunks.
4. Each chunk is converted into an embedding.
5. Embeddings are stored in ChromaDB.
6. User questions are converted into embeddings.
7. ChromaDB retrieves the most relevant chunks.
8. GPT-4o-mini generates an answer using only retrieved context.
9. The app displays the answer with PDF name and page number.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt