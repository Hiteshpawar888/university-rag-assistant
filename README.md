# Document Intelligence Assistant using RAG

A professional Retrieval-Augmented Generation application that allows users to ask questions from PDF documents and receive source-grounded answers with PDF name and page references.

This project supports both a pre-built university PDF knowledge base and custom PDF upload functionality.

---

## Project Overview

Universities and organisations often store important information inside long PDF documents such as student handbooks, academic policies, library guides and support documents. Searching these documents manually can be slow and time-consuming.

This project solves that problem by building an AI assistant that reads PDF documents, retrieves the most relevant sections, and generates clear answers using a Large Language Model.

The assistant does not answer from memory only. It first retrieves relevant document chunks from ChromaDB and then uses those chunks as context to generate a grounded answer.

---

## Key Features

- Pre-built university PDF knowledge base
- Custom PDF upload feature
- PDF text extraction using PyPDF
- Text chunking with overlap
- Semantic embeddings using Sentence Transformers
- Vector storage and retrieval using ChromaDB
- GPT-4o-mini answer generation
- Source PDF name and page number citations
- Chat-style Streamlit interface
- Conversation history using Streamlit session state
- Professional dashboard-style UI
- Evaluation questions and results included

---

## Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- ChromaDB
- OpenAI API
- python-dotenv
- GitHub

---

## RAG Pipeline

The application follows this Retrieval-Augmented Generation workflow:

1. PDF documents are loaded.
2. Text is extracted from each PDF page.
3. Extracted text is split into smaller overlapping chunks.
4. Each chunk is converted into a semantic embedding using Sentence Transformers.
5. Embeddings and metadata are stored in ChromaDB.
6. When a user asks a question, the question is converted into an embedding.
7. ChromaDB retrieves the most relevant document chunks.
8. The retrieved chunks are passed to GPT-4o-mini as context.
9. The model generates a short answer using only the retrieved context.
10. The app displays the answer with source PDF name and page number.

---

## Application Modes

### 1. Built-in University PDF Mode

This mode uses a pre-built university document collection.

The system processes:

- 15 PDF documents
- 455 PDF pages
- 998+ text chunks

Users can ask questions about academic integrity, plagiarism, library services, student support, misconduct policies, appeals and other university-related topics.

### 2. Custom PDF Upload Mode

This mode allows users to upload their own PDF files.

The app then:

1. Reads the uploaded PDFs.
2. Extracts text from the uploaded documents.
3. Creates text chunks.
4. Builds a temporary ChromaDB vector database.
5. Allows users to ask questions from the uploaded documents.

This makes the project more flexible and closer to a real-world document intelligence product.

---

## Example Questions

Users can ask questions such as:

- What is academic integrity?
- What is plagiarism?
- How can students use the library?
- What support is available for disabled students?
- What is academic misconduct?
- What happens if a student cheats?
- How can students appeal a decision?
- What library services are available?
- What is contract cheating?
- What support is available for students?

## Screenshots

### Built-in University PDF Mode
![Built-in Mode](assets/app_home.png)

### Custom PDF Upload Mode
![Upload Mode](assets/pdf_upload_demo.png)

---

## Project Structure

```text
university-rag-assistant/
│
├── data/
│   └── pdfs/
│
├── notebooks/
│   └── 01_rag_learning.ipynb
│
├── src/
│   ├── ingest_chroma.py
│   ├── rag_pipeline_chroma.py
│   ├── upload_rag_pipeline.py
│   └── rag_pipeline.py
│
├── app.py
├── README.md
├── requirements.txt
├── evaluation_questions.txt
├── evaluation_results.txt
├── .gitignore
└── .env