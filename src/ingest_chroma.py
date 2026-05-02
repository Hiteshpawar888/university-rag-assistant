from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


PDF_FOLDER = Path("data/pdfs")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "university_docs"


def load_pdfs():
    all_documents = []
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    print("Number of PDFs found:", len(pdf_files))

    for pdf_file in pdf_files:
        print("Reading:", pdf_file.name)
        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text:
                all_documents.append({
                    "source": pdf_file.name,
                    "page": page_number,
                    "text": text
                })

    print("Total pages loaded:", len(all_documents))
    return all_documents


def simple_text_splitter(text, chunk_size=1500, chunk_overlap=250):
    chunks_list = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks_list.append(chunk)
        start = end - chunk_overlap

    return chunks_list


def create_chunks(all_documents):
    chunks = []

    for doc in all_documents:
        split_texts = simple_text_splitter(doc["text"])

        for chunk_text in split_texts:
            if len(chunk_text.strip()) > 100:
                chunks.append({
                    "source": doc["source"],
                    "page": doc["page"],
                    "text": chunk_text
                })

    print("Total chunks created:", len(chunks))
    return chunks


def create_chroma_database(chunks):
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print("Old collection deleted.")
    except Exception:
        print("No old collection found. Creating new collection.")

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings. This may take a few minutes...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    print("Storing embeddings in ChromaDB...")

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        collection.add(
            ids=[str(i) for i in range(start, min(end, len(chunks)))],
            documents=[chunk["text"] for chunk in chunks[start:end]],
            metadatas=[
                {
                    "source": chunk["source"],
                    "page": chunk["page"]
                }
                for chunk in chunks[start:end]
            ],
            embeddings=[
                embeddings[i].tolist()
                for i in range(start, min(end, len(chunks)))
            ]
        )

    print("ChromaDB created successfully.")
    print("Total stored chunks:", collection.count())


def main():
    all_documents = load_pdfs()
    chunks = create_chunks(all_documents)
    create_chroma_database(chunks)


if __name__ == "__main__":
    main()