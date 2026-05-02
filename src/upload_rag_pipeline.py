import os
import uuid
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

client = OpenAI(api_key=api_key)


def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_uploaded_pdfs(uploaded_files):
    all_documents = []

    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text:
                all_documents.append({
                    "source": uploaded_file.name,
                    "page": page_number,
                    "text": text
                })

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
        split_texts = simple_text_splitter(
            doc["text"],
            chunk_size=1500,
            chunk_overlap=250
        )

        for chunk_text in split_texts:
            if len(chunk_text.strip()) > 100:
                chunks.append({
                    "source": doc["source"],
                    "page": doc["page"],
                    "text": chunk_text
                })

    return chunks


def build_uploaded_chroma_database(uploaded_files):
    all_documents = extract_text_from_uploaded_pdfs(uploaded_files)
    chunks = create_chunks(all_documents)

    if len(chunks) == 0:
        return None, 0, 0

    embedding_model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    )

    chroma_client = chromadb.EphemeralClient()

    collection_name = f"uploaded_docs_{uuid.uuid4().hex[:8]}"

    collection = chroma_client.create_collection(name=collection_name)

    ids = []
    documents = []
    metadatas = []
    embeddings_list = []

    for i, chunk in enumerate(chunks):
        ids.append(str(i))
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"]
        })
        embeddings_list.append(embeddings[i].tolist())

    batch_size = 100

    for start in range(0, len(ids), batch_size):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings_list[start:end]
        )

    return collection, len(all_documents), len(chunks)


def search_uploaded_documents(question, collection, n_results=2):
    embedding_model = get_embedding_model()

    question_embedding = embedding_model.encode([question])[0].tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    retrieved_results = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, metadata, distance in zip(documents, metadatas, distances):
        retrieved_results.append({
            "source": metadata["source"],
            "page": metadata["page"],
            "text": doc,
            "distance": round(distance, 3)
        })

    return retrieved_results


def generate_answer_from_uploaded_docs(question, collection):
    results = search_uploaded_documents(
        question=question,
        collection=collection,
        n_results=2
    )

    context = ""

    for result in results:
        source = result["source"]
        page = result["page"]
        text = result["text"][:1200]

        context += f"\nSource: {source}, Page: {page}\n{text}\n"

    prompt = f"""
You are a helpful Document Intelligence Assistant.

Answer the user's question using ONLY the uploaded PDF context below.
Do not use outside knowledge.
Do not make up information.

Question:
{question}

Context:
{context}

Instructions:
- Give a short and clear answer.
- Mention the source PDF name.
- Mention the page number.
- If the answer is not available in the uploaded documents, say:
"I could not find this information in the uploaded documents."
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=250
    )

    answer = response.choices[0].message.content

    sources = []

    for result in results:
        sources.append({
            "source": result["source"],
            "page": result["page"],
            "distance": result["distance"]
        })

    return answer, sources