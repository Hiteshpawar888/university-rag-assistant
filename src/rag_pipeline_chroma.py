import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb


load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "university_docs"

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

client = OpenAI(api_key=api_key)

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

print("Chroma RAG pipeline ready.")


def search_documents_chroma(question, n_results=2):
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


def generate_answer_chroma(question):
    results = search_documents_chroma(question, n_results=2)

    context = ""

    for result in results:
        source = result["source"]
        page = result["page"]
        text = result["text"][:1200]

        context += f"\nSource: {source}, Page: {page}\n{text}\n"

    prompt = f"""
You are a helpful University AI Assistant.

Answer the user's question using ONLY the context below.
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
- If the answer is not available in the context, say:
"I could not find this information in the provided university documents."
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