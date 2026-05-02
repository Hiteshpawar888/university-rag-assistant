from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

# OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# PDF folder path
PDF_FOLDER = Path("data/pdfs")


def load_pdfs():
    all_documents = []

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    for pdf_file in pdf_files:
        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text:
                all_documents.append({
                    "source": pdf_file.name,
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


# Load PDFs and build search system once
print("Loading PDFs...")
all_documents = load_pdfs()

print("Creating chunks...")
chunks = create_chunks(all_documents)

print("Creating TF-IDF search system...")
texts = [chunk["text"] for chunk in chunks]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(texts)

print("RAG pipeline ready.")
print("Total documents:", len(all_documents))
print("Total chunks:", len(chunks))


def search_documents_fast(question, n_results=2):
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-n_results:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "source": chunks[index]["source"],
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": similarities[index]
        })

    return results


def generate_answer_low_cost(question):
    results = search_documents_fast(question, n_results=2)

    context = ""

    for result in results:
        source = result["source"]
        page = result["page"]
        text = result["text"][:1000]

        context += f"\nSource: {source}, Page: {page}\n{text}\n"

    prompt = f"""
You are a helpful University AI Assistant.

Answer the question using ONLY the context below.
Do not use outside knowledge.

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
            "score": round(result["score"], 3)
        })

    return answer, sources