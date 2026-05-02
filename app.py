import streamlit as st
from src.rag_pipeline_chroma import generate_answer_chroma
from src.upload_rag_pipeline import (
    build_uploaded_chroma_database,
    generate_answer_from_uploaded_docs
)


st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📄",
    layout="wide"
)


st.markdown(
    """
    <style>
    .title-text {
        font-size: 42px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 5px;
    }

    .subtitle-text {
        font-size: 18px;
        color: #475569;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0px 4px 12px rgba(15, 23, 42, 0.08);
        text-align: center;
    }

    .metric-number {
        font-size: 28px;
        font-weight: 800;
        color: #2563eb;
    }

    .metric-label {
        font-size: 14px;
        color: #64748b;
    }

    .source-card {
        background-color: #ffffff;
        padding: 14px 18px;
        border-radius: 12px;
        border-left: 5px solid #2563eb;
        box-shadow: 0px 2px 8px rgba(15, 23, 42, 0.06);
        margin-bottom: 10px;
    }

    .small-note {
        color: #64748b;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_collection" not in st.session_state:
    st.session_state.uploaded_collection = None

if "uploaded_pages" not in st.session_state:
    st.session_state.uploaded_pages = 0

if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = 0


with st.sidebar:
    st.title("⚙️ Assistant Settings")

    app_mode = st.radio(
        "Choose knowledge source:",
        [
            "Built-in university PDFs",
            "Upload custom PDFs"
        ]
    )

    st.markdown("---")

    if app_mode == "Built-in university PDFs":
        st.markdown("### Knowledge Base")
        st.success("University PDF Collection")
        st.write("15 PDF documents")
        st.write("455 pages processed")
        st.write("ChromaDB vector database")

    else:
        st.markdown("### Upload PDFs")

        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.write(f"Uploaded files: {len(uploaded_files)}")

            if st.button("Build Knowledge Base"):
                with st.spinner("Reading PDFs, creating embeddings, and building ChromaDB..."):
                    collection, pages, chunks = build_uploaded_chroma_database(uploaded_files)

                    st.session_state.uploaded_collection = collection
                    st.session_state.uploaded_pages = pages
                    st.session_state.uploaded_chunks = chunks
                    st.session_state.messages = []

                if collection is not None:
                    st.success("Uploaded PDF knowledge base created successfully.")
                else:
                    st.error("No readable text found in the uploaded PDFs.")

        if st.session_state.uploaded_collection is not None:
            st.markdown("### Uploaded Knowledge Base")
            st.success("Ready")
            st.write(f"Pages processed: {st.session_state.uploaded_pages}")
            st.write(f"Chunks created: {st.session_state.uploaded_chunks}")

    st.markdown("---")

    st.markdown("### Model")
    st.info("GPT-4o-mini")

    st.markdown("### Retrieval")
    st.info("ChromaDB Vector Search")

    st.markdown("### Embeddings")
    st.info("Sentence Transformers")

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


st.markdown(
    '<div class="title-text">📄 Document Intelligence Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">A professional RAG chatbot for asking questions from PDF documents with source-grounded answers.</div>',
    unsafe_allow_html=True
)


if app_mode == "Built-in university PDFs":
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">15</div>
                <div class="metric-label">PDF Documents</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">455</div>
                <div class="metric-label">Pages Processed</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">998+</div>
                <div class="metric-label">Text Chunks</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">RAG</div>
                <div class="metric-label">AI Architecture</div>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{st.session_state.uploaded_pages}</div>
                <div class="metric-label">Uploaded Pages</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{st.session_state.uploaded_chunks}</div>
                <div class="metric-label">Created Chunks</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        status = "Ready" if st.session_state.uploaded_collection is not None else "Not Ready"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{status}</div>
                <div class="metric-label">Upload RAG Status</div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("---")


with st.expander("📌 How this assistant works"):
    st.write(
        """
        This application uses Retrieval-Augmented Generation.

        1. PDF documents are loaded.
        2. Text is extracted from each PDF page.
        3. The text is split into overlapping chunks.
        4. Each chunk is converted into a semantic embedding using Sentence Transformers.
        5. Embeddings are stored in ChromaDB.
        6. When the user asks a question, the system retrieves the most relevant chunks.
        7. GPT-4o-mini generates an answer using only the retrieved context.
        8. The app displays source PDF names and page numbers.
        """
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input("Ask a question from the selected PDF knowledge base...")


if question:
    if app_mode == "Upload custom PDFs" and st.session_state.uploaded_collection is None:
        st.warning("Please upload PDFs and click 'Build Knowledge Base' before asking questions.")
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant documents and generating answer..."):
                if app_mode == "Built-in university PDFs":
                    answer, sources = generate_answer_chroma(question)
                else:
                    answer, sources = generate_answer_from_uploaded_docs(
                        question,
                        st.session_state.uploaded_collection
                    )

            st.write(answer)

            st.markdown("### 📚 Retrieved Sources")

            for source in sources:
                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>Source:</strong> {source['source']}<br>
                        <strong>Page:</strong> {source['page']}<br>
                        <strong>Vector Distance:</strong> {source['distance']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                '<p class="small-note">Lower vector distance usually means the retrieved document is more relevant.</p>',
                unsafe_allow_html=True
            )

        source_text = "\n\nRetrieved Sources:\n"

        for source in sources:
            source_text += (
                f"- {source['source']}, Page {source['page']}, "
                f"Distance: {source['distance']}\n"
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer + source_text
            }
        )