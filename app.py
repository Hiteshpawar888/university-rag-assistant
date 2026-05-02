import streamlit as st
from src.rag_pipeline import generate_answer_low_cost


st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="centered"
)


st.title("🎓 University AI Assistant using RAG")

st.write(
    "Ask questions about university PDF documents. "
    "This assistant retrieves relevant information from PDFs and answers with sources."
)


question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            answer, sources = generate_answer_low_cost(question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Sources")
        for source in sources:
            st.write(
                f"- **{source['source']}**, Page {source['page']} "
                f"(Score: {source['score']})"
            )