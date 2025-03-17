import streamlit as st
from mainRAG import qa_pipeline  # Import your RAG pipeline
import os
# Streamlit UI
st.set_page_config(page_title="RAG-LLM Chatbot", layout="wide")

st.title("RAG-LLM Q&A System")
st.write("Ask questions based on the scraped data.")

# User input
query = st.text_area("Enter your question:", height=100)

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Generating response..."):
            answer, context = qa_pipeline(query)  # Call the pipeline
        st.subheader("Answer:")
        st.write(answer)
        # Display retrieved context/chunks
        st.subheader("Retrieved Context:")
        st.text_area("Context Used for Answering", context, height=300, disabled=True)

    else:
        st.warning("Please enter a question.")

# Optional: Display query log
log_file = "query_log.txt"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        log_data = f.read()
    with st.expander("View Query Log"):
        st.text_area("Logged Interactions", log_data, height=200, disabled=True)

