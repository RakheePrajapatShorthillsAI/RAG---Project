import streamlit as st
from mainRAG import qa_pipeline  # Import your RAG pipeline
import os
import datetime
from datetime import datetime  # Correct import

import json
# Streamlit UI

# Optional: Display query log
def log_query_answer(query, answer, log_file="query_log.json"):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": query,
        "answer": answer
    }
    
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
                if not isinstance(logs, list):
                    logs = []
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
    
    logs.append(log_entry)
 
    with open(log_file, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=4, ensure_ascii=False)



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
        log_query_answer(query,answer)
    else:
        st.warning("Please enter a question.")

