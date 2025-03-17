import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import nltk
import os

nltk.download('punkt')

# ------------------ Persistent Vector DB Setup ------------------
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name  # Track model name for verification
        
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

# Use persistent storage
chroma_client = chromadb.PersistentClient(path="vector_db")
embedding_function = SentenceTransformerEmbeddingFunction()

# Get or create collection with proper embedding function
collection = chroma_client.get_or_create_collection(
    name="ml_articles",
    embedding_function=embedding_function
)

# ------------------ Verify Data Loading ------------------
def check_vector_db():
    """Ensure data is properly loaded in the vector DB"""
    if collection.count() == 0:
        print("Loading data into Vector DB...")
        from vectorDB import load_and_chunk_text  # Import your data loading function
        documents = load_and_chunk_text("cleaned_ml_articles.txt")
        collection.add(
            documents=[doc["text"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents],
            ids=[doc["id"] for doc in documents]
        )
        print(f"Loaded {collection.count()} chunks")

check_vector_db()  # Runs on startup

# ------------------ LLM Setup ------------------
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=500,
    temperature=0.7,
)

# ------------------ Improved Retrieval & Generation ------------------
def retrieve_relevant_text(query, n_results=3):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        if not results['documents'][0]:
            return "No relevant context found", []
            
        context = "\n\n".join([
            f"Source {i+1} (Article {meta['article_id']}): {doc}"
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]))
        ])
        return context, results['documents'][0]
    except Exception as e:
        return f"Retrieval error: {str(e)}", []

# ------------------ Improved Generation with Formatting ------------------
def generate_answer(query, context):
    if "No relevant context" in context or "error" in context.lower():
        return "I couldn't find relevant information in the knowledge base, but here's a general explanation:\n"
    
    # prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # You are a machine learning professor. Answer conciselyto user's queries. 
    # #Structure your answer with core definition, key components and main characteristics.

    
    # Base your answer primarily on this context:<|eot_id|>
    
    # <|start_header_id|>context<|end_header_id|>
    # {context}<|eot_id|>
    
    # <|start_header_id|>user<|end_header_id|>
    # {query}<|eot_id|>
    
    # <|start_header_id|>assistant<|end_header_id|>"""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert in machine learning. Provide a **concise** answer to the user's query using the given context. Keep the response in a **single paragraph**, avoiding bullet points, introductions, or disclaimers.

    Base your answer on the following context:<|eot_id|>

    <|start_header_id|>context<|end_header_id|>
    {context}<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    {query}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>"""

    
    outputs = generator(
        prompt,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Keeps balance between creativity and factuality
        top_k=50,
        repetition_penalty=1.1
    )
    return outputs[0]['generated_text'].strip()

# ------------------ Enhanced Pipeline ------------------
def qa_pipeline(query):
    # Retrieve relevant context
    context, source_docs = retrieve_relevant_text(query)
    word_count = len(context.split()) if source_docs else 0
    
    # Generate answer
    answer = generate_answer(query, context)
    
    # Display results
    # print(f"\nRetrieved {word_count} words of relevant context from {len(source_docs)} sources")
    print("\n=== Question ===")
    print(query)
    print("\n=== Answer ===")
    print(answer)
    print("\n=== Source Chunks ===")
    print(context[:1500] + "..." if context else "No sources retrieved")
    return answer, context 
    
# ------------------ Usage ------------------
if __name__ == "__main__":
    query = "What is supervised machine learning?"
    qa_pipeline(query)


# import chromadb
# from chromadb import Documents, EmbeddingFunction, Embeddings
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import re
# import nltk
# import os

# nltk.download('punkt')

# # ------------------ Persistent Vector DB Setup ------------------
# class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
#     def __init__(self, model_name='all-MiniLM-L6-v2'):
#         self.model = SentenceTransformer(model_name)
#         self.model_name = model_name  # Track model name for verification
        
#     def __call__(self, input: Documents) -> Embeddings:
#         return self.model.encode(input).tolist()

# # Use persistent storage
# chroma_client = chromadb.PersistentClient(path="vector_db")
# embedding_function = SentenceTransformerEmbeddingFunction()

# # Get or create collection with proper embedding function
# collection = chroma_client.get_or_create_collection(
#     name="ml_articles",
#     embedding_function=embedding_function
# )

# # ------------------ Verify Data Loading ------------------
# def check_vector_db():
#     """Ensure data is properly loaded in the vector DB"""
#     if collection.count() == 0:
#         print("Loading data into Vector DB...")
#         from vectorDB import load_and_chunk_text  # Import your data loading function
#         documents = load_and_chunk_text("cleaned_ml_articles.txt")
#         collection.add(
#             documents=[doc["text"] for doc in documents],
#             metadatas=[doc["metadata"] for doc in documents],
#             ids=[doc["id"] for doc in documents]
#         )
#         print(f"Loaded {collection.count()} chunks")

# check_vector_db()  # Runs on startup

# # ------------------ LLM Setup ------------------
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# generator = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device_map="auto",
#     max_new_tokens=500,
#     temperature=0.7,
# )

# # ------------------ Improved Retrieval & Generation ------------------
# def retrieve_relevant_text(query, n_results=3):
#     try:
#         results = collection.query(
#             query_texts=[query],
#             n_results=n_results,
#             include=["documents", "metadatas"]
#         )
#         if not results['documents'][0]:
#             return "No relevant context found", []
            
#         context = "\n\n".join([
#             f"Source {i+1} (Article {meta['article_id']}): {doc}"
#             for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]))
#         ])
#         return context, results['documents'][0]
#     except Exception as e:
#         return f"Retrieval error: {str(e)}", []

# # ------------------ Updated Prompt for Test Case Matching ------------------
# def generate_answer(query, context):
#     if "No relevant context" in context or "error" in context.lower():
#         return "I couldn't find relevant information in the knowledge base, but here's a general explanation:\n"
    
#     prompt = f"""
#     Answer the following question concisely using the given machine learning content:

#     {context}

    
#     """
    
#     outputs = generator(
#         prompt,
#         return_full_text=False,
#         pad_token_id=tokenizer.eos_token_id,
#         temperature=0.7,
#         top_k=50,
#         repetition_penalty=1.1
#     )
#     return outputs[0]['generated_text'].strip()


# # ------------------ Enhanced Pipeline ------------------
# def qa_pipeline(query):
#     # Retrieve relevant context
#     context, source_docs = retrieve_relevant_text(query)
#     word_count = len(context.split()) if source_docs else 0
    
#     # Generate answer
#     answer = generate_answer(query, context)
    
#     # Display results
#     # print(f"\nRetrieved {word_count} words of relevant context from {len(source_docs)} sources")
#     print("\n=== Question ===")
#     print(query)
#     print("\n=== Answer ===")
#     print(answer)
#     # print("\n=== Source Chunks ===")
#     # print(context[:1500] + "..." if context else "No sources retrieved")

# # ------------------ Usage ------------------
# if __name__ == "__main__":
#     query = "What is unsupervised machine learning?"
#     qa_pipeline(query)

#usiimg minstral ai
# import chromadb
# from chromadb import Documents, EmbeddingFunction, Embeddings
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import re
# import nltk
# import os

# nltk.download('punkt')

# # ------------------ Persistent Vector DB Setup ------------------
# class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
#     def __init__(self, model_name='all-MiniLM-L6-v2'):
#         self.model = SentenceTransformer(model_name)
#         self.model_name = model_name  # Track model name for verification
        
#     def __call__(self, input: Documents) -> Embeddings:
#         return self.model.encode(input).tolist()

# # Use persistent storage
# chroma_client = chromadb.PersistentClient(path="vector_db")
# embedding_function = SentenceTransformerEmbeddingFunction()

# # Get or create collection with proper embedding function
# collection = chroma_client.get_or_create_collection(
#     name="ml_articles",
#     embedding_function=embedding_function
# )

# # ------------------ Verify Data Loading ------------------
# def check_vector_db():
#     """Ensure data is properly loaded in the vector DB"""
#     if collection.count() == 0:
#         print("Loading data into Vector DB...")
#         from vectorDB import load_and_chunk_text  # Import your data loading function
#         documents = load_and_chunk_text("cleaned_ml_articles.txt")
#         collection.add(
#             documents=[doc["text"] for doc in documents],
#             metadatas=[doc["metadata"] for doc in documents],
#             ids=[doc["id"] for doc in documents]
#         )
#         print(f"Loaded {collection.count()} chunks")

# check_vector_db()  # Runs on startup

# # ------------------ LLM Setup ------------------
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# generator = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device_map="auto",
#     max_new_tokens=150,
#     temperature=0.7,
#     top_k=50,
# )

# # ------------------ Improved Retrieval & Generation ------------------
# def retrieve_relevant_text(query, n_results=3):
#     try:
#         results = collection.query(
#             query_texts=[query],
#             n_results=n_results,
#             include=["documents", "metadatas"]
#         )
#         if not results['documents'][0]:
#             return "No relevant context found", []
            
#         context = "\n\n".join([
#             f"Source {i+1} (Article {meta['article_id']}): {doc}"
#             for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]))
#         ])
#         return context, results['documents'][0]
#     except Exception as e:
#         return f"Retrieval error: {str(e)}", []

# # ------------------ Concise Generation Matching Test Format ------------------
# def generate_answer(query, context):
#     if "No relevant context" in context or "error" in context.lower():
#         return "I couldn't find relevant information in the knowledge base."
    
#     prompt = f"""
#     Based on the following machine learning content,  provide a concise answer:
    
#     {context}
    
#     Format the response strictly as:
#     A: [Generated answer]
#     """
    
#     outputs = generator(
#         prompt,
#         return_full_text=False,
#         pad_token_id=tokenizer.eos_token_id,
#         temperature=0.3,
#         top_k=20,
#         repetition_penalty=1.1
#     )
#     return outputs[0]['generated_text'].strip()

# # ------------------ Enhanced Pipeline ------------------
# def qa_pipeline(query):
#     # Retrieve relevant context
#     context, source_docs = retrieve_relevant_text(query)
#     word_count = len(context.split()) if source_docs else 0
    
#     # Generate answer
#     answer = generate_answer(query, context)
    
#     # Display results
#     # print(f"\nRetrieved {word_count} words of relevant context from {len(source_docs)} sources")
#     print("\n=== Question ===")
#     print(query)
#     print("\n=== Answer ===")
#     print(answer)
#     # print("\n=== Source Chunks ===")
#     # print(context[:1500] + "..." if context else "No sources retrieved")

# # ------------------ Usage ------------------
# if __name__ == "__main__":
#     query = "What is supervised machine learning?"
#     qa_pipeline(query)
