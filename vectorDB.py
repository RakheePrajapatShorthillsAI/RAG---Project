# Install required packages if needed
# pip install chromadb sentence-transformers nltk

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import re
from nltk.tokenize import sent_tokenize

# Download NLTK data for sentence splitting
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()

def load_and_chunk_text(file_path, chunk_size=500):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split articles using your actual delimiter (=== with title/URL pattern)
    articles = re.split(r'\n===+?\n', text)
    
    chunks = []
    for art_idx, article in enumerate(articles):
        article = article.strip()
        if not article:
            continue

        # Split into sentences using NLTK
        sentences = sent_tokenize(article)
        
        # Create chunks preserving sentence boundaries
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent_length = len(sent)
            
            if current_length + sent_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "id": f"art{art_idx}_chunk{len(chunks)}",
                    "text": ' '.join(current_chunk),
                    "metadata": {"article_id": art_idx}
                })
                current_chunk = []
                current_length = 0
                
            current_chunk.append(sent)
            current_length += sent_length
        
        # Add remaining sentences
        if current_chunk:
            chunks.append({
                "id": f"art{art_idx}_chunk{len(chunks)}",
                "text": ' '.join(current_chunk),
                "metadata": {"article_id": art_idx}
            })
    
    return chunks

# Initialize Chroma client and collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="ml_articles",
    embedding_function=SentenceTransformerEmbeddingFunction()
)

# Load and process data
documents = load_and_chunk_text("cleaned_ml_articles.txt")

# Add to database
collection.add(
    documents=[doc["text"] for doc in documents],
    metadatas=[doc["metadata"] for doc in documents],
    ids=[doc["id"] for doc in documents]
)

print(f"Successfully stored {len(documents)} chunks in vector DB")
# Check stored documents
print("Total chunks:", collection.count())

# Test similarity search
results = collection.query(
    query_texts=["What is neural networks?"],
    n_results=3
)

print("Top matching chunks:")
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"\nArticle {meta['article_id']}:")
    print(doc[:200] + "...")