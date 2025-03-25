import unittest
from vectorDB import load_and_chunk_text, SentenceTransformerEmbeddingFunction
import os

class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.sample_file = "test_articles.txt"
        with open(self.sample_file, "w", encoding="utf-8") as f:
            f.write("Title: Test Article\nContent: This is a test article. It contains multiple sentences.")

    def tearDown(self):
        if os.path.exists(self.sample_file):
            os.remove(self.sample_file)

    def test_load_and_chunk_text(self):
        chunks = load_and_chunk_text(self.sample_file, chunk_size=50)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIn("id", chunks[0])
        self.assertIn("text", chunks[0])
        self.assertIn("metadata", chunks[0])

    def test_embedding_function(self):
        embedding_function = SentenceTransformerEmbeddingFunction()
        embeddings = embedding_function(["This is a test sentence."])
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 1)
    
if __name__ == "__main__":
    unittest.main()
