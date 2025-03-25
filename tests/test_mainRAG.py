import unittest
from mainRAG import retrieve_relevant_text, generate_answer

class TestMainRAG(unittest.TestCase):
    def setUp(self):
        # Mock context and query for testing
        self.query = "What is supervised learning?"
        self.context = """
            Supervised learning is a type of machine learning where models are trained on labeled data.
            Examples include classification and regression tasks.
            """

    def test_retrieve_relevant_text(self):
        context, documents = retrieve_relevant_text(query=self.query, n_results=1)
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)

    def test_generate_answer_with_context(self):
        answer = generate_answer(query=self.query, context=self.context)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_generate_answer_without_context(self):
        answer = generate_answer(query=self.query, context="No relevant context found")
        self.assertIsInstance(answer, str)
    
if __name__ == "__main__":
    unittest.main()
