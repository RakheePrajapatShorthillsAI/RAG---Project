# import unittest
# from WikiScraper import scrape_ml_articles, clean_text, save_to_txt

# class TestWikiScraper(unittest.TestCase):
#     def setUp(self):
#         self.sample_text = """
#         Machine learning is a field of artificial intelligence (AI) [1] that uses statistical techniques to give computer systems the ability to "learn" from data.
#         == History ==
#         Machine learning has evolved over time.
#         """
#         self.cleaned_text = "Machine learning is a field of artificial intelligence AI that uses statistical techniques to give computer systems the ability to learn from data. Machine learning has evolved over time."

#     def test_clean_text(self):
#         result = clean_text(self.sample_text)
#         self.assertEqual(result, self.cleaned_text)

#     def test_scrape_ml_articles(self):
#         articles = scrape_ml_articles(max_pages=2)
#         self.assertIsInstance(articles, list)
#         self.assertGreater(len(articles), 0)
#         self.assertIn("title", articles[0])
#         self.assertIn("url", articles[0])
#         self.assertIn("text", articles[0])

#     def test_save_to_txt(self):
#         articles = [
#             {"title": "Test Article", "url": "http://example.com", "text": "This is a test article."}
#         ]
#         save_to_txt(articles, filename="test_output.txt")
#         with open("test_output.txt", "r", encoding="utf-8") as f:
#             content = f.read()
#         self.assertIn("Test Article", content)
#         self.assertIn("http://example.com", content)
#         self.assertIn("This is a test article.", content)

# if __name__ == "__main__":
#     unittest.main()


import unittest
from WikiScraper import scrape_ml_articles, clean_text, save_to_txt

class TestWikiScraper(unittest.TestCase):
    def setUp(self):
        self.sample_text = """
        Machine learning is a field of artificial intelligence (AI) [1] that uses statistical techniques to give computer systems the ability to "learn" from data.
        == History ==
        Machine learning has evolved over time.
        """
        self.cleaned_text = "Machine learning is a field of artificial intelligence AI that uses statistical techniques to give computer systems the ability to learn from data. Machine learning has evolved over time."

    def test_clean_text(self):
        result = clean_text(self.sample_text)
        self.assertEqual(result, self.cleaned_text)

    def test_scrape_ml_articles(self):
        articles = scrape_ml_articles(max_pages=2)
        self.assertIsInstance(articles, list)
        self.assertGreater(len(articles), 0)
        self.assertIn("title", articles[0])
        self.assertIn("url", articles[0])
        self.assertIn("text", articles[0])

    def test_save_to_txt(self):
        articles = [
            {"title": "Test Article", "url": "http://example.com", "text": "This is a test article."}
        ]
        save_to_txt(articles, filename="test_output.txt")
        with open("test_output.txt", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Test Article", content)
        self.assertIn("http://example.com", content)
        self.assertIn("This is a test article.", content)

if __name__ == "__main__":
    unittest.main()
