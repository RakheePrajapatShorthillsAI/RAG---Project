
import wikipediaapi
import time
import json
import logging
import re
from typing import Set, List

# Initialize logging
logging.basicConfig(filename='scraping.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MLScraper/1.0 (rakheeprajapat2002@gmail.com)',  # Identify your bot
    extract_format=wikipediaapi.ExtractFormat.WIKI  # Clean text format
)

# Target categories
TARGET_CATEGORIES = [
    "Machine_learning",  # Root category
    "Machine_learning_algorithms",
    "Generative_adversarial_networks",
    "Convolutional_neural_networks",
    "Reinforcement_learning",
    "Recurrent_neural_networks",
    "Transformer_(machine_learning_model)",  # Correct category name
]



# Keywords to exclude
EXCLUDE_KEYWORDS = ["History of", "Timeline", "List of", "Comparison", "Applications of"]

# Store scraped data
seen_titles: Set[str] = set()
articles = []

def scrape_ml_articles(max_pages: int = 200) -> List[dict]:
    """Scrape Wikipedia articles from given categories."""
    
    def process_category(category_name: str):
        """Recursively scrape pages from a category and its subcategories."""
        category = wiki.page(f"Category:{category_name}")
        
        # ✅ FIXED: Use manual URL instead of `category.fullurl`
        print(f"Checking category: https://en.wikipedia.org/wiki/Category:{category_name}")  

        if not category.exists():
            logging.warning(f"Category {category_name} not found.")
            print(f"⚠️ Category {category_name} not found!")
            return

        for member in category.categorymembers.values():
            if len(articles) >= max_pages:
                return
            
            if member.ns in [wikipediaapi.Namespace.MAIN, wikipediaapi.Namespace.CATEGORY]:  
                if member.title not in seen_titles and not any(kw in member.title for kw in EXCLUDE_KEYWORDS):
                    seen_titles.add(member.title)
                    articles.append({
                        "title": member.title,
                        "url": member.fullurl,  # This is valid only for articles
                        "text": clean_text(member.text)
                    })
                    logging.info(f"Scraped: {member.title}")
                    print(f"✅ Scraped: {member.title}")  # Debugging
                    time.sleep(1.5)  # Increase delay to prevent API blocks

                # Process subcategories recursively
                if member.ns == wikipediaapi.Namespace.CATEGORY:
                    process_category(member.title.split(":")[1])  

    # Scrape all target categories
    for category in TARGET_CATEGORIES:
        process_category(category)
        if len(articles) >= max_pages:
            break

    return articles[:max_pages]  # Return up to max_pages articles

def clean_text(text: str) -> str:
    """Remove citations, formulas, and section headers from text."""
    text = re.sub(r"\[\d+\]", "", text)  # Remove citations [1], [2], etc.
    text = re.sub(r"==+.*?==+", "", text)  # Remove section headers
    text = re.sub(r"\n\s*\n", "\n", text)  # Remove excessive newlines
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"\$\$.*?\$\$", "", text)  # Remove LaTeX formulas
    text = re.sub(r"\b[A-Za-z]+\d+\b", "", text)  # Remove variable-like words (e.g., X1, Y2)
    return text.strip()

def save_to_txt(articles: List[dict], filename: str = "ml_articles.txt"):
    """Save scraped data to a .txt file."""
    with open(filename, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(f"Title: {article['title']}\n")
            f.write(f"URL: {article['url']}\n")
            f.write("Content:\n")
            f.write(article["text"] + "\n\n")
            f.write("="*80 + "\n\n")  # Separator for readability

    logging.info(f"Saved {len(articles)} articles to {filename}.")
    print(f"✅ Data saved to {filename}!")

if __name__ == "__main__":
    # Scrape 150-200 pages
    articles = scrape_ml_articles(max_pages=200)
    save_to_txt(articles)
    print(f"✅ Scraped {len(articles)} articles. Check ml_articles.txt!")
