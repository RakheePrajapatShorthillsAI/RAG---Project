
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import textwrap
from tqdm import tqdm
import time

# Set your Mistral AI API key
MISTRAL_API_KEY = "hFfAFziygUsWelnWgLwv9I764x3kSP5g"  # Replace with your actual API key

# Initialize Mistral Client with a timeout
client = MistralClient(api_key=MISTRAL_API_KEY)

# Function to load scraped data
def load_scraped_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to generate both question and answer
def generate_test_case(context):
    prompt = f"""
    Based on the following machine learning content, generate a test question and provide a detailed answer:

    {context}

    Format the response strictly as:
    Q: [Generated question]
    A: [Generated answer]
    """

    try:
        response = client.chat(
            model="ministral-3b-latest",  # Ensure correct model name
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.7,
            max_tokens=150
        )

        # Extract generated text
        generated_text = response.choices[0].message.content.strip()

        # Parse question and answer
        lines = generated_text.split("\n")
        question, answer = "", ""
        for line in lines:
            if line.startswith("Q:"):
                question = line[2:].strip()
            elif line.startswith("A:"):
                answer = line[2:].strip()

        return question, answer

    except Exception as e:
        print(f"⚠️ Error generating test case: {e}")
        return None, None  # Return empty values on failure

# Function to split text into chunks
def split_into_chunks(text, chunk_size=500):
    return textwrap.wrap(text, chunk_size)

# Load and process data
scraped_text = load_scraped_data("cleaned_ml_articles.txt")
chunks = split_into_chunks(scraped_text, 500)

# Generate test cases with error handling and progress saving
test_cases = []
output_file = "generated_test_cases_new.txt"

# Start fresh to ensure all test cases include context
start_index = 0
print(f"▶️ Starting from index {start_index}")

with open(output_file, "w", encoding="utf-8") as f:  # Overwrite existing file
    for i in tqdm(range(start_index, min(start_index + 1100, len(chunks)))):
        question, answer = generate_test_case(chunks[i])
        if question and answer:
            f.write(f"{i+1}. Context: {chunks[i]}\nQ: {question}\nA: {answer}\n\n")  # Numbering questions and saving context

print("✅ 1100 Question-Answer pairs (with context) generated and saved in generated_test_cases_new.txt")
