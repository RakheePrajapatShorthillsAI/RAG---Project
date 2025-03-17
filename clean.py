# import re

# # Input and output file names
# input_file = "wikipedia_articles.txt"
# output_file = "cleaned_wikipedia_articles_for_rag.txt"

# # Function to clean the text
# def clean_text_for_rag(text):
#     # Remove LaTeX-style formulas: $E=mc^2$, \( x^2 \), \[ equation \]
#     text = re.sub(r'\$.*?\$', '', text)  # Inline LaTeX formulas
#     text = re.sub(r'\\\(.*?\\\)', '', text)  # Equations in \( \)
#     text = re.sub(r'\\\[.*?\\\]', '', text)  # Equations in \[ \]

#     # Remove displaystyle, mathematical notations, and symbols
#     text = re.sub(r'\b(displaystyle|sum|prod|int|frac|sqrt|log|lim|sin|cos|tan|theta|lambda|sigma|pi|alpha|beta|gamma|delta|omega|xi|zeta|eta|mu|nu|phi|psi|chi|kappa|rho|tau|upsilon)\b', '', text, flags=re.IGNORECASE)

#     # Remove parenthetical math-related expressions
#     text = re.sub(r'\([^()]*[=+\-*/^><][^()]*\)', '', text)  # Parentheses containing math-like content

#     # Remove special characters except normal punctuation
#     text = re.sub(r'[^\w\s.,!?;:]', '', text)

#     # Remove multiple spaces and excessive newlines
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
#     text = re.sub(r'\n+', '\n', text)  # Remove excessive newlines

#     return text.strip()

# # Read, clean, and save the text
# with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         cleaned_line = clean_text_for_rag(line)
#         if cleaned_line.strip():  # Avoid writing empty lines
#             outfile.write(cleaned_line + "\n")

# print("✅ Cleaning complete! Saved as 'cleaned_wikipedia_articles_for_rag.txt'")


# # Read the original file and clean its content
# with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         cleaned_line = clean_text_for_rag(line)
#         if cleaned_line.strip():  # Avoid empty lines
#             outfile.write(cleaned_line + "\n")

# print("✅ Cleaning complete! Saved as 'cleaned_wikipedia_articles.txt'")



# import pandas as pd
# import re

# # Load the CSV file
# input_file = "wikipedia_articles.txt"   # Replace with your actual file
# output_file = "cleaned_file_new.txt"

# # Read CSV as text
# df = pd.read_csv(input_file, dtype=str)

# # Function to clean each cell
# def clean_text(cell):
#     if isinstance(cell, str):  
#         cell = re.sub(r"=[^=]+\([^)]*\)", "", cell)  # Remove Excel formulas
#         cell = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", cell)  # Keep only text & punctuation
#         cell = re.sub(r"\s+", " ", cell).strip()  # Remove extra spaces
#     return cell

# # Apply cleaning to all cells
# df = df.applymap(clean_text)

# # Add proper line numbers
# df.insert(0, "Line Number", range(1, len(df) + 1))

# # Save the cleaned CSV
# df.to_csv(output_file, index=False)

# print(f"Cleaned CSV with line numbers saved as {output_file}")



# import re

# # Input and output file names
# input_file = "wikipedia_articles.txt"
# output_file = "cleaned_wikipedia_articles.txt"

# # Function to clean text
# def clean_text(text):
#     # Remove LaTeX-style formulas (e.g., $E=mc^2$ or \( x^2 \))
#     text = re.sub(r'\$.*?\$', '', text)  # Removes inline LaTeX formulas
#     text = re.sub(r'\\\(.*?\\\)', '', text)  # Removes LaTeX equations
    
#     # Remove extra spaces and newlines
#     text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

#     # Remove any text inside brackets (like references: [1], [2])
#     text = re.sub(r'\[.*?\]', '', text)

#     # Remove special characters except basic punctuation
#     text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)

#     return text.strip()

# # Read the original file and clean its content
# with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         cleaned_line = clean_text(line)
#         if cleaned_line.strip():  # Avoid empty lines
#             outfile.write(cleaned_line + "\n")

# print("✅ Cleaning complete! Saved as 'cleaned_wikipedia_articles.txt'")


import re

def clean_text(text: str) -> str:
    """Cleans text by removing formulas, LaTeX expressions, citations, and unnecessary symbols while preserving meaningful content."""

    # ✅ Remove inline LaTeX formulas like $E=mc^2$, $$\sum_{i=1}^{n} x_i$$
    text = re.sub(r"\$\$.*?\$\$", "", text)  # Remove block formulas
    text = re.sub(r"\$.*?\$", "", text)  # Remove inline LaTeX formulas

    # ✅ Remove citations like [1], [23], etc.
    text = re.sub(r"\[\d+\]", "", text)

    # ✅ Remove section headers like "== Introduction =="
    text = re.sub(r"==+.*?==+", "", text)

    # ✅ Remove excessive newlines and spaces
    text = re.sub(r"\n\s*\n", "\n", text)  # Remove multiple newlines
    text = re.sub(r"\s+", " ", text)  # Normalize spaces

    # ✅ Remove standalone equations like "E = mc^2", "F(x) = ax + b"
    text = re.sub(r"\b[A-Za-z]+\s*=\s*[^.\n]+\b", "", text)

    # ✅ Remove programming-like syntax (e.g., "f(x) -> y", "def function()")
    text = re.sub(r"\b[a-zA-Z_]+\([a-zA-Z0-9_, ]*\)\b", "", text)

    # ✅ Remove special character clusters (non-standard symbols)
    text = re.sub(r"[^\w\s.,!?()/-]", "", text)  # Keep punctuation for readability

    return text.strip()

def clean_file(input_file: str, output_file: str):
    """Reads input file, cleans text, and saves to output file."""
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned_text = clean_text(text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"✅ Cleaned text saved to {output_file}")

if __name__ == "__main__":
    input_file = "ml_articles.txt"  # Original file with raw Wikipedia data
    output_file = "cleaned_ml_articles.txt"  # New cleaned file
    clean_file(input_file, output_file)
