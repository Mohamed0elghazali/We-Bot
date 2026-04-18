import fitz  
import docx
from pathlib import Path
from bs4 import BeautifulSoup

def extract_text_from_file(file_path: str, tokens_limit=10_000):
    # 1. Get extension
    ext = Path(file_path).suffix.lower().lstrip('.')
    text = ""

    # 2. Parse based on type
    if ext == "pdf":
        with fitz.open(file_path) as doc:
            # Efficiently join text from all pages
            text = chr(12).join([page.get_text() for page in doc])
            
    elif ext == "docx":
        doc_obj = docx.Document(file_path)
        text = "\n".join([para.text for para in doc_obj.paragraphs])
        
    elif ext == "html":
        with open(file_path, "r", encoding="utf-8") as f:
            text = BeautifulSoup(f, "html.parser").get_text(separator="\n")
            
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    else:
        print(f"[WARNING] Unsupported format: {file_path}")
    
    # 3. Count Words & Tokens
    word_count = len(text.split())
    token_est = int(word_count * 1.35) # Rough estimate for safety

    if token_est > tokens_limit:
        print(f"[WARNING] {file_path} is large ({token_est} tokens). Consider splitting.")
        char_limit = tokens_limit * 4  # Assuming 1 token ≈ 4 characters
        tokens_splits = (char_limit // 2) - 500 # 500 char buffer
        text = text[:tokens_splits] + "\n...[omitted]...\n" + text[-tokens_splits:]  

    return text

if __name__ == "__main__":
    test_files = [
        r"tmp\test_1.txt",
        r"tmp\test_2.docx",
        r"tmp\test_3.pdf",
        r"tmp\test_4.html"
    ]

    for i in test_files:
        print(extract_text_from_file(i))