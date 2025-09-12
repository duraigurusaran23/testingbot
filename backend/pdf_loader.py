from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdfs(pdf_dir: str, progress_callback=None):
    """
    Load PDFs with optimized chunking for faster processing
    """
    all_texts = []
    pdf_path = Path(pdf_dir)

    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"DEBUG: Found {len(pdf_files)} PDF files in {pdf_dir}")

    if not pdf_files:
        print("DEBUG: No PDF files found!")
        return []

    for i, file in enumerate(pdf_files):
        print(f"DEBUG: Processing file: {file.name}")
        reader = PdfReader(file)
        text = ""

        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            print(f"DEBUG: Page {page_num + 1} extracted {len(page_text)} characters")
            if page_text.strip():  
                text += page_text + "\n"

        print(f"DEBUG: Total text extracted from {file.name}: {len(text)} characters")
        if text.strip():  
            all_texts.append(text.strip())
            print(f"DEBUG: Added text from {file.name} to processing list")
        else:
            print(f"DEBUG: No text extracted from {file.name}")

        if progress_callback:
            progress_callback((i + 1) / len(pdf_files), f"Processing {file.name}...")

    print(f"DEBUG: Total documents to chunk: {len(all_texts)}")

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  
        chunk_overlap=100,  
        separators=["\n\n", "\n", ". ", ": ", " ", ""]  
    )

    chunks = []
    for doc in all_texts:
        doc_chunks = splitter.split_text(doc)
        print(f"DEBUG: Document split into {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)

    print(f"DEBUG: Total chunks created: {len(chunks)}")
    if chunks:
        print(f"DEBUG: First chunk preview: {chunks[0][:200]}...")

    return chunks
