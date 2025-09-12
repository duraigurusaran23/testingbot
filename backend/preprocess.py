#!/usr/bin/env python3
"""
PDF Preprocessing Script for RAG Chatbot
Processes PDFs once and saves chunks to disk for faster loading
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

class PDFProcessor:
    def __init__(self, pdf_dir: str = "../pdfs", output_dir: str = "../processed_data"):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ": ", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a single PDF file"""
        print(f"[PDF] Processing: {pdf_path.name}")

        reader = PdfReader(pdf_path)
        text = ""

        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    # Handle encoding issues
                    try:
                        page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')
                    except:
                        page_text = page_text.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                    text += page_text + "\n"
                print(f"  Page {page_num + 1}: {len(page_text) if page_text else 0} characters")
            except Exception as e:
                print(f"  Page {page_num + 1}: Error extracting text - {e}")
                continue

        return text.strip()

    def process_all_pdfs(self) -> Dict[str, Any]:
        """Process all PDFs in the directory"""
        print("Scanning for PDF files...")

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.pdf_dir}")

        all_texts = []
        metadata = {
            "processing_date": datetime.now().isoformat(),
            "pdf_files": [],
            "total_chunks": 0,
            "total_characters": 0
        }

        
        for pdf_file in pdf_files:
            print(f"\nProcessing {pdf_file.name}...")
            text = self.extract_text_from_pdf(pdf_file)

            if text:
                all_texts.append(text)
                metadata["pdf_files"].append({
                    "filename": pdf_file.name,
                    "path": str(pdf_file),
                    "characters": len(text)
                })
                metadata["total_characters"] += len(text)
            else:
                print(f"Warning: No text extracted from {pdf_file.name}")

        if not all_texts:
            raise ValueError("No text could be extracted from any PDF files")

        
        print("\nCreating text chunks...")
        all_chunks = []
        for text in all_texts:
            chunks = self.splitter.split_text(text)
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks from document")

        metadata["total_chunks"] = len(all_chunks)

        print(f"\nProcessing complete!")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Total characters: {metadata['total_characters']}")

        return {
            "chunks": all_chunks,
            "metadata": metadata
        }

    def save_chunks(self, chunks: List[str], metadata: Dict[str, Any]):
        """Save chunks and metadata to disk"""
        
        chunks_file = self.output_dir / "chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": chunks,
                "metadata": metadata
            }, f, indent=2, ensure_ascii=False)

        
        pickle_file = self.output_dir / "chunks.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                "chunks": chunks,
                "metadata": metadata
            }, f)

        print(f"Chunks saved to:")
        print(f"   JSON: {chunks_file}")
        print(f"   Pickle: {pickle_file}")

    def load_chunks(self) -> Dict[str, Any]:
        """Load pre-processed chunks from disk"""
        pickle_file = self.output_dir / "chunks.pkl"
        json_file = self.output_dir / "chunks.json"

        
        if pickle_file.exists():
            print(f"Loading chunks from pickle: {pickle_file}")
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
        elif json_file.exists():
            print(f"Loading chunks from JSON: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"No processed chunks found in {self.output_dir}")

        print(f"Loaded {data['metadata']['total_chunks']} chunks")
        return data

    def is_processed_data_fresh(self) -> bool:
        """Check if processed data exists and is up-to-date"""
        pickle_file = self.output_dir / "chunks.pkl"
        json_file = self.output_dir / "chunks.json"

        if not (pickle_file.exists() or json_file.exists()):
            return False

        
        try:
            data = self.load_chunks()
            processing_date = datetime.fromisoformat(data['metadata']['processing_date'])

            
            for pdf_info in data['metadata']['pdf_files']:
                pdf_path = Path(pdf_info['path'])
                if pdf_path.exists() and pdf_path.stat().st_mtime > processing_date.timestamp():
                    print(f"{pdf_path.name} has been modified since last processing")
                    return False

            return True
        except Exception as e:
            print(f"Warning: Error checking data freshness: {e}")
            return False

def main():
    """Main function to run PDF preprocessing"""
    print("PDF Preprocessing Script")
    print("=" * 50)

    processor = PDFProcessor()

    try:
        
        if processor.is_processed_data_fresh():
            print("Processed data is up-to-date!")
            choice = input("Reprocess anyway? (y/N): ").lower().strip()
            if choice != 'y':
                print("Loading existing processed data...")
                data = processor.load_chunks()
                print(f"Summary: {data['metadata']['total_chunks']} chunks from {len(data['metadata']['pdf_files'])} PDFs")
                return

        
        print("\nStarting PDF processing...")
        result = processor.process_all_pdfs()

        
        print("\nSaving processed data...")
        processor.save_chunks(result['chunks'], result['metadata'])

        print("\nPreprocessing complete!")
        print(f"   Output directory: {processor.output_dir}")
        print(f"   Total chunks: {result['metadata']['total_chunks']}")
        print(f"   Total characters: {result['metadata']['total_characters']}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()