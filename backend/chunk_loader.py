
"""
Chunk Loader for RAG Chatbot
Loads pre-processed chunks from disk for faster app startup
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from preprocess import PDFProcessor

class ChunkLoader:
    def __init__(self, processed_data_dir: str = "../processed_data"):
        self.processed_data_dir = Path(processed_data_dir)

    def load_chunks(self) -> Optional[List[str]]:
        """Load pre-processed chunks from disk"""
        try:
            pickle_file = self.processed_data_dir / "chunks.pkl"
            json_file = self.processed_data_dir / "chunks.json"

           
            if pickle_file.exists():
                print("Loading chunks from pickle file...")
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
            elif json_file.exists():
                print("Loading chunks from JSON file...")
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                print("No processed chunks found. Run preprocess.py first.")
                return None

            chunks = data['chunks']
            metadata = data['metadata']

            print("Chunks loaded successfully!")
            print(f"   Total chunks: {len(chunks)}")
            print(f"   Total characters: {metadata['total_characters']}")
            print(f"   Processed: {metadata['processing_date']}")
            if 'source_url' in metadata:
                print(f"   Source URL: {metadata['source_url']}")
            elif 'pdf_files' in metadata:
                print(f"   Source PDFs: {len(metadata['pdf_files'])}")

            return chunks

        except Exception as e:
            print(f"Error loading chunks: {e}")
            return None

    def get_chunk_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the loaded chunks"""
        try:
            pickle_file = self.processed_data_dir / "chunks.pkl"
            json_file = self.processed_data_dir / "chunks.json"

            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
            elif json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                return None

            metadata = data['metadata']
            stats = {
                "total_chunks": len(data['chunks']),
                "total_characters": metadata['total_characters'],
                "processing_date": metadata['processing_date'],
                "avg_chunk_length": sum(len(chunk) for chunk in data['chunks']) / len(data['chunks']) if data['chunks'] else 0
            }

            # Handle both old PDF format and new URL format
            if 'source_url' in metadata:
                stats["source_url"] = metadata['source_url']
            elif 'pdf_files' in metadata:
                stats["pdf_files"] = len(metadata['pdf_files'])

            return stats

        except Exception as e:
            print(f"Error getting chunk stats: {e}")
            return None

def load_processed_chunks() -> Optional[List[str]]:
    """Convenience function to load chunks"""
    loader = ChunkLoader()
    return loader.load_chunks()

def get_chunk_statistics() -> Optional[Dict[str, Any]]:
    """Convenience function to get chunk statistics"""
    loader = ChunkLoader()
    return loader.get_chunk_stats()

if __name__ == "__main__":
    print("Testing Chunk Loader")
    print("=" * 30)

    chunks = load_processed_chunks()
    if chunks:
        print(f"Successfully loaded {len(chunks)} chunks")
        print(f"First chunk preview: {chunks[0][:100]}...")

        stats = get_chunk_statistics()
        if stats:
            print("\nStatistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
    else:
        print("Failed to load chunks")
        print("Run 'python preprocess.py' first to process your PDFs")