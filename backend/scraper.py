#!/usr/bin/env python3
"""
Website Scraper for RAG Chatbot
Scrapes all internal pages of a website and saves as PDF
"""

import os
import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from urllib.parse import urljoin, urlparse
from collections import deque

class WebsiteScraper:
    def __init__(self, pdf_dir: str = "../pdfs", max_pages: int = 50):
        self.pdf_dir = pdf_dir
        self.max_pages = max_pages  # avoid huge sites
        os.makedirs(pdf_dir, exist_ok=True)

    def scrape_page(self, url: str) -> str:
        """Scrape a single page"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
            }
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            # Remove script and style
            for script in soup(["script", "style", "noscript"]):
                script.decompose()

            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)

            return text
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            return ""

    def crawl_website(self, start_url: str, progress_callback=None, stop_check=None) -> str:
        """Crawl all internal pages up to max_pages"""
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc

        visited = set()
        queue = deque([start_url])
        all_text = []

        total_pages = min(self.max_pages, 50)  # estimate

        while queue and len(visited) < self.max_pages:
            if stop_check and stop_check():
                break

            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            progress = len(visited) / total_pages * 100
            if progress_callback:
                progress_callback(progress, f"Scraping page {len(visited)}/{total_pages}: {url}")

            print(f"Scraping: {url}")
            page_text = self.scrape_page(url)
            if page_text:
                all_text.append(f"\n\n--- Page: {url} ---\n\n{page_text}")

            # Extract links
            try:
                soup = BeautifulSoup(requests.get(url).content, 'lxml')
                for link in soup.find_all("a", href=True):
                    new_url = urljoin(url, link["href"])
                    parsed_new = urlparse(new_url)
                    if parsed_new.netloc == base_domain and new_url not in visited:
                        if new_url.startswith("http"):
                            queue.append(new_url)
            except:
                pass

        if progress_callback:
            progress_callback(100, "Scraping complete")

        return "\n".join(all_text)

    def save_as_pdf(self, text: str, url: str, filename: str = None):
        """Save scraped text as PDF"""
        if not filename:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace('www.', '').replace('.', '_')
            filename = f"{domain}_scraped.pdf"

        pdf_path = os.path.join(self.pdf_dir, filename)

        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        content_style = ParagraphStyle('Content', parent=styles['Normal'], fontSize=10, leading=12)

        story = []
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=14)
        story.append(Paragraph(f"Scraped Content from: {url}", title_style))
        story.append(Spacer(1, 0.25*inch))

        for para in text.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), content_style))
                story.append(Spacer(1, 0.1*inch))

        doc.build(story)
        print(f"Saved PDF: {pdf_path}")
        return pdf_path

    def scrape_and_save(self, url: str, progress_callback=None, stop_check=None) -> str:
        """Full workflow: crawl + save"""
        text = self.crawl_website(url, progress_callback, stop_check)
        return self.save_as_pdf(text, url)

    def scrape_to_chunks(self, url: str, progress_callback=None, stop_check=None) -> list:
        """Scrape website and return text chunks directly (faster than PDF)"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text = self.crawl_website(url, progress_callback, stop_check)

        # Split text into chunks (optimized for speed)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced for faster processing
            chunk_overlap=50,  # Reduced overlap
            separators=["\n\n", "\n", ". ", ": ", " ", ""]
        )

        chunks = splitter.split_text(text)
        return chunks

if __name__ == "__main__":
    scraper = WebsiteScraper(max_pages=20)  # adjust pages as needed
    test_url = "https://example.com"
    pdf_path = scraper.scrape_and_save(test_url)
    print(f"Completed: {pdf_path}")
