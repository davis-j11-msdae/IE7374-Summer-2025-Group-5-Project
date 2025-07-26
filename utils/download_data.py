#!/usr/bin/env python3
"""
Data download script for the Personalized Storytelling System.
Downloads required texts from Project Gutenberg and the Mistral 7B model for training.
"""

import os
import sys
import logging
import requests
import time
from typing import List, Dict, Any
from urllib.parse import urljoin
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from helpers import set_cwd

# Get current working directory for path operations
cwd = set_cwd()

# Add utils to path for imports
sys.path.append(os.path.join(cwd, 'utils'))
from helpers import ensure_dir_exists, load_config

logger = logging.getLogger(__name__)


class ProjectGutenbergDownloader:
    """Downloads texts from Project Gutenberg based on bookshelf categories."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(cwd, "configs", "model_config.yaml")

        self.config = load_config(config_path)
        self.data_paths = self.config['paths']
        self.gutenberg_config = self.config.get('gutenberg', {})

        # Ensure directories exist
        ensure_dir_exists(self.data_paths['data_raw'])
        ensure_dir_exists(self.data_paths['models'])

        # Project Gutenberg configuration
        self.base_url = "https://www.gutenberg.org"
        self.bookshelf_url = f"{self.base_url}/ebooks/bookshelf"
        self.text_format = "txt.utf-8"

        # Rate limiting
        self.request_delay = 1.0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PersonalizedStorytelling/1.0 (Educational Research)'
        })

    def check_existing_data(self) -> Dict[str, Any]:
        """Check which data files and models already exist."""
        status = {
            'categories': {},
            'model_exists': False,
            'missing_categories': [],
            'existing_categories': []
        }

        # Check Project Gutenberg categories
        categories = self.gutenberg_config.get('categories', {})
        for category_name in categories.keys():
            file_name = f"{category_name}_stories.txt"
            file_path = os.path.join(self.data_paths['data_raw'], file_name)
            exists = os.path.exists(file_path) and os.path.getsize(file_path) > 1000

            status['categories'][category_name] = {
                'exists': exists,
                'path': file_path,
                'size_mb': os.path.getsize(file_path) / (1024 * 1024) if exists else 0
            }

            if exists:
                status['existing_categories'].append(category_name)
            else:
                status['missing_categories'].append(category_name)

        # Check Mistral 7B model
        model_path = os.path.join(self.data_paths['models'], 'mistral-7b-base')
        if os.path.exists(model_path) and os.listdir(model_path):
            status['model_exists'] = True
            model_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, dirnames, filenames in os.walk(model_path)
                             for filename in filenames)
            status['model_size_gb'] = model_size / (1024 ** 3)

        return status

    def get_bookshelf_books(self, bookshelf_id: int, min_downloads: int = 2000) -> List[Dict[str, Any]]:
        """Get list of books from a Project Gutenberg bookshelf with minimum download threshold."""
        print(f"  Fetching books from bookshelf {bookshelf_id} (min downloads: {min_downloads})")

        books = []
        page = 1

        while True:
            bookshelf_url = f"{self.bookshelf_url}/{bookshelf_id}?start_index={(page - 1) * 25}"

            time.sleep(self.request_delay)
            try:
                response = self.session.get(bookshelf_url, timeout=30)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"    Error fetching page {page}: {e}")
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            book_links = soup.find_all('li', class_='booklink')

            if not book_links:
                break

            page_books = []
            for link in book_links:
                title_link = link.find('a', class_='link')
                if not title_link:
                    continue

                book_url = title_link.get('href')
                if not book_url or not book_url.startswith('/ebooks/'):
                    continue

                book_id = book_url.split('/')[-1]
                title = title_link.get_text(strip=True)

                author_span = link.find('span', class_='subtitle')
                author = author_span.get_text(strip=True) if author_span else "Unknown"

                downloads_span = link.find('span', class_='extra')
                downloads = 0
                if downloads_span:
                    downloads_text = downloads_span.get_text()
                    downloads_match = re.search(r'(\d+)', downloads_text)
                    if downloads_match:
                        downloads = int(downloads_match.group(1))

                if downloads >= min_downloads:
                    page_books.append({
                        'id': book_id,
                        'title': title,
                        'author': author,
                        'downloads': downloads,
                        'url': urljoin(self.base_url, book_url)
                    })

            books.extend(page_books)
            page += 1

            if len(page_books) < 5:
                break

        books.sort(key=lambda x: x['downloads'], reverse=True)
        print(f"    Found {len(books)} qualifying books")
        return books

    def download_book_text(self, book_id: str) -> str:
        """Download plain text version of a book and return content."""
        text_urls = [
            f"{self.base_url}/files/{book_id}/{book_id}-0.txt",
            f"{self.base_url}/files/{book_id}/{book_id}.txt",
            f"{self.base_url}/ebooks/{book_id}.txt.utf-8"
        ]

        for url in text_urls:
            time.sleep(self.request_delay)
            try:
                response = self.session.get(url, stream=True, timeout=60)

                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text' in content_type or 'plain' in content_type or not content_type:
                        content = response.text
                        if self._validate_text_content(content):
                            return content
                elif response.status_code == 404:
                    continue

            except requests.RequestException:
                continue

        return None

    def _validate_text_content(self, content: str) -> bool:
        """Validate text content quality."""
        if not content or len(content) < 1000:
            return False

        if "project gutenberg" not in content.lower():
            return False

        # Check for reasonable text ratio
        sample = content[:2000]
        printable_ratio = sum(1 for c in sample if c.isprintable() or c.isspace()) / len(sample)
        if printable_ratio < 0.9:
            return False

        return True

    def clean_gutenberg_text(self, text: str) -> str:
        """Clean Project Gutenberg text by removing headers/footers and metadata."""
        lines = text.split('\n')

        # Find start of actual content
        start_idx = 0
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(marker in line_lower for marker in [
                "*** start of this project gutenberg",
                "*** start of the project gutenberg",
                "chapter i", "chapter 1", "prologue"
            ]):
                start_idx = i + 1
                break

        # Find end of actual content
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            line_lower = lines[i].lower().strip()
            if any(marker in line_lower for marker in [
                "*** end of this project gutenberg",
                "*** end of the project gutenberg",
                "end of project gutenberg"
            ]):
                end_idx = i
                break

        content_lines = lines[start_idx:end_idx]

        # Remove page numbers and metadata
        cleaned_lines = []
        for line in content_lines:
            line = line.strip()
            if re.match(r'^\d+$', line) or re.match(r'^page \d+', line.lower()):
                continue
            cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)

        return cleaned_text.strip()

    def download_category_books(self, category_name: str, bookshelf_id: int,
                                min_downloads: int = 2000) -> bool:
        """Download books from a specific category."""
        print(f"Downloading {category_name}...")

        books = self.get_bookshelf_books(bookshelf_id, min_downloads)

        if not books:
            print(f"  No books found for {category_name}")
            return False

        seen_books = set()
        successful_downloads = []

        with tqdm(total=len(books), desc=f"Processing {category_name}") as pbar:
            for book in books:
                book_id = book['id']
                title = book['title']
                author = book['author']

                # Simple deduplication
                normalized_title = re.sub(r'[^\w\s]', '', title.lower()).strip()
                normalized_author = re.sub(r'[^\w\s]', '', author.lower()).strip()
                book_signature = f"{normalized_title}_{normalized_author}"

                if book_signature in seen_books:
                    pbar.update(1)
                    continue

                seen_books.add(book_signature)

                # Download book content
                content = self.download_book_text(book_id)

                if content:
                    cleaned_content = self.clean_gutenberg_text(content)

                    if len(cleaned_content.split()) >= 1000:
                        successful_downloads.append({
                            'title': title,
                            'author': author,
                            'text': cleaned_content,
                            'id': book_id,
                            'downloads': book['downloads'],
                            'word_count': len(cleaned_content.split()),
                            'bookshelf_id': bookshelf_id
                        })

                pbar.update(1)
                pbar.set_postfix(downloaded=len(successful_downloads))

        if not successful_downloads:
            print(f"  No successful downloads for {category_name}")
            return False

        # Sort by popularity
        successful_downloads.sort(key=lambda x: x['downloads'], reverse=True)

        # Save combined file
        combined_file = os.path.join(self.data_paths['data_raw'],
                                     f"{category_name}_stories.txt")

        total_words = 0

        try:
            with open(combined_file, 'w', encoding='utf-8') as f:
                for i, book_data in enumerate(successful_downloads):
                    if i > 0:
                        f.write("\n\n" + "=" * 80 + "\n\n")

                    f.write(f"TITLE: {book_data['title']}\n")
                    f.write(f"AUTHOR: {book_data['author']}\n")
                    f.write(f"PROJECT_GUTENBERG_ID: {book_data['id']}\n")
                    f.write(f"DOWNLOADS: {book_data['downloads']}\n")
                    f.write(f"WORD_COUNT: {book_data['word_count']}\n")
                    f.write(f"BOOKSHELF_ID: {book_data['bookshelf_id']}\n")
                    f.write("\n" + book_data['text'])
                    total_words += book_data['word_count']

            print(f"  Saved {category_name}: {len(successful_downloads)} books, {total_words:,} words")
            print(f"  File: {combined_file}")
            return True

        except Exception as e:
            print(f"  Error saving {category_name}: {e}")
            return False

    def download_mistral_model(self) -> bool:
        """Download the Mistral 7B model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from dotenv import load_dotenv
        except ImportError as e:
            print(f"Required packages not available: {e}")
            return False

        load_dotenv()

        model_name = self.config['model']['base_model']
        model_path = os.path.join(self.data_paths['models'], 'mistral-7b-base')

        print(f"Downloading Mistral 7B model: {model_name}")
        ensure_dir_exists(model_path)

        hf_token = os.getenv('HF_TOKEN')

        try:
            print("  Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=True,
                token=hf_token
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            tokenizer.save_pretrained(model_path)
            print("    Tokenizer saved")

            print("  Downloading model (this may take 10-15 minutes)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=None,
                trust_remote_code=True,
                token=hf_token,
                low_cpu_mem_usage=True
            )

            model.save_pretrained(model_path)
            print("    Model saved")

            # Calculate total size
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)

            size_gb = total_size / (1024 ** 3)
            print(f"  Model download complete: {size_gb:.2f} GB")
            return True

        except Exception as e:
            print(f"  Model download failed: {e}")
            return False

    def download_missing_categories(self, missing_categories: List[str]) -> bool:
        """Download only missing categories."""
        if not missing_categories:
            print("All categories already exist")
            return True

        categories = self.gutenberg_config.get('categories', {})
        min_downloads = self.gutenberg_config.get('min_downloads', 2000)

        success_count = 0
        for category_name in missing_categories:
            if category_name in categories:
                config = categories[category_name]
                bookshelf_id = config['bookshelf_id']

                if self.download_category_books(category_name, bookshelf_id, min_downloads):
                    success_count += 1

        return success_count == len(missing_categories)


def display_status(status: Dict[str, Any]):
    """Display current download status."""
    print("\nCURRENT STATUS")
    print("=" * 50)

    # Categories status
    print("Project Gutenberg Categories:")
    if status['existing_categories']:
        print("  Existing:")
        for category in status['existing_categories']:
            size_mb = status['categories'][category]['size_mb']
            print(f"    {category}: {size_mb:.1f} MB")

    if status['missing_categories']:
        print("  Missing:")
        for category in status['missing_categories']:
            print(f"    {category}")

    # Model status
    print(f"\nMistral 7B Model:")
    if status['model_exists']:
        size_gb = status.get('model_size_gb', 0)
        print(f"  Exists: {size_gb:.1f} GB")
    else:
        print(f"  Missing")


def main():
    """Main function with menu system."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("PERSONALIZED STORYTELLING SYSTEM - DATA DOWNLOADER")
    print("=" * 70)

    downloader = ProjectGutenbergDownloader()

    while True:
        status = downloader.check_existing_data()
        display_status(status)

        print(f"\nDOWNLOAD OPTIONS")
        print("=" * 30)
        print("1. Download missing data and models")
        print("2. Force download all data")
        print("3. Force download model")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            print("\nDownloading missing data and models...")

            if status['missing_categories']:
                print(f"Missing categories: {', '.join(status['missing_categories'])}")
                categories_success = downloader.download_missing_categories(status['missing_categories'])
            else:
                print("All categories exist")
                categories_success = True

            if not status['model_exists']:
                print("Missing model, downloading...")
                model_success = downloader.download_mistral_model()
            else:
                print("Model exists")
                model_success = True

            if categories_success and model_success:
                print("\nAll missing data downloaded successfully!")
            else:
                print("\nSome downloads failed")

        elif choice == "2":
            print("\nForce downloading all categories...")

            categories = downloader.gutenberg_config.get('categories', {})
            if not categories:
                print("No categories configured")
                continue

            print(f"Will download {len(categories)} categories:")
            for name in categories.keys():
                print(f"  - {name}")

            confirm = input("\nThis will overwrite existing data. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                min_downloads = downloader.gutenberg_config.get('min_downloads', 2000)
                success_count = 0

                for category_name, config in categories.items():
                    bookshelf_id = config['bookshelf_id']
                    if downloader.download_category_books(category_name, bookshelf_id, min_downloads):
                        success_count += 1

                if success_count == len(categories):
                    print("\nAll categories downloaded successfully!")
                else:
                    print("\nSome category downloads failed")
            else:
                print("Cancelled")

        elif choice == "3":
            print("\nForce downloading Mistral 7B model...")

            model_path = os.path.join(downloader.data_paths['models'], 'mistral-7b-base')
            if os.path.exists(model_path):
                confirm = input("Model exists. Overwrite? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("Cancelled")
                    continue

                import shutil
                shutil.rmtree(model_path)
                print("  Removed existing model")

            success = downloader.download_mistral_model()
            if success:
                print("\nModel downloaded successfully!")
            else:
                print("\nModel download failed")

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()