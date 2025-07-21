#!/usr/bin/env python3
"""
Data download script for the Personalized Storytelling System.
Downloads required texts from Project Gutenberg and the Mixtral model for training.
"""

import os
import sys
import logging
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import random

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utils.helpers import ensure_dir_exists, load_config

logger = logging.getLogger(__name__)


class ProjectGutenbergDownloader:
    """Downloads texts from Project Gutenberg based on bookshelf categories."""

    def __init__(self, config_path: str = "configs/model_config.yaml"):
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
            file_path = Path(self.data_paths['data_raw']) / file_name
            exists = file_path.exists() and file_path.stat().st_size > 1000  # At least 1KB

            status['categories'][category_name] = {
                'exists': exists,
                'path': file_path,
                'size_mb': file_path.stat().st_size / (1024 * 1024) if exists else 0
            }

            if exists:
                status['existing_categories'].append(category_name)
            else:
                status['missing_categories'].append(category_name)

        # Check Mixtral model
        model_path = Path(self.data_paths['models']) / 'mixtral-8x7b-base'
        if model_path.exists() and list(model_path.glob('*.json')):  # Check for config files
            status['model_exists'] = True
            model_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            status['model_size_gb'] = model_size / (1024 ** 3)

        return status

    def get_bookshelf_books(self, bookshelf_id: int, min_downloads: int = 1000) -> List[Dict[str, Any]]:
        """Get list of books from a Project Gutenberg bookshelf with minimum download threshold."""
        print(f"📚 Fetching books from bookshelf {bookshelf_id}")

        books = []
        page = 1

        while True:
            bookshelf_url = f"{self.bookshelf_url}/{bookshelf_id}?start_index={(page - 1) * 25}"

            time.sleep(self.request_delay)
            response = self.session.get(bookshelf_url)
            response.raise_for_status()

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
                if not book_url.startswith('/ebooks/'):
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
        print(f"  Found {len(books)} qualifying books")
        return books

    def download_book_text(self, book_id: str, output_path: Path) -> bool:
        """Download plain text version of a book."""
        text_urls = [
            f"{self.base_url}/files/{book_id}/{book_id}-0.txt",
            f"{self.base_url}/files/{book_id}/{book_id}.txt",
            f"{self.base_url}/ebooks/{book_id}.txt.utf-8"
        ]

        for url in text_urls:
            time.sleep(self.request_delay)
            response = self.session.get(url, stream=True)

            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                if 'text' not in content_type and 'plain' not in content_type:
                    continue

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                if self._validate_text_file(output_path):
                    return True
                else:
                    output_path.unlink()
            elif response.status_code == 404:
                continue

        return False

    def check_copyright_status(self, text: str) -> Dict[str, Any]:
        """Check copyright status of Project Gutenberg text."""
        header_end_patterns = [r'\*{3,}', r'_{3,}', r'={3,}']

        header = text
        for pattern in header_end_patterns:
            match = re.search(pattern, text)
            if match:
                header = text[:match.start()]
                break

        if len(header) == len(text):
            header = text[:3000]

        header_lower = header.lower()

        copyright_info = {
            'is_public_domain': True,
            'copyright_notice': None,
            'restrictions': [],
            'status': 'public_domain',
            'header_length': len(header)
        }

        restrictive_patterns = [
            r'copyright.*(?:reserved|holder|owner)',
            r'all rights reserved',
            r'reproduction.*prohibited',
            r'may not be.*reproduced',
            r'permission.*required',
            r'unauthorized.*distribution.*prohibited',
            r'commercial.*use.*prohibited',
            r'not.*public domain',
            r'copyrighted.*work',
            r'rights.*reserved',
            r'©.*\d{4}',
            r'copyright \d{4}',
            r'protected.*copyright',
            r'exclusive.*rights',
            r'trademark.*registered',
            r'proprietary.*rights',
            r'restricted.*use'
        ]

        public_domain_patterns = [
            r'public domain',
            r'not copyrighted',
            r'copyright.*expired',
            r'free.*distribution',
            r'no copyright',
            r'may.*freely.*distributed',
            r'unrestricted.*use',
            r'project gutenberg.*public domain',
            r'gutenberg.*ebook.*public domain',
            r'freely.*available',
            r'copyright.*waived',
            r'dedicated.*public domain'
        ]

        found_restrictions = []
        for pattern in restrictive_patterns:
            matches = re.findall(pattern, header_lower, re.IGNORECASE)
            if matches:
                found_restrictions.extend(matches)

        public_domain_confirmations = []
        for pattern in public_domain_patterns:
            matches = re.findall(pattern, header_lower, re.IGNORECASE)
            if matches:
                public_domain_confirmations.extend(matches)

        if public_domain_confirmations:
            copyright_info['status'] = 'public_domain'
            copyright_info['is_public_domain'] = True
        elif found_restrictions:
            copyright_info['is_public_domain'] = False
            copyright_info['restrictions'] = found_restrictions
            copyright_info['status'] = 'copyrighted'

            copyright_lines = []
            for line in header.split('\n'):
                line_clean = line.strip()
                if any(term in line.lower() for term in ['copyright', '©', 'rights reserved']):
                    copyright_lines.append(line_clean)

            if copyright_lines:
                copyright_info['copyright_notice'] = '. '.join(copyright_lines[:3])

        if 'project gutenberg' in header_lower:
            if any(term in header_lower for term in ['tm license', 'license agreement', 'terms of use']):
                if not any(term in header_lower for term in ['may not', 'prohibited', 'unauthorized']):
                    copyright_info['status'] = 'gutenberg_license'
                    copyright_info['is_public_domain'] = True

        return copyright_info

    def _validate_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate downloaded file for text content and copyright status."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        validation_result = {
            'is_valid_text': False,
            'is_public_domain': True,
            'copyright_info': None,
            'reason': None
        }

        if "project gutenberg" not in content.lower():
            validation_result['reason'] = 'Not a Project Gutenberg file'
            return validation_result

        sample = content[:1000]
        printable_ratio = sum(1 for c in sample if c.isprintable() or c.isspace()) / len(sample)
        if printable_ratio <= 0.9:
            validation_result['reason'] = 'File appears to be binary or corrupted'
            return validation_result

        copyright_info = self.check_copyright_status(content)
        validation_result['copyright_info'] = copyright_info
        validation_result['is_public_domain'] = copyright_info['is_public_domain']

        if not copyright_info['is_public_domain']:
            validation_result['reason'] = f"Restrictive copyright: {copyright_info['status']}"
            if copyright_info['copyright_notice']:
                validation_result['reason'] += f" - {copyright_info['copyright_notice'][:100]}"
            return validation_result

        validation_result['is_valid_text'] = True
        validation_result['reason'] = f"Valid public domain text (header: {copyright_info['header_length']} chars)"
        return validation_result

    def clean_gutenberg_text(self, text: str) -> str:
        """Clean Project Gutenberg text by removing headers/footers and metadata."""
        lines = text.split('\n')

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
            elif line_lower.startswith("chapter") or "table of contents" in line_lower:
                start_idx = i
                break

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

        cleaned_lines = []
        for line in content_lines:
            line = line.strip()
            if re.match(r'^\d+$', line) or re.match(r'^page \d+', line.lower()):
                continue
            cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)

        return cleaned_text.strip()

    def download_category_books(self, category_name: str, bookshelf_id: int,
                                min_downloads: int = 1000) -> bool:
        """Download books from a specific category with deduplication."""
        print(f"🔄 Downloading {category_name}")

        books = self.get_bookshelf_books(bookshelf_id, min_downloads)

        if not books:
            print(f"❌ No books found for {category_name}")
            return False

        category_dir = Path(self.data_paths['data_raw']) / category_name.lower().replace(' ', '_')
        ensure_dir_exists(category_dir)

        seen_books = set()
        successful_downloads = 0
        all_texts = []

        with tqdm(total=len(books), desc=f"Processing {category_name}") as pbar:
            for book in books:
                book_id = book['id']
                title = book['title']
                author = book['author']

                normalized_title = re.sub(r'[^\w\s]', '', title.lower()).strip()
                normalized_author = re.sub(r'[^\w\s]', '', author.lower()).strip()
                book_signature = f"{normalized_title}_{normalized_author}"

                if book_signature in seen_books:
                    pbar.update(1)
                    continue

                seen_books.add(book_signature)

                safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
                safe_title = re.sub(r'\s+', '_', safe_title)

                temp_file = category_dir / f"{book_id}_{safe_title}.txt"

                if self.download_book_text(book_id, temp_file):
                    validation_result = self._validate_text_file(temp_file)

                    if not validation_result['is_valid_text']:
                        temp_file.unlink()
                        pbar.update(1)
                        continue

                    with open(temp_file, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_text = f.read()

                    cleaned_text = self.clean_gutenberg_text(raw_text)

                    if len(cleaned_text.split()) >= 1000:
                        copyright_info = validation_result['copyright_info']
                        all_texts.append({
                            'title': title,
                            'author': author,
                            'text': cleaned_text,
                            'id': book_id,
                            'downloads': book['downloads'],
                            'word_count': len(cleaned_text.split()),
                            'copyright_status': copyright_info['status'],
                            'is_public_domain': copyright_info['is_public_domain'],
                            'bookshelf_id': bookshelf_id
                        })
                        successful_downloads += 1

                    temp_file.unlink()

                pbar.update(1)
                pbar.set_postfix(downloaded=successful_downloads)

        if successful_downloads == 0:
            print(f"❌ No successful downloads for {category_name}")
            return False

        all_texts.sort(key=lambda x: x['downloads'], reverse=True)

        combined_file = Path(self.data_paths['data_raw']) / f"{category_name.lower().replace(' ', '_')}_stories.txt"

        total_words = 0
        copyright_stats = {}

        with open(combined_file, 'w', encoding='utf-8') as f:
            for i, book_data in enumerate(all_texts):
                if i > 0:
                    f.write("\n\n" + "=" * 80 + "\n\n")

                f.write(f"TITLE: {book_data['title']}\n")
                f.write(f"AUTHOR: {book_data['author']}\n")
                f.write(f"PROJECT_GUTENBERG_ID: {book_data['id']}\n")
                f.write(f"DOWNLOADS: {book_data['downloads']}\n")
                f.write(f"WORD_COUNT: {book_data['word_count']}\n")
                f.write(f"COPYRIGHT_STATUS: {book_data['copyright_status']}\n")
                f.write(f"PUBLIC_DOMAIN: {book_data['is_public_domain']}\n")
                f.write(f"BOOKSHELF_ID: {book_data['bookshelf_id']}\n")
                f.write("\n" + book_data['text'])
                total_words += book_data['word_count']

                status = book_data['copyright_status']
                copyright_stats[status] = copyright_stats.get(status, 0) + 1

        print(f"✅ {category_name}: {successful_downloads} books, {total_words:,} words")
        return True

    def download_mixtral_model(self) -> bool:
        """Download the Mixtral model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from dotenv import load_dotenv

        load_dotenv()

        model_name = self.config['model']['base_model']
        model_path = os.path.join(self.data_paths['models'], 'mixtral-8x7b-base')

        print(f"🔄 Downloading Mixtral model: {model_name}")

        ensure_dir_exists(model_path)

        hf_token = os.getenv('HF_TOKEN')

        print("  📥 Downloading tokenizer...")
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
        print("  ✅ Tokenizer saved")

        print("  📥 Downloading model (this may take 20-30 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=None,
            trust_remote_code=True,
            token=hf_token,
            low_cpu_mem_usage=True
        )

        model.save_pretrained(model_path)
        print("  ✅ Model saved")

        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)

        size_gb = total_size / (1024 ** 3)
        print(f"✅ Model download complete: {size_gb:.2f} GB")

        return True

    def download_missing_categories(self, missing_categories: List[str]) -> bool:
        """Download only missing categories."""
        if not missing_categories:
            print("✅ All categories already exist")
            return True

        categories = self.gutenberg_config.get('categories', {})
        min_downloads = self.gutenberg_config.get('min_downloads', 1000)

        success_count = 0
        for category_name in missing_categories:
            if category_name in categories:
                config = categories[category_name]
                bookshelf_id = config['bookshelf_id']

                if self.download_category_books(category_name, bookshelf_id, min_downloads):
                    success_count += 1

        return success_count == len(missing_categories)

    def download_all_categories(self) -> bool:
        """Download all configured categories (overwriting existing)."""
        categories = self.gutenberg_config.get('categories', {})
        min_downloads = self.gutenberg_config.get('min_downloads', 1000)

        success_count = 0
        total_count = len(categories)

        for category_name, config in categories.items():
            bookshelf_id = config['bookshelf_id']

            if self.download_category_books(category_name, bookshelf_id, min_downloads):
                success_count += 1

        return success_count == total_count


def display_status(status: Dict[str, Any]):
    """Display current download status."""
    print("\n📊 CURRENT STATUS")
    print("=" * 50)

    # Categories status
    print("📚 Project Gutenberg Categories:")
    if status['existing_categories']:
        print("  ✅ Existing:")
        for category in status['existing_categories']:
            size_mb = status['categories'][category]['size_mb']
            print(f"    {category}: {size_mb:.1f} MB")

    if status['missing_categories']:
        print("  ❌ Missing:")
        for category in status['missing_categories']:
            print(f"    {category}")

    # Model status
    print(f"\n🤖 Mixtral Model:")
    if status['model_exists']:
        size_gb = status.get('model_size_gb', 0)
        print(f"  ✅ Exists: {size_gb:.1f} GB")
    else:
        print(f"  ❌ Missing")


def main():
    """Main function with menu system."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("🎭 PERSONALIZED STORYTELLING SYSTEM - DATA DOWNLOADER")
    print("=" * 70)

    downloader = ProjectGutenbergDownloader()

    while True:
        # Check current status
        status = downloader.check_existing_data()
        display_status(status)

        print(f"\n📋 DOWNLOAD OPTIONS")
        print("=" * 30)
        print("1. Download missing data and models")
        print("2. Force download all data")
        print("3. Force download model")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            print("\n🔄 Downloading missing data and models...")

            # Download missing categories
            if status['missing_categories']:
                print(f"📚 Missing categories: {', '.join(status['missing_categories'])}")
                categories_success = downloader.download_missing_categories(status['missing_categories'])
            else:
                print("✅ All categories exist")
                categories_success = True

            # Download model if missing
            if not status['model_exists']:
                print("🤖 Missing model, downloading...")
                model_success = downloader.download_mixtral_model()
            else:
                print("✅ Model exists")
                model_success = True

            if categories_success and model_success:
                print("\n✅ All missing data downloaded successfully!")
            else:
                print("\n❌ Some downloads failed")

        elif choice == "2":
            print("\n🔄 Force downloading all categories...")

            categories = downloader.gutenberg_config.get('categories', {})
            if not categories:
                print("❌ No categories configured")
                continue

            print(f"📚 Will download {len(categories)} categories:")
            for name in categories.keys():
                print(f"  - {name}")

            confirm = input("\nThis will overwrite existing data. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                success = downloader.download_all_categories()
                if success:
                    print("\n✅ All categories downloaded successfully!")
                else:
                    print("\n❌ Some category downloads failed")
            else:
                print("Cancelled")

        elif choice == "3":
            print("\n🔄 Force downloading Mixtral model...")

            model_path = Path(downloader.data_paths['models']) / 'mixtral-8x7b-base'
            if model_path.exists():
                confirm = input("Model exists. Overwrite? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("Cancelled")
                    continue

                # Remove existing model
                import shutil
                shutil.rmtree(model_path)
                print("  🗑️ Removed existing model")

            success = downloader.download_mixtral_model()
            if success:
                print("\n✅ Model downloaded successfully!")
            else:
                print("\n❌ Model download failed")

        elif choice == "4":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()