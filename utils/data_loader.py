import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import re
from utils.helpers import (
    load_config, ensure_dir_exists, clean_text, check_cache_overwrite, log_operation_status
)
from utils.eval import load_evaluation_models, evaluate_stories_list, filter_safe_stories


def load_gutenberg_text_files() -> Dict[str, str]:
    """Load Project Gutenberg text files from data/raw directory."""
    config = load_config()
    raw_data_path = Path(config['paths']['data_raw'])

    text_data = {}
    categories = config['gutenberg']['categories']

    for category_name in categories.keys():
        log_operation_status(f"Loading {category_name}")

        file_name = f"{category_name}_stories.txt"
        file_path = raw_data_path / file_name

        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_data[category_name] = f.read()
            print(f"  ✅ Loaded {file_name} ({len(text_data[category_name]):,} chars)")
        else:
            print(f"  ❌ Missing {file_name}")

    return text_data


def extract_books_from_gutenberg_file(text: str, category_name: str) -> List[Dict[str, Any]]:
    """Extract individual books from Project Gutenberg combined file."""
    books = []
    book_sections = text.split("=" * 80)

    for section in book_sections:
        section = section.strip()
        if len(section) < 500:
            continue

        lines = section.split('\n')
        title = "Unknown"
        author = "Unknown"
        gutenberg_id = "Unknown"
        bookshelf_id = "Unknown"
        text_start_idx = 0

        # Extract metadata from first 15 lines
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if line.startswith("TITLE:"):
                title = line[6:].strip()
            elif line.startswith("AUTHOR:"):
                author = line[7:].strip()
            elif line.startswith("PROJECT_GUTENBERG_ID:"):
                gutenberg_id = line[22:].strip()
            elif line.startswith("BOOKSHELF_ID:"):
                bookshelf_id = line[13:].strip()
            elif not line or i > 12:
                text_start_idx = i
                break

        book_text = '\n'.join(lines[text_start_idx:]).strip()

        if len(book_text.split()) >= 1000:
            books.append({
                'title': title,
                'author': author,
                'gutenberg_id': gutenberg_id,
                'text': book_text,
                'category': category_name,
                'word_count': len(book_text.split()),
                'char_count': len(book_text),
                'bookshelf_id': bookshelf_id
            })

    return books


def extract_stories_from_books(books: List[Dict[str, Any]]) -> List[str]:
    """Extract story segments from books for training."""
    config = load_config()
    min_length = config['data']['min_story_length']
    max_length = config['data']['max_story_length']

    stories = []

    for book in books:
        text = book['text']

        # Handle short story collections (bookshelf 634)
        if book.get('bookshelf_id') == '634':
            individual_stories = split_short_story_collection(text)
            if len(individual_stories) > 1:
                for story in individual_stories:
                    if min_length <= len(story) <= max_length:
                        stories.append(clean_text(story))
                continue

        # Handle regular books
        if len(text) <= max_length:
            if len(text) >= min_length:
                stories.append(clean_text(text))
        else:
            # Split large books into chapters or chunks
            chapters = extract_chapters(text)
            if len(chapters) > 1:
                for chapter in chapters:
                    if min_length <= len(chapter) <= max_length:
                        stories.append(clean_text(chapter))
            else:
                chunks = extract_paragraph_chunks(text, min_length, max_length)
                stories.extend([clean_text(chunk) for chunk in chunks])

    return stories


def split_short_story_collection(text: str) -> List[str]:
    """Split short story collections into individual stories."""
    # Find table of contents
    contents_match = re.search(r'^\s*(?:contents?|table\s+of\s+contents?)\s*$', text, re.IGNORECASE | re.MULTILINE)
    if not contents_match:
        return [text]

    # Extract story titles from contents
    contents_start = contents_match.end()
    contents_end = contents_start + min(2000, len(text) - contents_start)
    contents_section = text[contents_start:contents_end]

    # Find story titles (lines that look like titles)
    story_titles = []
    for line in contents_section.split('\n'):
        line = line.strip()
        if (len(line) > 3 and len(line) < 100 and
                not any(skip in line.lower() for skip in ['page', 'chapter', 'part', 'contents']) and
                (line.isupper() or re.match(r'^[IVX]+\.\s+', line) or re.match(r'^\d+\.\s+', line))):
            # Clean title
            title = re.sub(r'^\s*(?:[IVX]+\.?|[0-9]+\.?)\s*', '', line)
            title = re.sub(r'\s*\.+\s*\d+\s*$', '', title)
            if len(title) > 3:
                story_titles.append(title.strip())

    if len(story_titles) < 2:
        return [text]

    # Split text by story titles
    stories = []
    text_lines = text.split('\n')

    title_positions = []
    for title in story_titles[:10]:  # Limit to first 10 stories
        for i, line in enumerate(text_lines):
            if title.lower() in line.lower() and abs(len(line.strip()) - len(title)) < 20:
                title_positions.append(i)
                break

    title_positions.sort()

    for i, start_pos in enumerate(title_positions):
        end_pos = title_positions[i + 1] if i + 1 < len(title_positions) else len(text_lines)
        story_lines = text_lines[start_pos + 1:end_pos]  # Skip title line

        # Remove empty lines and separators
        while story_lines and (not story_lines[0].strip() or re.match(r'^[-=_*]{3,}$', story_lines[0].strip())):
            story_lines.pop(0)

        story_text = '\n'.join(story_lines).strip()
        if story_text and len(story_text.split()) >= 100:
            stories.append(story_text)

    return stories if stories else [text]


def extract_chapters(text: str) -> List[str]:
    """Extract chapters from book text."""
    chapter_patterns = [
        r'\n\s*CHAPTER\s+[IVXLCDM\d]+[^\n]*\n',
        r'\n\s*Chapter\s+\d+[^\n]*\n',
        r'\n\s*\d+\.\s*[A-Z][^\n]*\n'
    ]

    for pattern in chapter_patterns:
        splits = re.split(pattern, text, flags=re.IGNORECASE)
        if len(splits) > 2:
            chapter_markers = re.findall(pattern, text, flags=re.IGNORECASE)
            chapters = []

            for i, split in enumerate(splits[1:]):
                if i < len(chapter_markers):
                    chapter_text = chapter_markers[i] + split
                else:
                    chapter_text = split

                chapter_text = chapter_text.strip()
                if len(chapter_text) > 500:
                    chapters.append(chapter_text)

            if chapters:
                return chapters

    return [text]


def extract_paragraph_chunks(text: str, min_length: int, max_length: int) -> List[str]:
    """Extract chunks based on paragraph boundaries."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

        if len(potential_chunk) <= max_length:
            current_chunk = potential_chunk
        else:
            if len(current_chunk) >= min_length:
                chunks.append(current_chunk)
            current_chunk = paragraph

    if len(current_chunk) >= min_length:
        chunks.append(current_chunk)

    return chunks


def assign_age_groups(stories: List[str], category_name: str) -> List[Dict[str, Any]]:
    """Assign age groups to stories based on Gutenberg category."""
    config = load_config()
    gutenberg_config = config['gutenberg']['categories']
    target_age_groups = gutenberg_config.get(category_name, {}).get('target_age_groups', ['adult'])

    story_data = []
    for story in stories:
        age_group = np.random.choice(target_age_groups)
        story_data.append({
            'text': story,
            'age_group': age_group,
            'source': category_name,
            'length': len(story),
            'word_count': len(story.split())
        })

    return story_data


def create_age_grouped_datasets(all_stories: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Create separate datasets for each age group."""
    age_groups = ['child', 'kid', 'teen', 'adult']
    datasets = {}

    for age_group in age_groups:
        group_stories = [story for story in all_stories if story['age_group'] == age_group]
        datasets[age_group] = pd.DataFrame(group_stories) if group_stories else pd.DataFrame()

    return datasets


def evaluate_and_filter_stories(datasets: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Evaluate stories and filter out inappropriate content."""
    log_operation_status("Loading evaluation models")
    models = load_evaluation_models()
    print("  ✅ Evaluation models loaded")

    filtered_datasets = {}

    for age_group, df in datasets.items():
        if df.empty:
            filtered_datasets[age_group] = df
            continue

        log_operation_status(f"Evaluating {age_group} stories")
        stories = df['text'].tolist()

        # Evaluate stories
        evaluations = evaluate_stories_list(stories, models, config, f"{age_group} stories")

        # Filter safe stories
        safe_stories = filter_safe_stories(stories, evaluations)

        # Create evaluation dataframe
        eval_df = pd.DataFrame(evaluations)

        # Filter original dataframe to safe stories only
        safe_indices = [i for i, eval_result in enumerate(evaluations) if not eval_result['is_toxic']]
        safe_df = df.iloc[safe_indices].reset_index(drop=True)
        safe_eval_df = eval_df.iloc[safe_indices].reset_index(drop=True)

        # Combine data with evaluations
        combined_df = pd.concat([safe_df, safe_eval_df], axis=1)
        filtered_datasets[age_group] = combined_df

        removed_count = len(df) - len(safe_df)
        print(f"  🧹 {age_group}: Kept {len(safe_df)}/{len(df)} stories (removed {removed_count} toxic)")

    return filtered_datasets


def save_processed_datasets(datasets: Dict[str, pd.DataFrame]) -> None:
    """Save processed datasets to data/processed directory."""
    config = load_config()
    processed_path = Path(config['paths']['data_processed'])
    ensure_dir_exists(processed_path)

    for age_group, df in datasets.items():
        if not df.empty:
            output_file = processed_path / f"{age_group}_stories.csv"
            df.to_csv(output_file, index=False)
            print(f"  ✅ Saved {age_group}: {len(df)} stories")
        else:
            print(f"  ⚠️ No stories for {age_group} group")

    # Save combined dataset
    if datasets:
        combined_df = pd.concat([df for df in datasets.values() if not df.empty], ignore_index=True)
        combined_file = processed_path / "combined_stories.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"  ✅ Saved combined: {len(combined_df)} stories")


def load_processed_datasets() -> Dict[str, pd.DataFrame]:
    """Load processed datasets from data/processed directory."""
    config = load_config()
    processed_path = Path(config['paths']['data_processed'])

    datasets = {}
    age_groups = ['child', 'kid', 'teen', 'adult']

    for age_group in age_groups:
        file_path = processed_path / f"{age_group}_stories.csv"
        datasets[age_group] = pd.read_csv(file_path) if file_path.exists() else pd.DataFrame()

    return datasets


def process_gutenberg_data() -> Dict[str, pd.DataFrame]:
    """Main data processing pipeline for Project Gutenberg data."""
    config = load_config()
    processed_path = Path(config['paths']['data_processed'])

    # Check if processed data exists
    if processed_path.exists() and not check_cache_overwrite(str(processed_path), "Processed datasets"):
        return load_processed_datasets()

    log_operation_status("Gutenberg data processing")

    # Load raw text files
    raw_texts = load_gutenberg_text_files()
    if not raw_texts:
        print("❌ No Gutenberg data found. Please run download_data.py first.")
        return {}

    # Process each category
    all_stories = []

    for category_name, text_content in raw_texts.items():
        log_operation_status(f"Processing {category_name}")

        # Extract books from combined file
        books = extract_books_from_gutenberg_file(text_content, category_name)
        print(f"  📚 Extracted {len(books)} books")

        if not books:
            print(f"  ⚠️ No books found in {category_name}")
            continue

        # Extract stories from books
        stories = extract_stories_from_books(books)
        print(f"  📖 Extracted {len(stories)} story segments")

        if not stories:
            print(f"  ⚠️ No story segments extracted from {category_name}")
            continue

        # Assign age groups
        story_data = assign_age_groups(stories, category_name)
        all_stories.extend(story_data)

        # Show category statistics
        category_word_count = sum(len(story.split()) for story in stories)
        avg_story_length = category_word_count / len(stories) if stories else 0

        print(f"  📊 {category_name} stats:")
        print(f"    Total stories: {len(stories)}")
        print(f"    Total words: {category_word_count:,}")
        print(f"    Avg story length: {avg_story_length:.0f} words")

    print(f"\n📊 Total stories extracted: {len(all_stories)}")

    if not all_stories:
        print("❌ No stories were extracted from any category")
        return {}

    # Create age-grouped datasets
    datasets = create_age_grouped_datasets(all_stories)

    # Print pre-evaluation statistics
    print("\n📈 Pre-evaluation age group distribution:")
    total_stories = 0
    for age_group, df in datasets.items():
        count = len(df)
        total_stories += count
        if count > 0:
            avg_length = df['length'].mean()
            print(f"  {age_group}: {count} stories (avg {avg_length:.0f} chars)")
        else:
            print(f"  {age_group}: 0 stories")

    print(f"\n🔍 Evaluating and filtering {total_stories} stories...")

    # Evaluate and filter stories for safety
    filtered_datasets = evaluate_and_filter_stories(datasets, config)

    # Print post-evaluation statistics
    print("\n📈 Post-evaluation age group distribution:")
    filtered_total = 0
    for age_group, df in filtered_datasets.items():
        count = len(df)
        filtered_total += count
        if count > 0:
            avg_length = df['length'].mean()
            avg_grammar = df['grammar_score'].mean() if 'grammar_score' in df.columns else 0
            avg_coherence = df['coherence_score'].mean() if 'coherence_score' in df.columns else 0
            print(
                f"  {age_group}: {count} stories (avg {avg_length:.0f} chars, grammar {avg_grammar:.1f}, coherence {avg_coherence:.1f})")
        else:
            print(f"  {age_group}: 0 stories")

    print(f"\n🎯 Filtered stories: {filtered_total} (removed {total_stories - filtered_total})")

    # Show source distribution
    if filtered_total > 0:
        combined_df = pd.concat([df for df in filtered_datasets.values() if not df.empty], ignore_index=True)
        source_counts = combined_df['source'].value_counts()

        print(f"\n📚 Source distribution:")
        for source, count in source_counts.items():
            percentage = (count / filtered_total) * 100
            print(f"  {source}: {count} stories ({percentage:.1f}%)")

    # Save processed datasets
    save_processed_datasets(filtered_datasets)

    log_operation_status("Gutenberg data processing", "completed")
    return filtered_datasets


def validate_processed_data(datasets: Dict[str, pd.DataFrame]) -> bool:
    """Validate that processed data meets requirements."""
    config = load_config()
    min_length = config['data']['min_story_length']
    max_length = config['data']['max_story_length']

    print(f"\n🔍 Data Validation Results:")
    all_valid = True

    for age_group, df in datasets.items():
        if df.empty:
            print(f"  ❌ {age_group}: No data")
            all_valid = False
            continue

        # Check length constraints
        length_violations = df[(df['length'] < min_length) | (df['length'] > max_length)]

        if len(length_violations) > 0:
            print(f"  ❌ {age_group}: {len(length_violations)} stories violate length constraints")
            all_valid = False
            continue

        # Check required columns
        required_cols = ['text', 'age_group', 'source', 'length', 'word_count']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  ❌ {age_group}: Missing columns: {missing_cols}")
            all_valid = False
            continue

        # Check evaluation columns if present
        if 'is_toxic' in df.columns:
            toxic_count = df['is_toxic'].sum()
            if toxic_count > 0:
                print(f"  ❌ {age_group}: Contains {toxic_count} toxic stories")
                all_valid = False
                continue

        print(f"  ✅ {age_group}: All checks passed")

    return all_valid


def get_dataset_statistics(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Generate statistics about processed datasets."""
    stats = {}

    for age_group, df in datasets.items():
        if not df.empty:
            group_stats = {
                'count': len(df),
                'avg_length': df['length'].mean(),
                'avg_word_count': df['word_count'].mean(),
                'sources': df['source'].value_counts().to_dict(),
                'total_words': df['word_count'].sum(),
                'length_range': {
                    'min': df['length'].min(),
                    'max': df['length'].max(),
                    'median': df['length'].median()
                }
            }

            # Add evaluation statistics if available
            if 'grammar_score' in df.columns:
                group_stats['evaluation'] = {
                    'avg_grammar': df['grammar_score'].mean(),
                    'avg_coherence': df['coherence_score'].mean(),
                    'avg_flesch_kincaid': df['flesch_kincaid_score'].mean(),
                    'avg_perplexity': df['perplexity'].mean(),
                    'toxic_count': df['is_toxic'].sum() if 'is_toxic' in df.columns else 0
                }

            stats[age_group] = group_stats
        else:
            stats[age_group] = {
                'count': 0, 'avg_length': 0, 'avg_word_count': 0,
                'sources': {}, 'total_words': 0,
                'length_range': {'min': 0, 'max': 0, 'median': 0}
            }

    return stats


def main():
    """Main function for data loading and processing."""
    log_operation_status("Project Gutenberg data loading and processing")

    # Process data
    datasets = process_gutenberg_data()

    if datasets:
        # Validate processed data
        is_valid = validate_processed_data(datasets)

        if not is_valid:
            print("\n⚠️ Data validation failed. Please check the issues above.")

        # Generate and display statistics
        stats = get_dataset_statistics(datasets)

        print("\n📊 DATASET STATISTICS")
        print("=" * 50)

        total_stories = 0
        total_words = 0

        for age_group, group_stats in stats.items():
            count = group_stats['count']
            total_stories += count
            total_words += group_stats['total_words']

            if count > 0:
                print(f"\n{age_group.upper()}:")
                print(f"  Stories: {count:,}")
                print(f"  Avg Length: {group_stats['avg_length']:.0f} chars")
                print(f"  Avg Words: {group_stats['avg_word_count']:.0f}")
                print(f"  Total Words: {group_stats['total_words']:,}")
                print(
                    f"  Length Range: {group_stats['length_range']['min']}-{group_stats['length_range']['max']} chars")

                # Show evaluation stats if available
                if 'evaluation' in group_stats:
                    eval_stats = group_stats['evaluation']
                    print(f"  Quality Scores:")
                    print(f"    Grammar: {eval_stats['avg_grammar']:.1f}/100")
                    print(f"    Coherence: {eval_stats['avg_coherence']:.1f}/100")
                    print(f"    Reading Level: {eval_stats['avg_flesch_kincaid']:.1f}")
                    print(f"    Perplexity: {eval_stats['avg_perplexity']:.1f}")

                if group_stats['sources']:
                    top_sources = list(group_stats['sources'].items())[:3]
                    sources_str = ", ".join([f"{k}({v})" for k, v in top_sources])
                    print(f"  Top Sources: {sources_str}")

        print(f"\n🎯 TOTALS:")
        print(f"  Stories: {total_stories:,}")
        print(f"  Words: {total_words:,}")
        print(f"  Avg Words/Story: {total_words / total_stories:.0f}" if total_stories > 0 else "  Avg Words/Story: 0")

        if total_stories > 0:
            print("✅ Data processing completed successfully!")
            print("📁 Ready for tokenization and training")
        else:
            print("❌ No stories were processed. Check source data.")
    else:
        print("❌ Data processing failed.")

    log_operation_status("Project Gutenberg data loading and processing", "completed")


if __name__ == "__main__":
    main()