import pandas as pd
import os
import sys
import re
import textstat
from helpers import set_cwd

# Get current working directory for path operations
cwd = set_cwd()

# Add utils to path for imports
sys.path.append(os.path.join(cwd, 'utils'))
from helpers import (
    load_config, ensure_dir_exists, clean_text, get_age_group,
    check_cache_overwrite, log_operation_status, create_progress_bar
)
from eval import load_evaluation_models, evaluate_stories_list, filter_safe_stories

# Add this line for debugging (#3):
DEBUG_SINGLE_STORY = False  # Set to True to process only 1 story for debugging


def load_gutenberg_text_files() -> dict[str, str]:
    """Load Project Gutenberg text files from data/raw directory."""
    config = load_config()
    raw_data_path = config['paths']['data_raw']

    text_data = {}
    categories = config['gutenberg']['categories']

    for category_name in categories.keys():
        log_operation_status(f"Loading {category_name}")

        file_name = f"{category_name}_stories.txt"
        file_path = os.path.join(raw_data_path, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_data[category_name] = f.read()
            print(f"  ✅ Loaded {file_name} ({len(text_data[category_name]):,} chars)")
        else:
            print(f"  ❌ Missing {file_name}")

    return text_data


def extract_books_from_gutenberg_file(text: str, category_name: str) -> list[dict[str, any]]:
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


def extract_stories_from_books(books: list[dict[str, any]]) -> list[str]:
    """Extract story segments from books for training."""
    config = load_config()
    min_length = config['data']['min_story_length']
    max_length = config['data']['max_story_length']

    stories = []

    for book in books:
        text = book['text']

        if book.get('bookshelf_id') == '634':
            individual_stories = split_short_story_collection(text)
            if len(individual_stories) > 1:
                for story in individual_stories:
                    if min_length <= len(story) <= max_length:
                        stories.append(clean_text(story))
                continue

        if len(text) <= max_length:
            if len(text) >= min_length:
                stories.append(clean_text(text))
        else:
            chapters = extract_chapters(text)
            if len(chapters) > 1:
                for chapter in chapters:
                    if min_length <= len(chapter) <= max_length:
                        stories.append(clean_text(chapter))
            else:
                chunks = extract_paragraph_chunks(text, min_length, max_length)
                stories.extend([clean_text(chunk) for chunk in chunks])

    return stories


def split_short_story_collection(text: str) -> list[str]:
    """Split short story collections into individual stories."""
    contents_match = re.search(r'^\s*(?:contents?|table\s+of\s+contents?)', text, re.IGNORECASE | re.MULTILINE)

    if contents_match:
        text = text[contents_match.end():]

    story_patterns = [
        r'\n\n([A-Z][A-Z\s]+)\n\n',
        r'\n\n(\d+\.?\s+[A-Z][^\n]+)\n\n',
        r'\n\n([IVXLC]+\.?\s+[A-Z][^\n]+)\n\n'
    ]

    stories = []
    for pattern in story_patterns:
        matches = list(re.finditer(pattern, text))
        if len(matches) > 1:
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                story = text[start:end].strip()
                if len(story) > 500:
                    stories.append(story)
            break

    return stories if len(stories) > 1 else [text]


def extract_chapters(text: str) -> list[str]:
    """Extract chapters from a book."""
    chapter_patterns = [
        r'\n\s*CHAPTER\s+[IVXLC\d]+[^\n]*\n',
        r'\n\s*Chapter\s+[IVXLC\d]+[^\n]*\n',
        r'\n\s*[IVXLC]+\.\s*[^\n]*\n',
        r'\n\s*\d+\.\s*[^\n]*\n'
    ]

    for pattern in chapter_patterns:
        chapters = re.split(pattern, text)
        if len(chapters) > 2:
            return [chapter.strip() for chapter in chapters[1:] if len(chapter.strip()) > 200]

    return [text]


def extract_paragraph_chunks(text: str, min_length: int, max_length: int) -> list[str]:
    """Extract paragraph-based chunks from text."""
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


def assign_age_groups_by_reading_level(stories: list[str], category_name: str) -> list[dict[str, any]]:
    """Assign age groups to stories based on Flesch-Kincaid reading level analysis."""
    story_data = []
    for story in stories:
        # Calculate Flesch-Kincaid grade level
        reading_level = textstat.flesch_kincaid_grade(story)

        # Determine age group based on reading level
        if reading_level <= 5.9:
            age_group = 'child'
        elif reading_level <= 12.9:
            age_group = 'kid'
        elif reading_level <= 17.9:
            age_group = 'teen'
        else:
            age_group = 'adult'

        story_data.append({
            'text': story,
            'age_group': age_group,
            'source': category_name,
            'length': len(story),
            'word_count': len(story.split()),
            'reading_level': reading_level
        })

    return story_data


def create_age_grouped_datasets(all_stories: list[dict[str, any]]) -> dict[str, pd.DataFrame]:
    """Create separate datasets for each age group."""
    age_groups = ['child', 'kid', 'teen', 'adult']
    datasets = {}

    for age_group in age_groups:
        group_stories = [story for story in all_stories if story['age_group'] == age_group]
        datasets[age_group] = pd.DataFrame(group_stories) if group_stories else pd.DataFrame()

    return datasets


def evaluate_and_filter_stories(datasets: dict[str, pd.DataFrame], config: dict[str, any]) -> dict[str, pd.DataFrame]:
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

        evaluations = evaluate_stories_list(stories, models, config, f"{age_group} stories")

        safe_stories = filter_safe_stories(stories, evaluations)

        eval_df = pd.DataFrame(evaluations)

        safe_indices = [i for i, eval_result in enumerate(evaluations) if not eval_result['is_toxic']]
        safe_df = df.iloc[safe_indices].reset_index(drop=True)
        safe_eval_df = eval_df.iloc[safe_indices].reset_index(drop=True)

        # Fix duplicate column names by removing duplicates before concatenation
        if not safe_df.empty and not safe_eval_df.empty:
            # Remove any overlapping columns from eval_df
            overlapping_cols = [col for col in safe_eval_df.columns if col in safe_df.columns]
            if overlapping_cols:
                safe_eval_df = safe_eval_df.drop(columns=overlapping_cols)

            combined_df = pd.concat([safe_df, safe_eval_df], axis=1)
        else:
            combined_df = safe_df if not safe_df.empty else pd.DataFrame()

        filtered_datasets[age_group] = combined_df

        removed_count = len(df) - len(safe_df)
        print(f"  🧹 {age_group}: Kept {len(safe_df)}/{len(df)} stories (removed {removed_count} toxic)")

    return filtered_datasets


def save_processed_datasets(datasets: dict[str, pd.DataFrame]) -> None:
    """Save processed datasets to data/processed directory."""
    config = load_config()
    processed_path = config['paths']['data_processed']
    ensure_dir_exists(processed_path)

    for age_group, df in datasets.items():
        if not df.empty:
            output_file = os.path.join(processed_path, f"{age_group}_stories.csv")
            df.to_csv(output_file, index=False)
            print(f"  ✅ Saved {age_group}: {len(df)} stories")
        else:
            print(f"  ⚠️ No stories for {age_group} group")

    if datasets:
        combined_df = pd.concat([df for df in datasets.values() if not df.empty], ignore_index=True)
        combined_file = os.path.join(processed_path, "combined_stories.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"  ✅ Saved combined: {len(combined_df)} stories")


def load_processed_datasets() -> dict[str, pd.DataFrame]:
    """Load processed datasets from data/processed directory."""
    config = load_config()
    processed_path = config['paths']['data_processed']

    datasets = {}
    age_groups = ['child', 'kid', 'teen', 'adult']

    for age_group in age_groups:
        file_path = os.path.join(processed_path, f"{age_group}_stories.csv")
        datasets[age_group] = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()

    return datasets


def validate_processed_data(datasets: dict[str, pd.DataFrame]) -> bool:
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

        # Fix duplicate column issue by removing duplicates before validation
        if df.columns.duplicated().any():
            print(f"  🔧 {age_group}: Removing duplicate columns")
            df = df.loc[:, ~df.columns.duplicated()]
            datasets[age_group] = df

        # Check if required columns exist
        required_cols = ['text', 'age_group', 'source', 'length', 'word_count']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  ❌ {age_group}: Missing columns: {missing_cols}")
            all_valid = False
            continue

        # Validate length constraints
        try:
            length_violations = df[(df['length'] < min_length) | (df['length'] > max_length)]

            if len(length_violations) > 0:
                print(f"  ❌ {age_group}: {len(length_violations)} stories violate length constraints")
                all_valid = False
                continue
        except Exception as e:
            print(f"  ❌ {age_group}: Error validating length constraints: {e}")
            all_valid = False
            continue

        if 'is_toxic' in df.columns:
            toxic_count = df['is_toxic'].sum()
            if toxic_count > 0:
                print(f"  ❌ {age_group}: Contains {toxic_count} toxic stories")
                all_valid = False
                continue

        print(f"  ✅ {age_group}: All checks passed")

    return all_valid


def get_dataset_statistics(datasets: dict[str, pd.DataFrame]) -> dict[str, any]:
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


def process_gutenberg_data() -> dict[str, pd.DataFrame]:
    """Main data processing pipeline for Project Gutenberg data."""
    config = load_config()
    processed_path = config['paths']['data_processed']

    if os.path.exists(processed_path) and not check_cache_overwrite(processed_path, "Processed datasets"):
        return load_processed_datasets()

    log_operation_status("Gutenberg data processing")

    raw_texts = load_gutenberg_text_files()
    if not raw_texts:
        print("❌ No Gutenberg data found. Please run download_data.py first.")
        return {}

    all_stories = []

    for category_name, text_content in raw_texts.items():
        log_operation_status(f"Processing {category_name}")

        books = extract_books_from_gutenberg_file(text_content, category_name)
        print(f"  📚 Extracted {len(books)} books")

        if not books:
            print(f"  ⚠️ No books found in {category_name}")
            continue

        stories = extract_stories_from_books(books)
        print(f"  📖 Extracted {len(stories)} story segments")

        if not stories:
            print(f"  ⚠️ No story segments extracted from {category_name}")
            continue

        # Apply debugging filter if enabled
        if DEBUG_SINGLE_STORY:
            stories = stories[:1]
            print(f"  🐛 DEBUG: Processing only 1 story")

        # Use Flesch-Kincaid reading level for age group assignment
        story_data = assign_age_groups_by_reading_level(stories, category_name)
        all_stories.extend(story_data)

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

    datasets = create_age_grouped_datasets(all_stories)

    print("\n📈 Pre-evaluation age group distribution:")
    total_stories = 0
    for age_group, df in datasets.items():
        count = len(df)
        total_stories += count
        if count > 0 and not df.empty:
            try:
                avg_length = df['length'].mean()
                avg_reading_level = df['reading_level'].mean() if 'reading_level' in df.columns else 0
                print(
                    f"  {age_group}: {count} stories (avg {avg_length:.0f} chars, reading level {avg_reading_level:.1f})")
            except Exception as e:
                print(f"  {age_group}: {count} stories (error calculating stats: {e})")
        else:
            print(f"  {age_group}: 0 stories")

    print(f"\n🔍 Evaluating and filtering {total_stories} stories...")

    if total_stories == 0:
        print("❌ No stories to evaluate. Check data extraction process.")
        return {}

    filtered_datasets = evaluate_and_filter_stories(datasets, config)

    print("\n📈 Post-evaluation age group distribution:")
    filtered_total = 0
    for age_group, df in filtered_datasets.items():
        count = len(df)
        filtered_total += count

        if count > 0 and not df.empty and 'length' in df.columns:
            try:
                # Robust mean calculation
                length_mean = df['length'].mean()
                avg_length = float(length_mean) if hasattr(length_mean, '__float__') else (
                    length_mean.iloc[0] if hasattr(length_mean, 'iloc') and len(length_mean) > 0 else 0
                )

                avg_grammar = 0
                avg_coherence = 0

                if 'grammar_score' in df.columns:
                    grammar_mean = df['grammar_score'].mean()
                    avg_grammar = float(grammar_mean) if hasattr(grammar_mean, '__float__') else (
                        grammar_mean.iloc[0] if hasattr(grammar_mean, 'iloc') and len(grammar_mean) > 0 else 0
                    )

                if 'coherence_score' in df.columns:
                    coherence_mean = df['coherence_score'].mean()
                    avg_coherence = float(coherence_mean) if hasattr(coherence_mean, '__float__') else (
                        coherence_mean.iloc[0] if hasattr(coherence_mean, 'iloc') and len(coherence_mean) > 0 else 0
                    )

                print(
                    f"  {age_group}: {count} stories (avg {avg_length:.0f} chars, grammar {avg_grammar:.1f}, coherence {avg_coherence:.1f})")
            except Exception as e:
                print(f"  {age_group}: {count} stories (ERROR calculating stats: {e})")
        else:
            print(f"  {age_group}: 0 stories")

    print(f"\n🎯 Filtered stories: {filtered_total} (removed {total_stories - filtered_total})")

    if filtered_total > 0:
        # Remove any empty DataFrames before concatenation
        non_empty_dfs = [df for df in filtered_datasets.values() if not df.empty]
        if non_empty_dfs:
            combined_df = pd.concat(non_empty_dfs, ignore_index=True)
            source_counts = combined_df['source'].value_counts()

            print(f"\n📚 Source distribution:")
            for source, count in source_counts.items():
                percentage = (count / filtered_total) * 100
                print(f"  {source}: {count} stories ({percentage:.1f}%)")

    save_processed_datasets(filtered_datasets)

    log_operation_status("Gutenberg data processing", "completed")
    return filtered_datasets


def main():
    """Main function for data loading and processing."""
    log_operation_status("Project Gutenberg data loading and processing")

    datasets = process_gutenberg_data()

    if datasets:
        is_valid = validate_processed_data(datasets)

        if not is_valid:
            print("\n⚠️ Data validation failed. Please check the issues above.")

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