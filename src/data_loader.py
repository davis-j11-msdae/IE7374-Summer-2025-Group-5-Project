import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import re
from utils.helpers import (
    load_config, ensure_dir_exists, clean_text, get_age_group,
    check_cache_overwrite, log_operation_status, create_progress_bar
)


def load_raw_text_files() -> Dict[str, str]:
    """Load raw text files from data/raw directory."""
    config = load_config()
    raw_data_path = Path(config['paths']['data_raw'])

    text_data = {}
    datasets = config['datasets']

    for dataset_name, dataset_info in datasets.items():
        log_operation_status(f"Loading {dataset_name}")

        for file_name in dataset_info['files']:
            file_path = raw_data_path / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_data[dataset_name] = f.read()
                print(f"  ✅ Loaded {file_name} ({len(text_data[dataset_name]):,} chars)")
            else:
                print(f"  ❌ Missing {file_name}")

    return text_data


def extract_stories_from_text(text: str, story_type: str) -> List[str]:
    """Extract individual stories from raw text."""
    stories = []

    if story_type == "children_stories":
        # Split on common fairy tale patterns
        patterns = [
            r'\n\s*(?:THE|The)\s+[A-Z][^.\n]+\n',
            r'\n\s*\d+\.\s*[A-Z][^.\n]+\n',
            r'\n\s*(?:Once upon a time|Long ago|In a|There was|There once)\b'
        ]

        for pattern in patterns:
            splits = re.split(pattern, text)
            if len(splits) > 3:  # Found good splitting pattern
                stories = [clean_text(story) for story in splits if len(story.strip()) > 100]
                break

    elif story_type == "scifi_stories":
        # Split on sci-fi story markers
        patterns = [
            r'\n\s*(?:Chapter|CHAPTER)\s+\d+',
            r'\n\s*\d+\s*\n\s*[A-Z][^.\n]+\n',
            r'\n\s*(?:In the year|The year|In a distant|On the planet)\b'
        ]

        for pattern in patterns:
            splits = re.split(pattern, text)
            if len(splits) > 3:
                stories = [clean_text(story) for story in splits if len(story.strip()) > 100]
                break

    # Fallback: split by multiple newlines
    if not stories:
        chunks = re.split(r'\n\s*\n\s*\n', text)
        stories = [clean_text(chunk) for chunk in chunks if len(chunk.strip()) > 100]

    # Filter by length
    config = load_config()
    min_length = config['data']['min_story_length']
    max_length = config['data']['max_story_length']

    filtered_stories = []
    for story in stories:
        if min_length <= len(story) <= max_length:
            filtered_stories.append(story)

    return filtered_stories


def assign_age_groups(stories: List[str], story_type: str) -> List[Dict[str, Any]]:
    """Assign age groups to stories based on type and content."""
    story_data = []

    for story in stories:
        # Base age group assignment by story type
        if story_type == "children_stories":
            # Randomly assign between child and kid
            age_group = np.random.choice(['child', 'kid'], p=[0.6, 0.4])
        else:  # scifi_stories
            # Randomly assign between teen and adult
            age_group = np.random.choice(['teen', 'adult'], p=[0.3, 0.7])

        story_data.append({
            'text': story,
            'age_group': age_group,
            'source': story_type,
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
        if group_stories:
            datasets[age_group] = pd.DataFrame(group_stories)
        else:
            datasets[age_group] = pd.DataFrame()

    return datasets


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
        if file_path.exists():
            datasets[age_group] = pd.read_csv(file_path)
        else:
            datasets[age_group] = pd.DataFrame()

    return datasets


def process_raw_data() -> Dict[str, pd.DataFrame]:
    """Main data processing pipeline."""
    config = load_config()
    processed_path = Path(config['paths']['data_processed'])

    # Check if processed data exists
    if processed_path.exists() and not check_cache_overwrite(str(processed_path), "Processed datasets"):
        return load_processed_datasets()

    log_operation_status("Raw data processing")

    # Load raw text files
    raw_texts = load_raw_text_files()

    if not raw_texts:
        print("❌ No raw data found. Please run download_data.py first.")
        return {}

    # Extract stories from each text file
    all_stories = []

    for story_type, text_content in raw_texts.items():
        log_operation_status(f"Extracting stories from {story_type}")

        stories = extract_stories_from_text(text_content, story_type)
        print(f"  📖 Extracted {len(stories)} stories")

        # Assign age groups
        story_data = assign_age_groups(stories, story_type)
        all_stories.extend(story_data)

    print(f"\n📊 Total stories extracted: {len(all_stories)}")

    # Create age-grouped datasets
    datasets = create_age_grouped_datasets(all_stories)

    # Print statistics
    print("\n📈 Age group distribution:")
    for age_group, df in datasets.items():
        print(f"  {age_group}: {len(df)} stories")

    # Save processed datasets
    save_processed_datasets(datasets)

    log_operation_status("Raw data processing", "completed")
    return datasets


def get_dataset_statistics(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Generate statistics about processed datasets."""
    stats = {}

    for age_group, df in datasets.items():
        if not df.empty:
            stats[age_group] = {
                'count': len(df),
                'avg_length': df['length'].mean(),
                'avg_word_count': df['word_count'].mean(),
                'sources': df['source'].value_counts().to_dict()
            }
        else:
            stats[age_group] = {
                'count': 0,
                'avg_length': 0,
                'avg_word_count': 0,
                'sources': {}
            }

    return stats


def main():
    """Main function for data loading and processing."""
    log_operation_status("Data loading and processing")

    # Process raw data
    datasets = process_raw_data()

    if datasets:
        # Generate and display statistics
        stats = get_dataset_statistics(datasets)

        print("\n📊 DATASET STATISTICS")
        print("=" * 50)

        total_stories = 0
        for age_group, group_stats in stats.items():
            count = group_stats['count']
            total_stories += count

            if count > 0:
                print(f"\n{age_group.upper()}:")
                print(f"  Stories: {count:,}")
                print(f"  Avg Length: {group_stats['avg_length']:.0f} chars")
                print(f"  Avg Words: {group_stats['avg_word_count']:.0f}")
                print(f"  Sources: {list(group_stats['sources'].keys())}")

        print(f"\n🎯 TOTAL STORIES: {total_stories:,}")

        if total_stories > 0:
            print("✅ Data processing completed successfully!")
        else:
            print("❌ No stories were processed. Check raw data files.")
    else:
        print("❌ Data processing failed.")

    log_operation_status("Data loading and processing", "completed")


if __name__ == "__main__":
    main()