import pandas as pd
import os
import sys
import random
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from typing import Dict, List, Any
from collections import defaultdict
from helpers import set_cwd

# Get current working directory for path operations
cwd = set_cwd()

# Add utils to path for imports
sys.path.append(os.path.join(cwd, 'utils'))
from helpers import (
    load_config, ensure_dir_exists, get_age_group,
    check_cache_overwrite, log_operation_status, create_progress_bar
)


def load_tokenizer() -> AutoTokenizer:
    """Load the tokenizer for the base model."""
    config = load_config()
    models_path = config['paths']['models']
    model_path = os.path.join(models_path, "mistral-7b-base")

    # Try local model first, then fall back to HuggingFace
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'], use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def load_processed_datasets() -> Dict[str, pd.DataFrame]:
    """Load processed datasets for tokenization."""
    config = load_config()
    processed_path = config['paths']['data_processed']

    datasets = {}
    age_groups = ['child', 'kid', 'teen', 'adult']

    for age_group in age_groups:
        file_path = os.path.join(processed_path, f"{age_group}_stories.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Clean up any duplicate columns that might exist
            if df.columns.duplicated().any():
                print(f"  Removing duplicate columns from {age_group}")
                df = df.loc[:, ~df.columns.duplicated()]

            datasets[age_group] = df
            print(f"  Loaded {age_group}: {len(df)} stories")
            print(f"    Columns: {list(df.columns)}")
        else:
            print(f"  Missing {age_group} dataset")
            datasets[age_group] = pd.DataFrame()

    return datasets


def format_training_examples_with_metadata(stories_df: pd.DataFrame, age_group: str) -> List[Dict[str, Any]]:
    """Format stories as training examples with age-appropriate instructions using Mistral chat format."""
    age_instructions = {
        'child': "Write a simple story for young children with easy words and short sentences.",
        'kid': "Write an engaging story for children with age-appropriate vocabulary.",
        'teen': "Write a compelling story for teenagers with more complex themes.",
        'adult': "Write a sophisticated story for adults with mature themes and vocabulary."
    }

    instruction = age_instructions[age_group]
    examples = []

    for _, row in stories_df.iterrows():
        story = row['text']

        # Use Mistral's chat format for training
        formatted_text = f"<s>[INST] {instruction}\n\nWrite a story. [/INST]{story}</s>"

        # Preserve metadata for stratification - handle missing columns gracefully
        source = row.get('source', 'unknown')
        if pd.isna(source) or source == '':
            source = 'unknown'

        example = {
            'text': formatted_text,
            'source': str(source),  # Ensure string type
            'age_group': age_group,
            'length': len(story),
            'word_count': len(story.split())
        }

        examples.append(example)

    return examples


def select_hyperparameter_tuning_samples(examples: List[Dict[str, Any]], config: Dict[str, Any]) -> List[
    Dict[str, Any]]:
    """Add hyperparameter tuning selection indicator to examples."""

    # Get tuning probability from config
    tuning_config = config.get('hyperparameter_tuning', {})
    sample_percentage = tuning_config.get('sample_percentage', 5)
    tuning_probability = sample_percentage / 100.0

    print(f"\nSelecting hyperparameter tuning samples:")
    print(f"  Sample probability: {sample_percentage}% ({tuning_probability:.3f})")

    # Set random seed for reproducibility
    random.seed(42)

    # Track selections by strata
    strata_selections = defaultdict(lambda: {'total': 0, 'selected': 0})

    for example in examples:
        # Each example has a chance to be selected for hyperparameter tuning
        is_tuning_sample = random.random() < tuning_probability
        example['hyperparameter_tuning'] = is_tuning_sample

        # Track statistics
        stratum = (example['source'], example['age_group'])
        strata_selections[stratum]['total'] += 1
        if is_tuning_sample:
            strata_selections[stratum]['selected'] += 1

    # Report selection statistics
    total_selected = sum(1 for ex in examples if ex['hyperparameter_tuning'])
    total_examples = len(examples)
    actual_percentage = (total_selected / total_examples) * 100

    print(f"  Total examples: {total_examples:,}")
    print(f"  Selected for tuning: {total_selected:,} ({actual_percentage:.1f}%)")

    print(f"\n  Selection by strata:")
    for (source, age_group), stats in sorted(strata_selections.items()):
        selected = stats['selected']
        total = stats['total']
        percentage = (selected / total) * 100 if total > 0 else 0
        print(f"    {source:20} | {age_group:8} | {selected:4}/{total:4} ({percentage:4.1f}%)")

    return examples


def tokenize_examples_with_metadata(examples: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int) -> List[
    Dict[str, Any]]:
    """Tokenize training examples while preserving metadata."""
    tokenized_examples = []

    print(f"Tokenizing {len(examples)} examples...")

    for example in create_progress_bar(examples, "Tokenizing"):
        tokens = tokenizer(
            example['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        tokenized_example = {
            'input_ids': tokens['input_ids'][0].tolist(),
            'attention_mask': tokens['attention_mask'][0].tolist(),
            'labels': tokens['input_ids'][0].tolist(),  # Add labels here
            'source': example['source'],
            'age_group': example['age_group'],
            'length': example['length'],
            'word_count': example['word_count'],
            'hyperparameter_tuning': example['hyperparameter_tuning']  # Preserve tuning indicator
        }

        tokenized_examples.append(tokenized_example)

    return tokenized_examples


def create_stratified_dataset_splits(tokenized_data: List[Dict[str, Any]], config: Dict[str, Any]) -> DatasetDict:
    """Create stratified train/validation/test splits preserving source and age_group distribution."""

    # Group examples by strata (source, age_group)
    strata_groups = defaultdict(list)
    for i, example in enumerate(tokenized_data):
        stratum = (example['source'], example['age_group'])
        strata_groups[stratum].append(i)

    print(f"\nStratified split creation:")
    print(f"Found {len(strata_groups)} strata:")
    for stratum, indices in strata_groups.items():
        source, age_group = stratum
        print(f"  {source:25} | {age_group:8} | {len(indices):6} examples")

    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']

    train_indices = []
    val_indices = []
    test_indices = []

    # Split each stratum proportionally
    random.seed(42)  # For reproducibility

    for stratum, indices in strata_groups.items():
        random.shuffle(indices)
        n_examples = len(indices)

        # Calculate splits ensuring at least 1 example per split if stratum is large enough
        n_test = max(1, int(n_examples * test_split)) if n_examples >= 3 else 0
        n_val = max(1, int(n_examples * val_split)) if n_examples >= 2 else 0
        n_train = n_examples - n_test - n_val

        # Ensure we don't have negative training examples
        if n_train < 0:
            n_train = n_examples
            n_val = 0
            n_test = 0

        if n_test > 0:
            test_indices.extend(indices[:n_test])
        if n_val > 0:
            val_indices.extend(indices[n_test:n_test + n_val])
        if n_train > 0:
            train_indices.extend(indices[n_test + n_val:])

        source, age_group = stratum
        print(f"  {source:25} | {age_group:8} | train={n_train:4}, val={n_val:3}, test={n_test:3}")

    # Create datasets from indices
    train_data = [tokenized_data[i] for i in train_indices]
    val_data = [tokenized_data[i] for i in val_indices]
    test_data = [tokenized_data[i] for i in test_indices]

    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_data):,} examples")
    print(f"  Validation: {len(val_data):,} examples")
    print(f"  Test: {len(test_data):,} examples")

    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })

    return dataset_dict


def verify_stratification_and_tuning_samples(datasets: DatasetDict) -> None:
    """Verify that stratification was preserved and show hyperparameter tuning sample distribution."""
    print("\nSTRATIFICATION AND TUNING SAMPLE VERIFICATION:")
    print("=" * 60)

    for split_name, dataset in datasets.items():
        print(f"\n{split_name.upper()} SET ({len(dataset):,} examples):")
        strata_counts = defaultdict(int)
        tuning_counts = defaultdict(int)

        for example in dataset:
            source = example.get('source', 'unknown')
            age_group = example.get('age_group', 'unknown')
            is_tuning = example.get('hyperparameter_tuning', False)

            strata_counts[(source, age_group)] += 1
            if is_tuning:
                tuning_counts[(source, age_group)] += 1

        total = len(dataset)
        total_tuning = sum(tuning_counts.values())

        print(f"  Total tuning samples: {total_tuning:,} ({(total_tuning / total) * 100:.1f}%)")
        print(f"  Stratification and tuning distribution:")

        for (source, age_group), count in sorted(strata_counts.items()):
            percentage = (count / total) * 100 if total > 0 else 0
            tuning_count = tuning_counts.get((source, age_group), 0)
            tuning_pct = (tuning_count / count) * 100 if count > 0 else 0
            print(
                f"    {source:20} | {age_group:8} | {count:6} ({percentage:4.1f}%) | tuning: {tuning_count:3} ({tuning_pct:4.1f}%)")


def save_tokenized_datasets(datasets: DatasetDict, tokenizer: AutoTokenizer) -> None:
    """Save tokenized datasets to disk."""
    config = load_config()
    tokenized_path = config['paths']['data_tokenized']
    ensure_dir_exists(tokenized_path)

    datasets.save_to_disk(os.path.join(tokenized_path, "datasets"))
    tokenizer.save_pretrained(os.path.join(tokenized_path, "tokenizer"))

    print(f"  Saved tokenized datasets to {tokenized_path}")


def load_tokenized_datasets() -> DatasetDict:
    """Load tokenized datasets from disk."""
    config = load_config()
    tokenized_path = config['paths']['data_tokenized']
    dataset_path = os.path.join(tokenized_path, "datasets")

    if os.path.exists(dataset_path):
        return DatasetDict.load_from_disk(dataset_path)
    else:
        return None


def tokenize_all_datasets() -> DatasetDict:
    """Main tokenization pipeline with metadata preservation and tuning sample selection."""
    config = load_config()
    tokenized_path = config['paths']['data_tokenized']

    if os.path.exists(tokenized_path) and not check_cache_overwrite(tokenized_path, "Tokenized datasets"):
        existing_datasets = load_tokenized_datasets()
        if existing_datasets:
            verify_stratification_and_tuning_samples(existing_datasets)
            return existing_datasets

    log_operation_status("Dataset tokenization with Mistral format")

    tokenizer = load_tokenizer()
    print(f"  Loaded tokenizer: {tokenizer.__class__.__name__}")

    datasets = load_processed_datasets()

    if not any(not df.empty for df in datasets.values()):
        print("No processed datasets found. Please run data_loader.py first.")
        return None

    all_examples = []

    for age_group, df in datasets.items():
        if not df.empty:
            log_operation_status(f"Processing {age_group} stories")

            # Check for required columns
            if 'text' not in df.columns:
                print(f"  Missing 'text' column in {age_group}")
                continue

            # Ensure source column exists
            if 'source' not in df.columns:
                print(f"  Adding missing 'source' column to {age_group}")
                df['source'] = 'processed_stories'

            examples = format_training_examples_with_metadata(df, age_group)
            all_examples.extend(examples)

            # Show source distribution for this age group
            source_counts = defaultdict(int)
            for example in examples:
                source_counts[example['source']] += 1

            print(f"  Formatted {len(examples)} {age_group} examples")
            print(f"    Sources: {dict(source_counts)}")

    print(f"\nTotal training examples: {len(all_examples):,}")

    # Show overall stratification before tokenization
    print(f"\nPre-tokenization stratification:")
    strata_counts = defaultdict(int)
    for example in all_examples:
        source = example['source']
        age_group = example['age_group']
        strata_counts[(source, age_group)] += 1

    total = len(all_examples)
    for (source, age_group), count in sorted(strata_counts.items()):
        percentage = (count / total) * 100
        print(f"  {source:25} | {age_group:8} | {count:6} ({percentage:5.1f}%)")

    # Select hyperparameter tuning samples
    all_examples = select_hyperparameter_tuning_samples(all_examples, config)

    log_operation_status("Tokenizing examples with metadata")
    max_length = config['data']['max_sequence_length']
    tokenized_data = tokenize_examples_with_metadata(all_examples, tokenizer, max_length)

    log_operation_status("Creating stratified dataset splits")
    dataset_splits = create_stratified_dataset_splits(tokenized_data, config)

    # Verify stratification and tuning samples in final splits
    verify_stratification_and_tuning_samples(dataset_splits)

    save_tokenized_datasets(dataset_splits, tokenizer)

    log_operation_status("Dataset tokenization", "completed")
    return dataset_splits


def get_tokenization_statistics(datasets: DatasetDict) -> Dict[str, Any]:
    """Generate statistics about tokenized datasets including tuning sample info."""
    if not datasets:
        return {}

    stats = {
        'total_examples': sum(len(dataset) for dataset in datasets.values()),
        'splits': {split: len(dataset) for split, dataset in datasets.items()}
    }

    if len(datasets['train']) > 0:
        sample_lengths = []
        for i in range(min(1000, len(datasets['train']))):
            example = datasets['train'][i]
            length = sum(1 for token_id in example['input_ids'] if token_id != 0)
            sample_lengths.append(length)

        stats['token_statistics'] = {
            'avg_length': sum(sample_lengths) / len(sample_lengths),
            'max_length': max(sample_lengths),
            'min_length': min(sample_lengths)
        }

    # Stratification and tuning statistics
    stats['stratification'] = {}
    stats['tuning_samples'] = {}

    for split_name, dataset in datasets.items():
        strata_counts = defaultdict(int)
        tuning_counts = defaultdict(int)

        for example in dataset:
            source = example.get('source', 'unknown')
            age_group = example.get('age_group', 'unknown')
            is_tuning = example.get('hyperparameter_tuning', False)

            strata_counts[(source, age_group)] += 1
            if is_tuning:
                tuning_counts[(source, age_group)] += 1

        stats['stratification'][split_name] = dict(strata_counts)
        stats['tuning_samples'][split_name] = dict(tuning_counts)

    return stats


def main():
    """Main function for tokenization."""
    log_operation_status("Data tokenization for Mistral 7B with stratification and tuning sample selection")

    datasets = tokenize_all_datasets()

    if datasets:
        stats = get_tokenization_statistics(datasets)

        print("\nTOKENIZATION STATISTICS")
        print("=" * 50)
        print(f"Total examples: {stats['total_examples']:,}")

        for split, count in stats['splits'].items():
            percentage = (count / stats['total_examples']) * 100
            tuning_count = sum(stats['tuning_samples'][split].values())
            tuning_pct = (tuning_count / count) * 100 if count > 0 else 0
            print(f"{split}: {count:,} ({percentage:.1f}%) | tuning samples: {tuning_count:,} ({tuning_pct:.1f}%)")

        if 'token_statistics' in stats:
            token_stats = stats['token_statistics']
            print(f"\nToken length statistics:")
            print(f"  Average: {token_stats['avg_length']:.1f}")
            print(f"  Range: {token_stats['min_length']}-{token_stats['max_length']}")

        print(f"\nFinal stratification and tuning sample summary:")
        for split_name, strata in stats['stratification'].items():
            print(f"\n{split_name.upper()}:")
            total_split = sum(strata.values())
            tuning_split = stats['tuning_samples'][split_name]

            for (source, age_group), count in sorted(strata.items()):
                percentage = (count / total_split) * 100 if total_split > 0 else 0
                tuning_count = tuning_split.get((source, age_group), 0)
                tuning_pct = (tuning_count / count) * 100 if count > 0 else 0
                print(
                    f"  {source:20} | {age_group:8} | {count:6} ({percentage:4.1f}%) | tuning: {tuning_count:3} ({tuning_pct:4.1f}%)")

        print("\nTokenization completed successfully!")
        print("Ready for Mistral 7B training with chat format")
    else:
        print("Tokenization failed.")

    log_operation_status("Data tokenization", "completed")


if __name__ == "__main__":
    main()