import pandas as pd
import torch
import os
import sys
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from typing import Dict, List, Any
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
    model_path = os.path.join(models_path, "mixtral-8x7b-base")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

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
            datasets[age_group] = pd.read_csv(file_path)
            print(f"  ✅ Loaded {age_group}: {len(datasets[age_group])} stories")
        else:
            print(f"  ❌ Missing {age_group} dataset")
            datasets[age_group] = pd.DataFrame()

    return datasets


def format_training_examples(stories_df: pd.DataFrame, age_group: str) -> List[str]:
    """Format stories as training examples with age-appropriate instructions."""
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
        formatted_example = f"{instruction}\n\nStory: {story}"
        examples.append(formatted_example)

    return examples


def tokenize_examples(examples: List[str], tokenizer: AutoTokenizer, max_length: int) -> List[Dict[str, Any]]:
    """Tokenize training examples."""
    tokenized_examples = []

    for example in create_progress_bar(examples, "Tokenizing"):
        tokens = tokenizer(
            example,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        tokenized_example = {
            'input_ids': tokens['input_ids'][0].tolist(),
            'attention_mask': tokens['attention_mask'][0].tolist()
        }

        tokenized_example['labels'] = tokenized_example['input_ids'].copy()

        tokenized_examples.append(tokenized_example)

    return tokenized_examples


def create_dataset_splits(tokenized_data: List[Dict[str, Any]], config: Dict[str, Any]) -> DatasetDict:
    """Create train/validation/test splits."""
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']

    train_val_data, test_data = train_test_split(
        tokenized_data,
        test_size=test_split,
        random_state=42
    )

    val_ratio = val_split / (train_split + val_split)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_ratio,
        random_state=42
    )

    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })

    return dataset_dict


def save_tokenized_datasets(datasets: DatasetDict, tokenizer: AutoTokenizer) -> None:
    """Save tokenized datasets to disk."""
    config = load_config()
    tokenized_path = config['paths']['data_tokenized']
    ensure_dir_exists(tokenized_path)

    datasets.save_to_disk(os.path.join(tokenized_path, "datasets"))

    tokenizer.save_pretrained(os.path.join(tokenized_path, "tokenizer"))

    print(f"  ✅ Saved tokenized datasets to {tokenized_path}")


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
    """Main tokenization pipeline."""
    config = load_config()
    tokenized_path = config['paths']['data_tokenized']

    if os.path.exists(tokenized_path) and not check_cache_overwrite(tokenized_path, "Tokenized datasets"):
        existing_datasets = load_tokenized_datasets()
        if existing_datasets:
            return existing_datasets

    log_operation_status("Dataset tokenization")

    tokenizer = load_tokenizer()
    print(f"  ✅ Loaded tokenizer: {tokenizer.__class__.__name__}")

    datasets = load_processed_datasets()

    if not any(not df.empty for df in datasets.values()):
        print("❌ No processed datasets found. Please run data_loader.py first.")
        return None

    all_examples = []

    for age_group, df in datasets.items():
        if not df.empty:
            log_operation_status(f"Processing {age_group} stories")
            examples = format_training_examples(df, age_group)
            all_examples.extend(examples)
            print(f"  ✅ Formatted {len(examples)} {age_group} examples")

    print(f"\n📊 Total training examples: {len(all_examples)}")

    log_operation_status("Tokenizing examples")
    max_length = config['data']['max_sequence_length']
    tokenized_data = tokenize_examples(all_examples, tokenizer, max_length)

    log_operation_status("Creating dataset splits")
    dataset_splits = create_dataset_splits(tokenized_data, config)

    print("\n📈 Dataset splits:")
    for split, dataset in dataset_splits.items():
        print(f"  {split}: {len(dataset):,} examples")

    save_tokenized_datasets(dataset_splits, tokenizer)

    log_operation_status("Dataset tokenization", "completed")
    return dataset_splits


def get_tokenization_statistics(datasets: DatasetDict) -> Dict[str, Any]:
    """Generate statistics about tokenized datasets."""
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

    return stats


def main():
    """Main function for tokenization."""
    log_operation_status("Data tokenization")

    datasets = tokenize_all_datasets()

    if datasets:
        stats = get_tokenization_statistics(datasets)

        print("\n📊 TOKENIZATION STATISTICS")
        print("=" * 50)
        print(f"Total examples: {stats['total_examples']:,}")

        for split, count in stats['splits'].items():
            percentage = (count / stats['total_examples']) * 100
            print(f"{split}: {count:,} ({percentage:.1f}%)")

        if 'token_statistics' in stats:
            token_stats = stats['token_statistics']
            print(f"\nToken length statistics:")
            print(f"  Average: {token_stats['avg_length']:.1f}")
            print(f"  Range: {token_stats['min_length']}-{token_stats['max_length']}")

        print("\n✅ Tokenization completed successfully!")
    else:
        print("❌ Tokenization failed.")

    log_operation_status("Data tokenization", "completed")


if __name__ == "__main__":
    main()