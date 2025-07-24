import pandas as pd
import torch
import os
import sys
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from typing import Dict, List, Any
from helpers import set_cwd
from collections import defaultdict

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


def format_training_examples_with_metadata(stories_df: pd.DataFrame, age_group: str) -> List[Dict[str, Any]]:
    """Format stories as training examples with age-appropriate instructions and preserve metadata."""
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
        formatted_text = f"{instruction}\n\nStory: {story}"
        
        # Preserve metadata for stratification
        example = {
            'text': formatted_text,
            'source': row.get('source', 'unknown'),
            'age_group': age_group,
            'length': len(story),
            'word_count': len(story.split())
        }
        
        examples.append(example)

    return examples


def tokenize_examples_with_metadata(examples: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int) -> List[Dict[str, Any]]:
    """Tokenize training examples while preserving metadata."""
    tokenized_examples = []

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
            'source': example['source'],
            'age_group': example['age_group'],
            'length': example['length'],
            'word_count': example['word_count']
        }

        tokenized_example['labels'] = tokenized_example['input_ids'].copy()
        tokenized_examples.append(tokenized_example)

    return tokenized_examples


def create_stratified_dataset_splits(tokenized_data: List[Dict[str, Any]], config: Dict[str, Any]) -> DatasetDict:
    """Create stratified train/validation/test splits preserving source and age_group distribution."""
    from collections import defaultdict
    import random
    
    # Group examples by strata (source, age_group)
    strata_groups = defaultdict(list)
    for i, example in enumerate(tokenized_data):
        stratum = (example['source'], example['age_group'])
        strata_groups[stratum].append(i)
    
    print(f"\nStratified split creation:")
    print(f"Found {len(strata_groups)} strata:")
    for stratum, indices in strata_groups.items():
        source, age_group = stratum
        print(f"  {source} | {age_group}: {len(indices)} examples")
    
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
        
        n_test = max(1, int(n_examples * test_split))
        n_val = max(1, int(n_examples * val_split))
        n_train = n_examples - n_test - n_val
        
        test_indices.extend(indices[:n_test])
        val_indices.extend(indices[n_test:n_test + n_val])
        train_indices.extend(indices[n_test + n_val:])
        
        source, age_group = stratum
        print(f"  {source} | {age_group}: train={n_train}, val={n_val}, test={n_test}")
    
    # Create datasets from indices
    train_data = [tokenized_data[i] for i in train_indices]
    val_data = [tokenized_data[i] for i in val_indices]
    test_data = [tokenized_data[i] for i in test_indices]
    
    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")

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


def verify_stratification(datasets: DatasetDict) -> None:
    """Verify that stratification was preserved in the datasets."""
    print("\n🔍 Verifying stratification preservation:")
    
    for split_name, dataset in datasets.items():
        print(f"\n{split_name.upper()} SET:")
        strata_counts = defaultdict(int)
        
        for example in dataset:
            source = example.get('source', 'unknown')
            age_group = example.get('age_group', 'unknown')
            strata_counts[(source, age_group)] += 1
        
        total = len(dataset)
        for (source, age_group), count in sorted(strata_counts.items()):
            percentage = (count / total) * 100
            print(f"  {source:25} | {age_group:8} | {count:6} ({percentage:5.1f}%)")


def tokenize_all_datasets() -> DatasetDict:
    """Main tokenization pipeline with metadata preservation."""
    config = load_config()
    tokenized_path = config['paths']['data_tokenized']

    if os.path.exists(tokenized_path) and not check_cache_overwrite(tokenized_path, "Tokenized datasets"):
        existing_datasets = load_tokenized_datasets()
        if existing_datasets:
            verify_stratification(existing_datasets)
            return existing_datasets

    log_operation_status("Dataset tokenization with metadata preservation")

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
            
            # Check required columns
            required_columns = ['text', 'source']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"  ⚠️ Missing columns in {age_group}: {missing_columns}")
                print(f"  Available columns: {list(df.columns)}")
                # Fill missing source column if needed
                if 'source' not in df.columns:
                    df['source'] = 'unknown'
            
            examples = format_training_examples_with_metadata(df, age_group)
            all_examples.extend(examples)
            print(f"  ✅ Formatted {len(examples)} {age_group} examples")

    print(f"\n📊 Total training examples: {len(all_examples)}")

    # Show stratification before tokenization
    print(f"\n📊 Pre-tokenization stratification:")
    strata_counts = defaultdict(int)
    for example in all_examples:
        source = example['source']
        age_group = example['age_group']
        strata_counts[(source, age_group)] += 1
    
    total = len(all_examples)
    for (source, age_group), count in sorted(strata_counts.items()):
        percentage = (count / total) * 100
        print(f"  {source:25} | {age_group:8} | {count:6} ({percentage:5.1f}%)")

    log_operation_status("Tokenizing examples with metadata")
    max_length = config['data']['max_sequence_length']
    tokenized_data = tokenize_examples_with_metadata(all_examples, tokenizer, max_length)

    log_operation_status("Creating stratified dataset splits")
    dataset_splits = create_stratified_dataset_splits(tokenized_data, config)

    # Verify stratification in final splits
    verify_stratification(dataset_splits)

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

    # Stratification statistics
    stats['stratification'] = {}
    for split_name, dataset in datasets.items():
        strata_counts = defaultdict(int)
        for example in dataset:
            source = example.get('source', 'unknown')
            age_group = example.get('age_group', 'unknown')
            strata_counts[(source, age_group)] += 1
        stats['stratification'][split_name] = dict(strata_counts)

    return stats


def main():
    """Main function for tokenization."""
    log_operation_status("Data tokenization with stratification")

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

        print(f"\nStratification verification:")
        for split_name, strata in stats['stratification'].items():
            print(f"\n{split_name}:")
            total_split = sum(strata.values())
            for (source, age_group), count in sorted(strata.items()):
                if isinstance(source, tuple):  # Handle if source is stored as tuple
                    source, age_group = source
                percentage = (count / total_split) * 100
                print(f"  {source:20} | {age_group:8} | {count:6} ({percentage:5.1f}%)")

        print("\n✅ Tokenization completed successfully!")
    else:
        print("❌ Tokenization failed.")

    log_operation_status("Data tokenization", "completed")


if __name__ == "__main__":
    main()