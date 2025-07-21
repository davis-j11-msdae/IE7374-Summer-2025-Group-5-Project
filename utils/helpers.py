import os
import json
import yaml
import pickle
import numpy as np
from typing import Dict, List, Any
import re
from datetime import datetime
import importlib.util

def set_cwd():
    if importlib.util.find_spec("google.colab") is not None:
        cwd = os.getcwd()
    else:
        cwd = os.getcwd().rstrip(r"\src")
    os.chdir(cwd)
    return cwd

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(set_cwd(), "configs", "model_config.yaml")

    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def ensure_dir_exists(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def clean_text(text: str) -> str:
    """Clean and preprocess text data."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,!?;:\-\'\"()]', '', text)
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    return text.strip()


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file."""
    ensure_dir_exists(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file."""
    ensure_dir_exists(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_age_group(age: int) -> str:
    """Determine age group based on age."""
    if age <= 5:
        return "child"
    elif age <= 12:
        return "kid"
    elif age <= 17:
        return "teen"
    else:
        return "adult"


def calculate_text_stats(text: str) -> Dict[str, Any]:
    """Calculate basic text statistics."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)

    return {
        'length': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0
    }


def format_prompt_with_history(prompt: str, history: List[str]) -> str:
    """Format prompt with user history."""
    if not history:
        return prompt

    history_text = "\n".join([f"Previous story context: {h}" for h in history])
    return f"{history_text}\n\nNew prompt: {prompt}"


def validate_story_length(text: str, min_length: int = 100, max_length: int = 5000) -> bool:
    """Validate story length requirements."""
    return min_length <= len(text) <= max_length


def batch_process(items: List[Any], batch_size: int = 32):
    """Process items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def check_cache_overwrite(filepath: str, operation_name: str) -> bool:
    """Check if cached file exists and prompt for overwrite."""
    if os.path.exists(filepath):
        response = input(f"\n{operation_name} cache found at {filepath}. Overwrite? (y/N): ")
        return response.lower() == 'y'
    return True


def create_progress_bar(iterable, desc: str):
    """Create progress bar for long operations."""
    from tqdm import tqdm
    return tqdm(iterable, desc=desc)


def log_operation_status(operation: str, status: str = "started"):
    """Log operation status with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {operation} {status}")