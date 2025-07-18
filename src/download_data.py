import os
import zipfile
import kaggle
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.helpers import load_config, ensure_dir_exists, check_cache_overwrite, log_operation_status, \
    create_progress_bar


def download_kaggle_datasets() -> bool:
    """Download datasets from Kaggle."""
    load_dotenv()
    config = load_config()

    raw_data_path = Path(config['paths']['data_raw'])
    ensure_dir_exists(raw_data_path)

    datasets = config['datasets']

    for dataset_name, dataset_info in datasets.items():
        log_operation_status(f"Downloading {dataset_name}")

        kaggle_id = dataset_info['kaggle_id']
        expected_files = dataset_info['files']

        # Check if files already exist
        all_exist = all((raw_data_path / f).exists() for f in expected_files)
        if all_exist and not check_cache_overwrite(str(raw_data_path), f"{dataset_name} dataset"):
            continue

        # Download dataset
        temp_dir = raw_data_path / f"temp_{dataset_name}"
        ensure_dir_exists(temp_dir)

        kaggle.api.dataset_download_files(kaggle_id, path=temp_dir, unzip=True)

        # Move expected files to raw data directory
        for file_name in expected_files:
            temp_file = temp_dir / file_name
            if temp_file.exists():
                temp_file.rename(raw_data_path / file_name)
                print(f"  ✅ Downloaded {file_name}")
            else:
                print(f"  ❌ Missing {file_name}")

        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)

    log_operation_status("Kaggle datasets download", "completed")
    return True


def download_base_model() -> bool:
    """Download the base Mixtral model."""
    config = load_config()
    model_name = config['model']['base_model']
    models_path = Path(config['paths']['models'])
    model_path = models_path / "mixtral-8x7b-base"

    ensure_dir_exists(models_path)

    if model_path.exists() and not check_cache_overwrite(str(model_path), "Base model"):
        return True

    log_operation_status("Downloading Mixtral model")

    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.save_pretrained(model_path)
    print("  ✅ Tokenizer downloaded")

    # Download model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None
    )
    model.save_pretrained(model_path)
    print("  ✅ Model downloaded")

    log_operation_status("Base model download", "completed")
    return True


def verify_downloads() -> Dict[str, bool]:
    """Verify all required downloads are complete."""
    config = load_config()
    verification_results = {}

    # Check datasets
    raw_data_path = Path(config['paths']['data_raw'])
    datasets = config['datasets']

    for dataset_name, dataset_info in datasets.items():
        expected_files = dataset_info['files']
        all_exist = all((raw_data_path / f).exists() for f in expected_files)
        verification_results[f"dataset_{dataset_name}"] = all_exist

    # Check base model
    models_path = Path(config['paths']['models'])
    model_path = models_path / "mixtral-8x7b-base"
    verification_results['base_model'] = model_path.exists()

    return verification_results


def main():
    """Main download function."""
    log_operation_status("Data and model download")

    print("📦 Downloading required datasets and models...")

    # Download datasets
    dataset_success = download_kaggle_datasets()

    # Download base model
    model_success = download_base_model()

    # Verify downloads
    verification = verify_downloads()

    print("\n📋 Download Summary:")
    for item, status in verification.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {item}")

    all_success = all(verification.values())
    if all_success:
        print("\n✅ All downloads completed successfully!")
    else:
        print("\n❌ Some downloads failed. Please check the logs above.")

    log_operation_status("Data and model download", "completed")
    return all_success


if __name__ == "__main__":
    main()