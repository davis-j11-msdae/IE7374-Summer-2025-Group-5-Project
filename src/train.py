import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import DatasetDict
import deepspeed
from pathlib import Path
from typing import Dict, Any
from utils.helpers import (
    load_config, ensure_dir_exists, save_pickle,
    check_cache_overwrite, log_operation_status
)


def load_base_model_and_tokenizer():
    """Load the base Mixtral model and tokenizer."""
    config = load_config()
    models_path = Path(config['paths']['models'])
    model_path = models_path / "mixtral-8x7b-base"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_tokenized_datasets() -> DatasetDict:
    """Load tokenized datasets for training."""
    config = load_config()
    tokenized_path = Path(config['paths']['data_tokenized'])
    dataset_path = tokenized_path / "datasets"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Tokenized datasets not found at {dataset_path}")

    return DatasetDict.load_from_disk(dataset_path)


def setup_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """Setup training arguments for fine-tuning."""
    training_config = config['training']

    training_args = TrainingArguments(
        output_dir=Path(config['paths']['models']) / "checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=training_config['epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        eval_steps=training_config['eval_steps'],
        save_steps=training_config['save_steps'],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to=None,
        remove_unused_columns=False,
        max_grad_norm=training_config['max_grad_norm'],
        deepspeed=config['deepspeed']['config_path'] if config['deepspeed']['enabled'] else None
    )

    return training_args


def create_trainer(model, tokenizer, datasets: DatasetDict, training_args: TrainingArguments) -> Trainer:
    """Create Hugging Face trainer."""

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    return trainer


def train_storytelling_model():
    """Main training pipeline."""
    config = load_config()
    models_path = Path(config['paths']['models'])
    output_model_path = models_path / "tuned_story_llm"

    # Check if model already exists
    if output_model_path.exists() and not check_cache_overwrite(str(output_model_path), "Trained model"):
        print(f"✅ Using existing trained model at {output_model_path}")
        return True

    log_operation_status("Model training")

    # Load base model and tokenizer
    print("📥 Loading base model and tokenizer...")
    model, tokenizer = load_base_model_and_tokenizer()
    print(f"  ✅ Loaded {model.__class__.__name__}")

    # Load tokenized datasets
    print("📊 Loading tokenized datasets...")
    datasets = load_tokenized_datasets()
    print(f"  ✅ Loaded datasets:")
    for split, dataset in datasets.items():
        print(f"    {split}: {len(dataset):,} examples")

    # Setup training arguments
    training_args = setup_training_arguments(config)
    print(f"  ✅ Training arguments configured")
    print(f"    Epochs: {training_args.num_train_epochs}")
    print(f"    Batch size: {training_args.per_device_train_batch_size}")
    print(f"    Learning rate: {training_args.learning_rate}")

    # Create trainer
    trainer = create_trainer(model, tokenizer, datasets, training_args)
    print(f"  ✅ Trainer created")

    # Start training
    print(f"\n🚀 Starting training...")
    print(
        f"Expected training time: ~{training_args.num_train_epochs * len(datasets['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) // 60} minutes")

    trainer.train()

    # Save the fine-tuned model
    print(f"\n💾 Saving fine-tuned model...")
    ensure_dir_exists(output_model_path)

    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(datasets['test'])
    print(f"Test results: {test_results}")

    # Save model metadata
    model_metadata = {
        'model_path': str(output_model_path),
        'base_model': config['model']['base_model'],
        'training_config': config['training'],
        'test_results': test_results,
        'datasets_info': {
            'train_size': len(datasets['train']),
            'val_size': len(datasets['validation']),
            'test_size': len(datasets['test'])
        }
    }

    metadata_file = Path(config['paths']['outputs']) / "model_metadata.pkl"
    save_pickle(model_metadata, metadata_file)

    print(f"✅ Model training completed!")
    print(f"📁 Model saved to: {output_model_path}")
    print(f"📊 Metadata saved to: {metadata_file}")

    log_operation_status("Model training", "completed")
    return True


def validate_training_setup() -> bool:
    """Validate that all prerequisites for training are met."""
    config = load_config()

    # Check if base model exists
    models_path = Path(config['paths']['models'])
    base_model_path = models_path / "mixtral-8x7b-base"

    if not base_model_path.exists():
        print(f"❌ Base model not found at {base_model_path}")
        print("Please run download_data.py first.")
        return False

    # Check if tokenized datasets exist
    tokenized_path = Path(config['paths']['data_tokenized'])
    dataset_path = tokenized_path / "datasets"

    if not dataset_path.exists():
        print(f"❌ Tokenized datasets not found at {dataset_path}")
        print("Please run data_tokenizer.py first.")
        return False

    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. Training will be very slow on CPU.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    else:
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✅ GPU available: {gpu_count} device(s), {gpu_memory:.1f}GB memory")

        if gpu_memory < 40:
            print("⚠️ Limited GPU memory. Consider reducing batch size or using DeepSpeed.")

    # Check DeepSpeed configuration
    if config['deepspeed']['enabled']:
        deepspeed_config_path = Path(config['deepspeed']['config_path'])
        if not deepspeed_config_path.exists():
            print(f"❌ DeepSpeed config not found at {deepspeed_config_path}")
            return False
        else:
            print(f"✅ DeepSpeed configuration found")

    return True


def main():
    """Main function for training."""
    log_operation_status("Training setup validation")

    # Validate setup
    if not validate_training_setup():
        print("❌ Training setup validation failed.")
        return

    print("✅ Training setup validated")

    # Train model
    success = train_storytelling_model()

    if success:
        print("\n🎉 Training pipeline completed successfully!")
    else:
        print("\n❌ Training pipeline failed.")


if __name__ == "__main__":
    main()