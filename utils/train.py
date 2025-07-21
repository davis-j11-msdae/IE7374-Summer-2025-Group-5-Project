import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import DatasetDict
from pathlib import Path
from typing import Dict, Any
from utils.helpers import (
    load_config, ensure_dir_exists, save_pickle,
    log_operation_status
)

def load_base_model_and_tokenizer():
    """Load the base Mixtral model and tokenizer."""
    config = load_config()
    models_path = Path(config['paths']['models'])
    model_path = models_path / "mixtral-8x7b-base"
    
    print(f"📥 Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"📥 Loading model from {model_path} (this may take several minutes)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False
    )
    
    print(f"🔧 Resizing token embeddings")
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def load_tokenized_datasets() -> DatasetDict:
    """Load tokenized datasets for training."""
    config = load_config()
    tokenized_path = Path(config['paths']['data_tokenized'])
    dataset_path = tokenized_path / "datasets"
    
    print(f"📊 Loading datasets from {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Tokenized datasets not found at {dataset_path}")
    
    # Force file:// protocol to avoid ambiguity
    absolute_path = dataset_path.resolve()
    file_url = f"file://{absolute_path}"
    print(f"📊 Using file URL: {file_url}")
    
    from datasets import DatasetDict
    try:
        return DatasetDict.load_from_disk(file_url)
    except:
        # Fallback: try without protocol
        print(f"📊 Fallback: trying direct path")
        return DatasetDict.load_from_disk(str(absolute_path))

def setup_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """Setup training arguments for fine-tuning."""
    training_config = config['training']
    
    deepspeed_config = None
    if config['deepspeed']['enabled']:
        deepspeed_config_path = Path(config['deepspeed']['config_path'])
        if deepspeed_config_path.exists():
            deepspeed_config = str(deepspeed_config_path)
            print(f"🚀 Using DeepSpeed config: {deepspeed_config}")
    
    training_args = TrainingArguments(
        output_dir=str(Path(config['paths']['models']) / "checkpoints"),
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
        eval_strategy="steps",
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
        deepspeed=deepspeed_config
    )
    
    return training_args

def create_trainer(model, tokenizer, datasets: DatasetDict, training_args: TrainingArguments) -> Trainer:
    """Create Hugging Face trainer."""
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    return trainer

def train_storytelling_model():
    """Main training pipeline."""
    config = load_config()
    models_path = Path(config['paths']['models'])
    output_model_path = models_path / "tuned_story_llm"
    
    if output_model_path.exists():
        response = input(f"\nTrained model exists at {output_model_path}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print(f"✅ Using existing trained model")
            return True
    
    log_operation_status("Model training")
    
    datasets = load_tokenized_datasets()
    print(f"✅ Loaded datasets:")
    for split, dataset in datasets.items():
        print(f"  {split}: {len(dataset):,} examples")
    
    model, tokenizer = load_base_model_and_tokenizer()
    print(f"✅ Loaded {model.__class__.__name__}")
    
    training_args = setup_training_arguments(config)
    print(f"✅ Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    
    trainer = create_trainer(model, tokenizer, datasets, training_args)
    print(f"✅ Trainer created")
    
    estimated_steps = len(datasets['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    estimated_time = (estimated_steps * training_args.num_train_epochs) // 60
    print(f"🚀 Starting training (~{estimated_time} minutes)")
    
    trainer.train()
    
    print(f"💾 Saving fine-tuned model")
    ensure_dir_exists(str(output_model_path))
    
    trainer.save_model(str(output_model_path))
    tokenizer.save_pretrained(str(output_model_path))
    
    print(f"📊 Evaluating on test set")
    test_results = trainer.evaluate(datasets['test'])
    print(f"Test loss: {test_results['eval_loss']:.4f}")
    
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
    save_pickle(model_metadata, str(metadata_file))
    
    print(f"✅ Training completed")
    log_operation_status("Model training", "completed")
    return True

def validate_training_setup() -> bool:
    """Validate training prerequisites."""
    config = load_config()
    
    models_path = Path(config['paths']['models'])
    base_model_path = models_path / "mixtral-8x7b-base"
    
    if not base_model_path.exists():
        print(f"❌ Base model not found at {base_model_path}")
        return False
    
    tokenized_path = Path(config['paths']['data_tokenized'])
    dataset_path = tokenized_path / "datasets"
    
    if not dataset_path.exists():
        print(f"❌ Tokenized datasets not found at {dataset_path}")
        return False
    
    gpu_count = torch.cuda.device_count()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ GPU: {gpu_count} device(s), {gpu_memory:.1f}GB memory")
    
    if config['deepspeed']['enabled']:
        deepspeed_config_path = Path(config['deepspeed']['config_path'])
        if not deepspeed_config_path.exists():
            print(f"❌ DeepSpeed config not found")
            return False
        print(f"✅ DeepSpeed configuration found")
    
    return True

def main():
    """Main training function."""
    log_operation_status("Training setup validation")
    
    if not validate_training_setup():
        print("❌ Training setup validation failed")
        return
    
    print("✅ Training setup validated")
    
    success = train_storytelling_model()
    
    if success:
        print("🎉 Training pipeline completed successfully!")
    else:
        print("❌ Training pipeline failed")

if __name__ == "__main__":
    main()