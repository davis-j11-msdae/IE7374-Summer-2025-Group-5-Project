import os
import sys
import yaml
import json
import shutil
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling, BitsAndBytesConfig, EarlyStoppingCallback
)
from transformers.integrations import TensorBoardCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
from collections import defaultdict

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import set_cwd
from hyperparameter_tuning import run_hyperparameter_tuning, StratifiedSampler, verify_dataset_stratification

# Get current working directory for path operations
cwd = set_cwd()

def check_training_state(output_dir):
    """Check if previous training exists and its completion state."""
    if not os.path.exists(output_dir):
        return "new", None
    
    trainer_state_file = os.path.join(output_dir, "trainer_state.json")
    training_info_file = os.path.join(output_dir, "training_info.json")
    
    if not os.path.exists(trainer_state_file):
        return "new", None
    
    training_info = {}
    if os.path.exists(training_info_file):
        with open(training_info_file, 'r') as f:
            training_info = json.load(f)
    
    with open(trainer_state_file, 'r') as f:
        trainer_state = json.load(f)
    
    early_stopped = training_info.get('early_stopped', False)
    original_epochs = training_info.get('original_epochs', 0)
    completed_epochs = trainer_state.get('epoch', 0)
    max_additional_epochs = 3
    
    if early_stopped:
        return "early_stopped", training_info
    elif completed_epochs < original_epochs + max_additional_epochs:
        return "resumable", training_info
    else:
        return "completed", training_info

def save_training_info(output_dir, config, early_stopped=False, resumed=False, original_epochs=None, tuned_hyperparams=None):
    """Save training configuration and state information."""
    training_info = {
        'config_hash': hash(str(config)),
        'early_stopped': early_stopped,
        'resumed': resumed,
        'original_epochs': original_epochs or config['training']['epochs'],
        'model_config': config['model'],
        'training_config': config['training'],
        'lora_config': config['lora'],
        'quantization_config': config['quantization'],
        'tuned_hyperparams': tuned_hyperparams
    }
    
    training_info_file = os.path.join(output_dir, "training_info.json")
    with open(training_info_file, 'w') as f:
        json.dump(training_info, f, indent=2)

def configs_match(current_config, saved_info):
    """Check if current config matches saved training config."""
    if not saved_info:
        return False
    
    current_hash = hash(str({
        'model': current_config['model'],
        'training': {k: v for k, v in current_config['training'].items() if k != 'epochs'},
        'lora': current_config['lora'],
        'quantization': current_config['quantization']
    }))
    
    saved_configs = {
        'model': saved_info.get('model_config', {}),
        'training': {k: v for k, v in saved_info.get('training_config', {}).items() if k != 'epochs'},
        'lora': saved_info.get('lora_config', {}),
        'quantization': saved_info.get('quantization_config', {})
    }
    saved_hash = hash(str(saved_configs))
    
    return current_hash == saved_hash

class EarlyStoppingCallbackCustom(EarlyStoppingCallback):
    """Custom early stopping with minimal improvement threshold."""
    
    def __init__(self, early_stopping_patience=2, early_stopping_threshold=0.0001):
        super().__init__(early_stopping_patience=early_stopping_patience, 
                        early_stopping_threshold=early_stopping_threshold)
        self.early_stopped = False
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        current_score = state.log_history[-1]["eval_loss"]
        if self.early_stopping_patience > 0:
            if self.is_metric_improved(current_score, state.best_metric):
                self.early_stopping_patience_counter = 0
            else:
                self.early_stopping_patience_counter += 1
            
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                control.should_training_stop = True
                self.early_stopped = True
                print(f"Early stopping triggered - validation loss improved by less than {self.early_stopping_threshold}")

def load_datasets(tokenized_path):
    """Load tokenized datasets with fallback loading."""
    try:
        tokenized_datasets = load_from_disk(tokenized_path)
        print(f"Datasets loaded: {list(tokenized_datasets.keys())}")
    except Exception as e:
        print(f"Direct load failed: {e}")
        dataset_dict = {}
        for split in ["train", "validation", "test"]:
            split_path = os.path.join(tokenized_path, split)
            if os.path.exists(split_path):
                dataset_dict[split] = Dataset.load_from_disk(split_path)
                print(f"Loaded {split}: {len(dataset_dict[split])} examples")
        tokenized_datasets = DatasetDict(dataset_dict)
    
    return tokenized_datasets

def apply_dataset_reduction(tokenized_datasets, config):
    """Apply VRAM optimization by reducing dataset size."""
    training_config = config['training']
    max_train_samples = training_config.get('max_train_samples', 8000)
    max_eval_samples = training_config.get('max_eval_samples', 800)
    
    if len(tokenized_datasets["train"]) > max_train_samples:
        print(f"Reducing training set from {len(tokenized_datasets['train'])} to {max_train_samples} samples")
        train_sampler = StratifiedSampler(tokenized_datasets["train"])
        tokenized_datasets["train"] = train_sampler.get_proportional_stratified_sample(max_train_samples)
    
    if len(tokenized_datasets["validation"]) > max_eval_samples:
        print(f"Reducing validation set from {len(tokenized_datasets['validation'])} to {max_eval_samples} samples")
        eval_sampler = StratifiedSampler(tokenized_datasets["validation"])
        tokenized_datasets["validation"] = eval_sampler.get_proportional_stratified_sample(max_eval_samples)
    
    return tokenized_datasets

def prepare_datasets_for_training(tokenized_datasets):
    """Prepare datasets with labels and PyTorch format."""
    def add_labels(example):
        example["labels"] = example["input_ids"].copy()
        return example
    
    tokenized_datasets = tokenized_datasets.map(add_labels)
    torch_columns = ["input_ids", "attention_mask", "labels"]
    tokenized_datasets.set_format("torch", columns=torch_columns)
    
    return tokenized_datasets

def load_model(config, base_model_path, output_dir, training_state):
    """Load model with quantization and LoRA configuration."""
    torch.cuda.empty_cache()
    
    quantization_enabled = config.get('quantization', {}).get('enabled', False)
    quantization_config = None

    if quantization_enabled:
        quant_config = config['quantization']
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'float16')),
            bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True)
        )
        print(f"Using 4-bit quantization: {quant_config.get('bnb_4bit_quant_type', 'nf4')}")

    use_deepspeed = config.get('deepspeed', {}).get('enabled', False)

    if training_state == "resumable":
        print("Loading model from checkpoint for resume...")
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=torch.float16,
            device_map="auto" if not use_deepspeed else None,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )
    else:
        print("Loading base model...")
        if os.path.exists(base_model_path):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto" if not use_deepspeed else None,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config['model']['base_model'],
                torch_dtype=torch.float16,
                device_map="auto" if not use_deepspeed else None,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )
        
        if quantization_enabled:
            model = prepare_model_for_kbit_training(model)
        
        model.gradient_checkpointing_enable()
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            lora_dropout=config['lora']['lora_dropout'],
            target_modules=config['lora']['target_modules']
        )
        
        model = get_peft_model(model, lora_config)
        print(f"Applied LoRA: r={config['lora']['r']}, alpha={config['lora']['lora_alpha']}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model

def setup_training_arguments(config, output_dir, training_state, tokenized_datasets):
    """Setup training arguments with DeepSpeed configuration."""
    batch_size = config['training']['batch_size']
    gradient_accumulation = config['training']['gradient_accumulation_steps']
    effective_batch_size = batch_size * gradient_accumulation
    
    num_epochs = config['training']['epochs']
    num_training_steps = (len(tokenized_datasets["train"]) // effective_batch_size) * num_epochs
    
    print(f"Training configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Training steps: {num_training_steps}")
    print(f"  Epochs: {num_epochs}")

    deepspeed_config = None
    use_deepspeed = config.get('deepspeed', {}).get('enabled', False)
    if use_deepspeed:
        deepspeed_config_path = os.path.join(cwd, config['deepspeed']['config_path'])
        if os.path.exists(deepspeed_config_path):
            deepspeed_config = deepspeed_config_path
            print(f"Using DeepSpeed config: {deepspeed_config}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=config['training']['warmup_steps'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_steps=config['training'].get('logging_steps', 50),
        evaluation_strategy="steps",
        eval_steps=config['training'].get('eval_steps', 250),
        save_strategy="steps",
        save_steps=config['training'].get('save_steps', 500),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        group_by_length=True,
        report_to=["tensorboard"],
        logging_dir=os.path.join(output_dir, "logs"),
        resume_from_checkpoint=output_dir if training_state == "resumable" else None,
        deepspeed=deepspeed_config,
        remove_unused_columns=False
    )
    
    return training_args

def run_production_training(config, base_model_path, tokenized_path, output_dir, training_state, training_info, optimal_hyperparams):
    """Run the main production training."""
    print("\n🚀 STARTING PRODUCTION TRAINING")
    print("=" * 60)
    
    if optimal_hyperparams:
        config['training']['learning_rate'] = optimal_hyperparams['learning_rate']
        config['lora']['r'] = optimal_hyperparams['lora']['r']
        config['lora']['lora_alpha'] = optimal_hyperparams['lora']['alpha']
        config['training']['batch_size'] = optimal_hyperparams['batch_size']
        config['lora']['lora_dropout'] = optimal_hyperparams['lora_dropout']
        config['training']['weight_decay'] = optimal_hyperparams['weight_decay']
        config['training']['warmup_steps'] = optimal_hyperparams['warmup_steps']
        print("✅ Applied tuned hyperparameters to production training config")
    
    print("Loading tokenizer for production training...")
    if os.path.exists(base_model_path):
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading tokenized datasets from: {tokenized_path}")
    tokenized_datasets = load_datasets(tokenized_path)
    
    verify_dataset_stratification(tokenized_datasets)
    
    tokenized_datasets = apply_dataset_reduction(tokenized_datasets, config)
    tokenized_datasets = prepare_datasets_for_training(tokenized_datasets)

    print("Loading model...")
    model = load_model(config, base_model_path, output_dir, training_state)
    
    training_args = setup_training_arguments(config, output_dir, training_state, tokenized_datasets)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    callbacks = []
    early_stopping = EarlyStoppingCallbackCustom(
        early_stopping_patience=2,
        early_stopping_threshold=0.0001
    )
    callbacks.append(early_stopping)
    callbacks.append(TensorBoardCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    print("\n🚀 Starting training...")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    save_training_info(
        output_dir=output_dir,
        config=config,
        early_stopped=early_stopping.early_stopped,
        resumed=(training_state == "resumable"),
        original_epochs=config['training']['epochs'],
        tuned_hyperparams=optimal_hyperparams
    )

    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    print("\n🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Final validation loss: {eval_results['eval_loss']:.4f}")
    print(f"Training steps completed: {train_result.global_step}")
    print(f"Model saved to: {output_dir}")
    
    return trainer, eval_results

def main():
    """Main training function."""
    print("🚀 MIXTRAL STORYTELLING MODEL TRAINING")
    print("=" * 60)
    
    config_path = os.path.join(cwd, "configs", "model_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from: {config_path}")
    
    base_model_path = os.path.join(config['paths']['models'], 'mixtral-8x7b-base')
    tokenized_path = os.path.join(config['paths']['data_tokenized'], 'datasets')
    output_dir = os.path.join(config['paths']['models'], 'tuned_story_llm')
    
    if not os.path.exists(tokenized_path):
        print(f"❌ Tokenized datasets not found at: {tokenized_path}")
        print("Please run data_tokenizer.py first.")
        return
    
    training_state, training_info = check_training_state(output_dir)
    
    print(f"\nTraining state: {training_state}")
    
    if training_state == "completed":
        print("✅ Training already completed.")
        response = input("Restart training from scratch? (y/N): ").strip().lower()
        if response != 'y':
            print("Training cancelled.")
            return
        else:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            training_state = "new"
            training_info = None
    
    elif training_state == "resumable":
        if configs_match(config, training_info):
            print("📄 Configuration matches previous training.")
            response = input("Resume training? (Y/n): ").strip().lower()
            if response == 'n':
                print("Training cancelled.")
                return
        else:
            print("⚠️ Configuration differs from previous training.")
            print("Starting fresh training...")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            training_state = "new"
            training_info = None
    
    elif training_state == "early_stopped":
        print("⏹️ Previous training stopped early.")
        response = input("Continue training with additional epochs? (Y/n): ").strip().lower()
        if response == 'n':
            print("Training cancelled.")
            return
        else:
            config['training']['epochs'] = training_info.get('original_epochs', 3) + 2
            training_state = "resumable"
    
    optimal_hyperparams = None
    tuning_enabled = config.get('hyperparameter_tuning', {}).get('enabled', False)
    
    if tuning_enabled and training_state == "new":
        print("\n🔬 Hyperparameter tuning enabled")
        response = input("Run hyperparameter tuning? (Y/n): ").strip().lower()
        if response != 'n':
            optimal_hyperparams = run_hyperparameter_tuning(config, base_model_path, tokenized_path)
        else:
            print("Skipping hyperparameter tuning, using config defaults")
    elif training_state == "resumable" and training_info:
        optimal_hyperparams = training_info.get('tuned_hyperparams')
        if optimal_hyperparams:
            print("✅ Using previously tuned hyperparameters")
    
    try:
        trainer, eval_results = run_production_training(
            config=config,
            base_model_path=base_model_path,
            tokenized_path=tokenized_path,
            output_dir=output_dir,
            training_state=training_state,
            training_info=training_info,
            optimal_hyperparams=optimal_hyperparams
        )
        
        print("\n📊 TRAINING STATISTICS:")
        print(f"  Final Loss: {eval_results['eval_loss']:.4f}")
        model_file = os.path.join(output_dir, 'pytorch_model.bin')
        if os.path.exists(model_file):
            print(f"  Model Size: {os.path.getsize(model_file) / (1024**3):.2f} GB")
        print(f"  Output Directory: {output_dir}")
        
        if optimal_hyperparams:
            hyperparams_file = os.path.join(output_dir, "optimal_hyperparameters.json")
            with open(hyperparams_file, 'w') as f:
                json.dump(optimal_hyperparams, f, indent=2)
            print(f"  Hyperparameters: {hyperparams_file}")
        
        print("\n✅ Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()