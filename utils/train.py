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

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import set_cwd

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


def save_training_info(output_dir, config, early_stopped=False, resumed=False, original_epochs=None,
                       tuned_hyperparams=None):
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


def apply_hyperparameters_to_config(config, optimal_hyperparams):
    """Apply optimal hyperparameters to the config."""
    if not optimal_hyperparams:
        return config

    print("Applying hyperparameters to training config...")

    if 'learning_rate' in optimal_hyperparams:
        config['training']['learning_rate'] = optimal_hyperparams['learning_rate']
        print(f"  Learning Rate: {optimal_hyperparams['learning_rate']}")

    if 'lora' in optimal_hyperparams:
        config['lora']['r'] = optimal_hyperparams['lora']['r']
        config['lora']['lora_alpha'] = optimal_hyperparams['lora']['alpha']
        print(f"  LoRA Rank: {optimal_hyperparams['lora']['r']}")
        print(f"  LoRA Alpha: {optimal_hyperparams['lora']['alpha']}")

    if 'batch_size' in optimal_hyperparams:
        config['training']['batch_size'] = optimal_hyperparams['batch_size']
        print(f"  Batch Size: {optimal_hyperparams['batch_size']}")

    if 'lora_dropout' in optimal_hyperparams:
        config['lora']['lora_dropout'] = optimal_hyperparams['lora_dropout']
        print(f"  LoRA Dropout: {optimal_hyperparams['lora_dropout']}")

    if 'weight_decay' in optimal_hyperparams:
        config['training']['weight_decay'] = optimal_hyperparams['weight_decay']
        print(f"  Weight Decay: {optimal_hyperparams['weight_decay']}")

    if 'warmup_steps' in optimal_hyperparams:
        config['training']['warmup_steps'] = optimal_hyperparams['warmup_steps']
        print(f"  Warmup Steps: {optimal_hyperparams['warmup_steps']}")

    print("Hyperparameters applied successfully")
    return config


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
                print(
                    f"Early stopping triggered - validation loss improved by less than {self.early_stopping_threshold}")


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
    max_train_samples = training_config.get('max_train_samples', 12000)
    max_eval_samples = training_config.get('max_eval_samples', 1200)

    if len(tokenized_datasets["train"]) > max_train_samples:
        print(f"Reducing training set from {len(tokenized_datasets['train'])} to {max_train_samples} samples")
        tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42).select(range(max_train_samples))

    if len(tokenized_datasets["validation"]) > max_eval_samples:
        print(f"Reducing validation set from {len(tokenized_datasets['validation'])} to {max_eval_samples} samples")
        tokenized_datasets["validation"] = tokenized_datasets["validation"].shuffle(seed=42).select(
            range(max_eval_samples))

    return tokenized_datasets


def filter_preselected_samples(tokenized_datasets):
    """Filter datasets to use only pre-selected hyperparameter tuning samples."""
    print("Filtering to use pre-selected tuning samples...")

    # Filter train dataset
    train_indices = [i for i, ex in enumerate(tokenized_datasets['train']) if ex.get('hyperparameter_tuning', False)]
    if train_indices:
        tokenized_datasets['train'] = tokenized_datasets['train'].select(train_indices)
        print(f"  Train samples: {len(train_indices):,}")

    # Filter validation dataset
    val_indices = [i for i, ex in enumerate(tokenized_datasets['validation']) if ex.get('hyperparameter_tuning', False)]
    if val_indices:
        tokenized_datasets['validation'] = tokenized_datasets['validation'].select(val_indices)
        print(f"  Validation samples: {len(val_indices):,}")

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


def configure_for_tuning(config):
    """Configure training parameters for hyperparameter tuning."""
    config['training']['epochs'] = 1
    config['training']['eval_steps'] = 50
    config['training']['save_steps'] = 1000
    config['training']['logging_steps'] = 999999  # Disable logging
    config['training']['max_steps'] = 10  # Very limited for quick evaluation
    return config


def load_model(config, base_model_path, output_dir, training_state):
    """Load Mistral 7B model with quantization and LoRA configuration."""
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

    if training_state == "resumable":
        print("Loading model from checkpoint for resume...")
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )
    else:
        print("Loading base Mistral 7B model...")
        if os.path.exists(base_model_path):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config['model']['base_model'],
                torch_dtype=torch.float16,
                device_map="auto",
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
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

    return model


def setup_training_arguments_generic(config, output_dir, tokenized_datasets, tuning_ind):
    """Setup training arguments for both tuning and production."""
    batch_size = config['training']['batch_size']
    if tuning_ind == 1:
        batch_size = min(batch_size, 2)  # Reduce batch size for tuning

    gradient_accumulation = config['training']['gradient_accumulation_steps']
    if tuning_ind == 1:
        gradient_accumulation = max(2, 4 // batch_size)  # Increase accumulation for tuning

    num_epochs = config['training']['epochs']

    print(f"    Training configuration:")
    print(f"      Batch size: {batch_size}")
    print(f"      Gradient accumulation: {gradient_accumulation}")
    print(f"      Epochs: {num_epochs}")
    print(f"      Max steps: {config['training'].get('max_steps', 'unlimited')}")

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
        evaluation_strategy="epoch" if tuning_ind == 1 else "steps",
        eval_steps=config['training'].get('eval_steps', 200),
        save_strategy="no" if tuning_ind == 1 else "steps",
        save_steps=config['training'].get('save_steps', 400),
        save_total_limit=3 if tuning_ind == 0 else 0,
        load_best_model_at_end=tuning_ind == 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_num_workers=2 if tuning_ind == 0 else 0,
        dataloader_pin_memory=tuning_ind == 0,
        group_by_length=tuning_ind == 0,
        report_to="none" if tuning_ind == 1 else ["tensorboard"],
        logging_dir=os.path.join(output_dir, "logs") if tuning_ind == 0 else None,
        remove_unused_columns=False,
        disable_tqdm=tuning_ind == 1,
        log_level="error" if tuning_ind == 1 else "info",
        max_steps=config['training'].get('max_steps', -1),
        # Memory optimization settings
        gradient_checkpointing=True,
        optim="adamw_torch",  # Use PyTorch optimizer instead of transformers default
        lr_scheduler_type="cosine",
        warmup_ratio=0.1
    )

    return training_args


def run_training_generic(config, hyperparams, base_model_path, tokenized_path, output_dir, tuning_ind=0):
    """Generic training function that handles both tuning and production training."""

    # Apply hyperparameters to config
    config = apply_hyperparameters_to_config(config, hyperparams)

    # Configure for tuning if indicated
    if tuning_ind == 1:
        config = configure_for_tuning(config)
        print("    Configured for hyperparameter tuning")

    # Load tokenizer
    if os.path.exists(base_model_path):
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    tokenized_datasets = load_datasets(tokenized_path)

    # Filter datasets based on tuning indicator
    if tuning_ind == 1:
        tokenized_datasets = filter_preselected_samples(tokenized_datasets)
    else:
        tokenized_datasets = apply_dataset_reduction(tokenized_datasets, config)

    tokenized_datasets = prepare_datasets_for_training(tokenized_datasets)

    # Load model
    model = load_model(config, base_model_path, output_dir, "new")

    # Setup training arguments
    training_args = setup_training_arguments_generic(config, output_dir, tokenized_datasets, tuning_ind)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    callbacks = []
    if tuning_ind == 0:  # Only use callbacks for production training
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

    print("    Starting training...")
    trainer.train()

    print("    Evaluating...")
    eval_results = trainer.evaluate()
    validation_loss = eval_results['eval_loss']

    if tuning_ind == 0:
        # Production training - save model and info
        save_training_info(
            output_dir=output_dir,
            config=config,
            early_stopped=getattr(callbacks[0], 'early_stopped', False) if callbacks else False,
            resumed=False,
            original_epochs=config['training']['epochs'],
            tuned_hyperparams=hyperparams
        )

        print("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        print("TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Final validation loss: {validation_loss:.4f}")
        print(f"Model saved to: {output_dir}")

    # Cleanup
    del trainer, model
    torch.cuda.empty_cache()

    return validation_loss


def run_training(config, base_model_path, tokenized_path, output_dir, training_state, training_info,
                 optimal_hyperparams):
    """Run production training using the generic training function."""
    print("\nSTARTING MISTRAL 7B TRAINING")
    print("=" * 60)

    return run_training_generic(
        config=config,
        hyperparams=optimal_hyperparams,
        base_model_path=base_model_path,
        tokenized_path=tokenized_path,
        output_dir=output_dir,
        tuning_ind=0  # Production training
    )


def main(optimal_hyperparams=None):
    """Main training function that accepts optional hyperparameters."""
    print("MISTRAL 7B STORYTELLING MODEL TRAINING")
    print("=" * 60)

    config_path = os.path.join(cwd, "configs", "model_config.yaml")

    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded configuration from: {config_path}")

    # Show hyperparameter source
    if optimal_hyperparams:
        print("Using supplied optimal hyperparameters")
    else:
        print("Using default hyperparameters from config file")

    base_model_path = os.path.join(config['paths']['models'], 'mistral-7b-base')
    tokenized_path = os.path.join(config['paths']['data_tokenized'], 'datasets')
    output_dir = os.path.join(config['paths']['models'], 'tuned_story_llm')

    if not os.path.exists(tokenized_path):
        print(f"Tokenized datasets not found at: {tokenized_path}")
        print("Please run data_tokenizer.py first.")
        return

    training_state, training_info = check_training_state(output_dir)

    print(f"\nTraining state: {training_state}")

    if training_state == "completed":
        print("Training already completed.")
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
            print("Configuration matches previous training.")
            response = input("Resume training? (Y/n): ").strip().lower()
            if response == 'n':
                print("Training cancelled.")
                return
        else:
            print("Configuration differs from previous training.")
            print("Starting fresh training...")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            training_state = "new"
            training_info = None

    elif training_state == "early_stopped":
        print("Previous training stopped early.")
        response = input("Continue training with additional epochs? (Y/n): ").strip().lower()
        if response == 'n':
            print("Training cancelled.")
            return
        else:
            config['training']['epochs'] = training_info.get('original_epochs', 3) + 2
            training_state = "resumable"

    try:
        validation_loss = run_training(
            config=config,
            base_model_path=base_model_path,
            tokenized_path=tokenized_path,
            output_dir=output_dir,
            training_state=training_state,
            training_info=training_info,
            optimal_hyperparams=optimal_hyperparams
        )

        print("\nTRAINING STATISTICS:")
        print(f"  Final Loss: {validation_loss:.4f}")
        model_file = os.path.join(output_dir, 'pytorch_model.bin')
        if os.path.exists(model_file):
            print(f"  Model Size: {os.path.getsize(model_file) / (1024 ** 3):.2f} GB")
        print(f"  Output Directory: {output_dir}")

        if optimal_hyperparams:
            hyperparams_file = os.path.join(output_dir, "optimal_hyperparameters.json")
            with open(hyperparams_file, 'w') as f:
                json.dump(optimal_hyperparams, f, indent=2)
            print(f"  Hyperparameters: {hyperparams_file}")

        print("\nMistral 7B training pipeline completed successfully!")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()