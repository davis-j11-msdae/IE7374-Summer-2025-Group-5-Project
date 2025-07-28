import os
import sys
import json
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch

import warnings

from utils.helpers import set_cwd, load_datasets

# Suppress TensorFlow warnings - these come from detoxify using TensorFlow backend
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for consistent results
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow INFO/WARNING messages
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizer warnings

# Filter specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tf_keras")
warnings.filterwarnings("ignore", message=".*tf.losses.sparse_softmax_cross_entropy.*")
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*use_reentrant.*")

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import set_cwd

# Get current working directory for path operations
cwd = set_cwd()

def check_training_state(output_dir):
    """Check if previous training exists and its completion state."""
    if not os.path.exists(output_dir):
        return "new", None

    # Check for checkpoint directories (resumable training)
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(output_dir, d))]
    
    # Check for final model files (completed training)
    final_model_files = ["config.json", "pytorch_model.bin", "adapter_config.json", "adapter_model.bin"]
    final_model_exists = any(os.path.exists(os.path.join(output_dir, f)) for f in final_model_files)
    
    print(f"Checking {output_dir}:")
    print(f"  Checkpoint dirs found: {len(checkpoint_dirs)} {checkpoint_dirs}")
    print(f"  Final model files exist: {final_model_exists}")
    
    # If no checkpoints and no final model, directory is empty
    if not checkpoint_dirs and not final_model_exists:
        print("  No training artifacts found")
        return "new", None

    trainer_state_file = os.path.join(output_dir, "trainer_state.json")
    training_info_file = os.path.join(output_dir, "training_info.json")

    # Load training info if available
    training_info = {}
    if os.path.exists(training_info_file):
        try:
            with open(training_info_file, 'r') as f:
                training_info = json.load(f)
        except Exception as e:
            print(f"  Error reading training_info.json: {e}")

    # Check trainer state
    if os.path.exists(trainer_state_file):
        try:
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
            
            completed_epochs = trainer_state.get('epoch', 0)
            training_info['completed_epochs'] = completed_epochs
            
            # Check for early stopping
            early_stopped = training_info.get('early_stopped', False)
            original_epochs = training_info.get('original_epochs', 5)  # Default to 5
            max_additional_epochs = 3

            print(f"  Trainer state: {completed_epochs} epochs completed")
            
            if early_stopped:
                print("  Status: Training was early stopped")
                return "early_stopped", training_info
            elif final_model_exists and completed_epochs >= original_epochs:
                # Check if we can do additional epochs
                if completed_epochs < original_epochs + max_additional_epochs:
                    print("  Status: Training completed but can resume for additional epochs")
                    return "resumable", training_info
                else:
                    print("  Status: Training fully completed")
                    return "completed", training_info
            else:
                print("  Status: Training incomplete, can be resumed")
                return "resumable", training_info
                
        except Exception as e:
            print(f"  Error reading trainer_state.json: {e}")
    
    # If we have checkpoints but no trainer state, we can still resume
    if checkpoint_dirs:
        print("  Status: Checkpoints found but no trainer state - can be resumed")
        # Try to determine latest checkpoint
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
        training_info['latest_checkpoint'] = latest_checkpoint
        return "resumable", training_info
    
    # If we have final model files but no trainer state, consider it completed
    if final_model_exists:
        print("  Status: Final model found but no trainer state - assuming completed")
        return "completed", training_info
    
    # Fallback
    return "new", None


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


def get_hardware_batch_config(config):
    """Get hardware-appropriate batch configuration based on effective batch size."""

    effective_batch_size = config['training']['effective_batch_size']
    limited_gpu_batch_size = config['training']['limited_gpu_batch_size']

    if torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0 >= 40:  # A100
        return {
            'batch_size': effective_batch_size,
            'gradient_accumulation_steps': 1
        }
    else:  # 10GB GPU or smaller
        return {
            'batch_size': limited_gpu_batch_size,
            'gradient_accumulation_steps': effective_batch_size // limited_gpu_batch_size
        }

def _set_nested_config_value(config, path, value):
    """Set a nested configuration value using a path list."""
    current = config
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


class EarlyStoppingCallbackCustom(EarlyStoppingCallback):
    """Custom early stopping with minimal improvement threshold."""

    def __init__(self, early_stopping_patience=2, early_stopping_threshold=0.0001):
        super().__init__(early_stopping_patience=early_stopping_patience,
                         early_stopping_threshold=early_stopping_threshold)
        self.early_stopped = False
        # Initialize missing attributes that parent class expects
        self.metric_for_best_model = "eval_loss"
        self.greater_is_better = False

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

    def is_metric_improved(self, metric, reference_metric):
        """
        Check if the metric has improved based on the threshold and direction.
        """
        if reference_metric is None:
            return True

        if self.greater_is_better:
            return metric > reference_metric + self.early_stopping_threshold
        else:
            return metric < reference_metric - self.early_stopping_threshold

def setup_quantization_config(config):
    """Setup quantization configuration with Windows compatibility."""
    quantization_enabled = config.get('quantization', {}).get('enabled', False)

    if not quantization_enabled:
        return None

    try:
        # Try to import bitsandbytes
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig

        # Test if bitsandbytes is working properly - handle different versions
        bnb_available = False
        if hasattr(bnb, 'is_available'):
            bnb_available = bnb.is_available()
        else:
            # For older versions without is_available, try a simple test
            try:
                # Test if we can create a simple optimizer (basic functionality test)
                import torch
                test_param = torch.nn.Parameter(torch.randn(2, 2))
                bnb.optim.Adam8bit([test_param], lr=0.001)
                bnb_available = True
            except:
                bnb_available = False

        if not bnb_available:
            print("BitsAndBytes not available, disabling quantization")
            return None

        quant_config = config['quantization']


        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quant_config['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype']),
            bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant']
        )
        print(f"Using 4-bit quantization configuration")

        return quantization_config

    except ImportError:
        print("BitsAndBytes not installed, disabling quantization")
        return None
    except Exception as e:
        print(f"BitsAndBytes setup failed: {e}")
        print("Continuing without quantization")
        return None


def get_device_config():
    """Determine device configuration based on available GPU memory."""
    if not torch.cuda.is_available():
        return "cpu", {}

    if torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0 >= 40:  # A100 or similar
        return "auto", {}
    else:  # 10GB GPU - use sequential device mapping instead of CPU offload
        device_map = {
            "model.embed_tokens": 0,
            "model.layers": 0,  # Keep all layers on GPU for training stability
            "model.norm": 0,
            "lm_head": 0
        }
        max_memory = {0: "10GB"}  # Use most of available VRAM
        return device_map, max_memory

def load_model(config, base_model_path, output_dir, training_state):
    """Load Mistral 7B model with optional quantization and LoRA configuration."""
    torch.cuda.empty_cache()

    print(f"Detected GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0:.1f}GB")
    # Setup quantization config with Windows compatibility
    quantization_config = setup_quantization_config(config)
    device_config = get_device_config()
    
    # Handle device configuration based on return values
    if len(device_config) == 2:
        device_map, max_memory = device_config
    else:
        device_map = device_config[0]
        max_memory = {}

    model_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': device_map,
        'low_cpu_mem_usage': True,
        'trust_remote_code': True
    }

    if max_memory:
        model_kwargs['max_memory'] = max_memory

    # Determine device map based on quantization and VRAM constraints
    if quantization_config is not None:
        model_kwargs['quantization_config'] = quantization_config

    if training_state == "resumable":
        print("Loading model from checkpoint for resume...")
        model = AutoModelForCausalLM.from_pretrained(
            output_dir, **model_kwargs
        )
    else:
        print("Loading base Mistral 7B model...")
        if os.path.exists(base_model_path):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config['model']['base_model'],
                torch_dtype=torch.float16,
                device_map=device_map,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

        # Only apply LoRA preparation if quantization is enabled
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # Extract LoRA rank value properly
        lora_rank = config['lora']['lora_rank']
        if isinstance(lora_rank, dict):
            lora_rank = lora_rank.get('lora_rank', lora_rank.get('r', 16))
        
        # Extract LoRA alpha value properly  
        lora_alpha = config['lora']['lora_alpha']
        if isinstance(lora_alpha, dict):
            lora_alpha = lora_alpha.get('lora_alpha', lora_alpha.get('alpha', 32))

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=config['lora']['lora_dropout'],
            target_modules=config['lora']['target_modules']
        )

        model = get_peft_model(model, lora_config)
        print(f"Applied LoRA: rank={lora_rank}, alpha={lora_alpha}")

    # Move model to GPU if it's not already there (fixes meta tensor issue)
    if not quantization_config and hasattr(model, 'to'):
        try:
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            print(f"Warning: Could not move model to GPU: {e}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

    return model


def setup_training_arguments_generic(config, output_dir):
    """Setup training arguments for both tuning and production."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_steps=999999,  # Disable logging for all training
        eval_strategy='epoch',  # Always use epoch-based evaluation
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,  # Disable pin memory to save VRAM
        group_by_length=False,
        report_to="none",  # No reporting for all training
        remove_unused_columns=False,
        disable_tqdm=False,
        log_level="error",  # Error level logging for all
        # Memory optimization settings
        gradient_checkpointing=False,
        optim="adamw_torch",  # Use PyTorch optimizer instead of transformers default
        lr_scheduler_type="constant",
        warmup_ratio=config['training']['warmup_ratio'],
        # Additional memory optimizations for hanging issue
        dataloader_persistent_workers=False,  # Disable persistent workers
        include_inputs_for_metrics=False,  # Reduce memory during evaluation
    )

    return training_args

def run_training_core(config, hyperparams, base_model_path, tokenized_datasets, output_dir, training_state, save_strategy, save_total_limit, load_best_model_at_end):
    """Core training function called by hyperparameter tuning and production training."""

    print(f"Training samples: {len(tokenized_datasets['train'])}")
    print(f"Validation samples: {len(tokenized_datasets['validation'])}")

    # Load model with quantization
    model = load_model(config, base_model_path, output_dir, "new")

    # Load tokenizer
    if os.path.exists(base_model_path):
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'], use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set epochs based on training state
    if training_state == 'tuning':
        config['training']['epochs'] = 1
    elif training_state == "early_stopped":
        config['training']['epochs'] = 3
    elif training_state == "resumable":
        config['training']['epochs'] = 3
    else:
        config['training']['epochs'] = 5

    # Set training arguments
    config['training']['save_total_limit'] = save_total_limit
    config['training']['save_strategy'] = save_strategy
    config['training']['load_best_model_at_end'] = load_best_model_at_end
    config['training']['disable_tqdm'] = False
    config['training']['group_by_length'] = True

    # Ensure batch size configuration exists - get hardware-appropriate defaults first
    if 'batch_size' not in config['training'] or 'gradient_accumulation_steps' not in config['training']:
        batch_config = get_hardware_batch_config(config)
        config['training']['batch_size'] = batch_config['batch_size']
        config['training']['gradient_accumulation_steps'] = batch_config['gradient_accumulation_steps']

    # Apply hyperparameters
    for param_name, param_value in hyperparams.items():
        if param_name == 'lora_rank':
            # Handle both dict and direct value formats
            if isinstance(param_value, dict):
                rank_value = param_value.get('lora_rank', param_value.get('r', 16))
            else:
                rank_value = param_value
            config['lora']['lora_rank'] = rank_value
            config['lora']['lora_alpha'] = rank_value * 2
        elif 'lora' in param_name:
            config['lora'][param_name] = param_value
        elif param_name == 'effective_batch_size':
            # Handle effective batch size by recalculating batch config
            effective_batch_size = param_value
            batch_config = get_hardware_batch_config(config)
            # Override the effective batch size calculation
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0
            if gpu_memory_gb >= 40:  # A100
                config['training']['batch_size'] = effective_batch_size
                config['training']['gradient_accumulation_steps'] = 1
            else:  # 10GB GPU
                limited_gpu_batch_size = config['training']['limited_gpu_batch_size']
                config['training']['batch_size'] = limited_gpu_batch_size
                config['training']['gradient_accumulation_steps'] = effective_batch_size // limited_gpu_batch_size
        else:
            config['training'][param_name] = param_value

    print(f"Using batch_size: {config['training']['batch_size']}")
    print(f"Using gradient_accumulation_steps: {config['training']['gradient_accumulation_steps']}")

    training_args = setup_training_arguments_generic(
        config, output_dir
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None
    )
    callbacks = []
    if training_state != 'tuning':
        # Add early stopping for production training
        early_stopping = EarlyStoppingCallbackCustom(
            early_stopping_patience=2, 
            early_stopping_threshold=0.0001
        )
        callbacks.append(early_stopping)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=callbacks
    )

    # Run training with appropriate messaging
    if training_state == 'tuning':
        print("Starting hyperparameter tuning training...")
    else:
        print("Starting production training...")
    
    trainer.train()

    # Evaluate and return validation loss
    eval_results = trainer.evaluate()
    validation_loss = eval_results['eval_loss']

    if training_state == 'tuning':
        print(f"Hyperparameter tuning completed. Validation Loss: {validation_loss:.4f}")
    else:
        print(f"Production training completed. Validation Loss: {validation_loss:.4f}")

    # Save model for production training
    if training_state != 'tuning':
        print("Saving production model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")

    # Clean up model from memory
    del model
    del trainer
    torch.cuda.empty_cache()

    return validation_loss

def production_training(config, optimal_hyperparams, base_model_path,tokenized_path, output_dir,training_state):
    tokenized_datasets=load_datasets(tokenized_path,'production')
    run_training_core(config, optimal_hyperparams, base_model_path, tokenized_datasets, output_dir,
                            training_state, 'epoch', '5', True)