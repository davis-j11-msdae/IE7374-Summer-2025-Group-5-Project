import os
import sys
import copy
import shutil
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import numpy as np
from collections import defaultdict

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import set_cwd

# Get current working directory for path operations
cwd = set_cwd()


class StratifiedSampler:
    """Handles stratified sampling across source and age group."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.strata_info = self._analyze_strata()

    def _analyze_strata(self):
        """Analyze dataset stratification across source and age_group."""
        strata_counts = defaultdict(int)

        for item in self.dataset:
            source = item.get('source', 'unknown')
            age_group = item.get('age_group', 'unknown')
            strata_counts[(source, age_group)] += 1

        print("\nDataset Stratification Analysis:")
        print("=" * 60)
        total_samples = len(self.dataset)

        for (source, age_group), count in sorted(strata_counts.items()):
            percentage = (count / total_samples) * 100
            print(f"{source:25} | {age_group:8} | {count:6} ({percentage:5.1f}%)")

        return dict(strata_counts)

    def get_proportional_stratified_sample(self, target_size):
        """Get proportionally stratified sample maintaining original distribution."""
        total_samples = len(self.dataset)
        stratified_indices = []

        print(f"\nCreating stratified sample ({target_size} from {total_samples}):")

        for (source, age_group), stratum_count in self.strata_info.items():
            proportion = stratum_count / total_samples
            stratum_sample_size = max(1, int(target_size * proportion))

            stratum_indices = [
                i for i, item in enumerate(self.dataset)
                if item.get('source') == source and item.get('age_group') == age_group
            ]

            if len(stratum_indices) >= stratum_sample_size:
                np.random.seed(42)
                sampled_indices = np.random.choice(
                    stratum_indices,
                    size=stratum_sample_size,
                    replace=False
                ).tolist()
                stratified_indices.extend(sampled_indices)
                print(f"  {source:25} | {age_group:8} | {stratum_sample_size:3} samples")
            else:
                stratified_indices.extend(stratum_indices)
                print(f"  {source:25} | {age_group:8} | {len(stratum_indices):3} samples (all)")

        return self.dataset.select(stratified_indices)


class SequentialHyperparameterTuner:
    """Sequential hyperparameter optimization with configurable search spaces."""

    def __init__(self, base_config, tokenizer, base_model_path, tuning_sample, eval_sample):
        self.base_config = copy.deepcopy(base_config)
        self.tokenizer = tokenizer
        self.base_model_path = base_model_path
        self.tuning_sample = tuning_sample
        self.eval_sample = eval_sample
        self.optimization_history = {}

        tuning_config = base_config.get('hyperparameter_tuning', {})
        self.search_spaces = tuning_config.get('search_spaces', {})

        required_spaces = ['learning_rate', 'lora_rank_alpha', 'batch_size', 'lora_dropout', 'weight_decay',
                           'warmup_steps']
        for space in required_spaces:
            if space not in self.search_spaces:
                raise ValueError(f"Missing search space for {space} in hyperparameter_tuning config")

        print(f"Loaded hyperparameter search spaces from config:")
        for space_name, space_values in self.search_spaces.items():
            print(f"  {space_name}: {space_values}")

    def tune_sequential(self):
        """Run sequential hyperparameter optimization."""
        print("\n🔬 STARTING SEQUENTIAL HYPERPARAMETER TUNING")
        print("=" * 60)

        print("\n📈 Phase 1: Tuning Learning Rate...")
        best_lr = self._tune_learning_rate()
        self.optimization_history['learning_rate'] = best_lr
        print(f"✅ Best Learning Rate: {best_lr}")

        print("\n🎯 Phase 2: Tuning LoRA Rank and Alpha...")
        best_lora = self._tune_lora_params(best_lr)
        self.optimization_history['lora'] = best_lora
        print(f"✅ Best LoRA Config: r={best_lora['r']}, alpha={best_lora['alpha']}")

        print("\n📦 Phase 3: Tuning Batch Size...")
        best_batch = self._tune_batch_size(best_lr, best_lora)
        self.optimization_history['batch_size'] = best_batch
        print(f"✅ Best Batch Size: {best_batch}")

        print("\n🎲 Phase 4: Tuning LoRA Dropout...")
        best_dropout = self._tune_lora_dropout(best_lr, best_lora, best_batch)
        self.optimization_history['lora_dropout'] = best_dropout
        print(f"✅ Best LoRA Dropout: {best_dropout}")

        print("\n⚖️ Phase 5: Tuning Weight Decay...")
        best_weight_decay = self._tune_weight_decay(best_lr, best_lora, best_batch, best_dropout)
        self.optimization_history['weight_decay'] = best_weight_decay
        print(f"✅ Best Weight Decay: {best_weight_decay}")

        print("\n🌡️ Phase 6: Tuning Warmup Steps...")
        best_warmup = self._tune_warmup_steps(best_lr, best_lora, best_batch, best_dropout, best_weight_decay)
        self.optimization_history['warmup_steps'] = best_warmup
        print(f"✅ Best Warmup Steps: {best_warmup}")

        print("\n🎉 HYPERPARAMETER TUNING COMPLETED!")
        print("=" * 60)
        self._print_optimization_summary()

        return self.optimization_history

    def _create_config_for_trial(self, **overrides):
        """Create config for a specific trial with parameter overrides."""
        trial_config = copy.deepcopy(self.base_config)

        for key, value in overrides.items():
            if key == 'learning_rate':
                trial_config['training']['learning_rate'] = value
            elif key == 'lora_r':
                trial_config['lora']['r'] = value
            elif key == 'lora_alpha':
                trial_config['lora']['lora_alpha'] = value
            elif key == 'batch_size':
                trial_config['training']['batch_size'] = value
            elif key == 'lora_dropout':
                trial_config['lora']['lora_dropout'] = value
            elif key == 'weight_decay':
                trial_config['training']['weight_decay'] = value
            elif key == 'warmup_steps':
                trial_config['training']['warmup_steps'] = value

        return trial_config

    def _evaluate_config(self, trial_config, trial_name="trial"):
        """Evaluate a configuration and return validation loss."""
        print(f"  🧪 Testing {trial_name}...")

        try:
            torch.cuda.empty_cache()

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path if os.path.exists(self.base_model_path) else trial_config['model']['base_model'],
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )

            model = prepare_model_for_kbit_training(model)
            model.gradient_checkpointing_enable()

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=trial_config['lora']['r'],
                lora_alpha=trial_config['lora']['lora_alpha'],
                lora_dropout=trial_config['lora']['lora_dropout'],
                target_modules=trial_config['lora']['target_modules']
            )

            model = get_peft_model(model, lora_config)

            training_args = TrainingArguments(
                output_dir="./temp_tuning",
                num_train_epochs=1,
                per_device_train_batch_size=trial_config['training']['batch_size'],
                per_device_eval_batch_size=trial_config['training']['batch_size'],
                gradient_accumulation_steps=max(1, 12 // trial_config['training']['batch_size']),
                warmup_steps=trial_config['training']['warmup_steps'],
                learning_rate=float(trial_config['training']['learning_rate']),
                weight_decay=trial_config['training']['weight_decay'],
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=20,
                save_strategy="no",
                fp16=True,
                dataloader_num_workers=0,
                dataloader_pin_memory=True,
                report_to="none",
                disable_tqdm=True,
                log_level="error"
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=self.tuning_sample,
                eval_dataset=self.eval_sample,
                tokenizer=self.tokenizer,
            )

            trainer.train()
            eval_results = trainer.evaluate()
            validation_loss = eval_results['eval_loss']

            del model, trainer
            torch.cuda.empty_cache()

            print(f"    📊 Validation Loss: {validation_loss:.4f}")
            return validation_loss

        except Exception as e:
            print(f"    ❌ Trial failed: {e}")
            torch.cuda.empty_cache()
            return float('inf')

    def _tune_learning_rate(self):
        """Tune learning rate."""
        best_lr, best_score = None, float('inf')

        for lr in self.search_spaces['learning_rate']:
            trial_config = self._create_config_for_trial(learning_rate=lr)
            score = self._evaluate_config(trial_config, f"LR={lr}")

            if score < best_score:
                best_lr, best_score = lr, score

        return best_lr

    def _tune_lora_params(self, best_lr):
        """Tune LoRA rank and alpha together."""
        best_lora, best_score = None, float('inf')

        for lora_config in self.search_spaces['lora_rank_alpha']:
            trial_config = self._create_config_for_trial(
                learning_rate=best_lr,
                lora_r=lora_config['r'],
                lora_alpha=lora_config['alpha']
            )
            score = self._evaluate_config(trial_config, f"LoRA r={lora_config['r']}, α={lora_config['alpha']}")

            if score < best_score:
                best_lora, best_score = lora_config, score

        return best_lora

    def _tune_batch_size(self, best_lr, best_lora):
        """Tune batch size."""
        best_batch, best_score = None, float('inf')

        for batch_size in self.search_spaces['batch_size']:
            trial_config = self._create_config_for_trial(
                learning_rate=best_lr,
                lora_r=best_lora['r'],
                lora_alpha=best_lora['alpha'],
                batch_size=batch_size
            )
            score = self._evaluate_config(trial_config, f"Batch={batch_size}")

            if score < best_score:
                best_batch, best_score = batch_size, score

        return best_batch

    def _tune_lora_dropout(self, best_lr, best_lora, best_batch):
        """Tune LoRA dropout."""
        best_dropout, best_score = None, float('inf')

        for dropout in self.search_spaces['lora_dropout']:
            trial_config = self._create_config_for_trial(
                learning_rate=best_lr,
                lora_r=best_lora['r'],
                lora_alpha=best_lora['alpha'],
                batch_size=best_batch,
                lora_dropout=dropout
            )
            score = self._evaluate_config(trial_config, f"Dropout={dropout}")

            if score < best_score:
                best_dropout, best_score = dropout, score

        return best_dropout

    def _tune_weight_decay(self, best_lr, best_lora, best_batch, best_dropout):
        """Tune weight decay."""
        best_wd, best_score = None, float('inf')

        for wd in self.search_spaces['weight_decay']:
            trial_config = self._create_config_for_trial(
                learning_rate=best_lr,
                lora_r=best_lora['r'],
                lora_alpha=best_lora['alpha'],
                batch_size=best_batch,
                lora_dropout=best_dropout,
                weight_decay=wd
            )
            score = self._evaluate_config(trial_config, f"WD={wd}")

            if score < best_score:
                best_wd, best_score = wd, score

        return best_wd

    def _tune_warmup_steps(self, best_lr, best_lora, best_batch, best_dropout, best_wd):
        """Tune warmup steps."""
        best_warmup, best_score = None, float('inf')

        for warmup in self.search_spaces['warmup_steps']:
            trial_config = self._create_config_for_trial(
                learning_rate=best_lr,
                lora_r=best_lora['r'],
                lora_alpha=best_lora['alpha'],
                batch_size=best_batch,
                lora_dropout=best_dropout,
                weight_decay=best_wd,
                warmup_steps=warmup
            )
            score = self._evaluate_config(trial_config, f"Warmup={warmup}")

            if score < best_score:
                best_warmup, best_score = warmup, score

        return best_warmup

    def _print_optimization_summary(self):
        """Print summary of optimization results."""
        print("\n📋 OPTIMIZATION SUMMARY:")
        print("-" * 40)
        print(f"Learning Rate:     {self.optimization_history['learning_rate']}")
        print(f"LoRA Rank:         {self.optimization_history['lora']['r']}")
        print(f"LoRA Alpha:        {self.optimization_history['lora']['alpha']}")
        print(f"Batch Size:        {self.optimization_history['batch_size']}")
        print(f"LoRA Dropout:      {self.optimization_history['lora_dropout']}")
        print(f"Weight Decay:      {self.optimization_history['weight_decay']}")
        print(f"Warmup Steps:      {self.optimization_history['warmup_steps']}")


def verify_dataset_stratification(tokenized_datasets):
    """Verify and display stratification of loaded datasets."""
    print("\n📊 DATASET STRATIFICATION VERIFICATION")
    print("=" * 60)

    for split_name, dataset in tokenized_datasets.items():
        print(f"\n{split_name.upper()} SET ({len(dataset)} examples):")
        strata_counts = defaultdict(int)

        for example in dataset:
            source = example.get('source', 'unknown')
            age_group = example.get('age_group', 'unknown')
            strata_counts[(source, age_group)] += 1

        total = len(dataset)
        for (source, age_group), count in sorted(strata_counts.items()):
            percentage = (count / total) * 100
            print(f"  {source:25} | {age_group:8} | {count:6} ({percentage:5.1f}%)")


def run_hyperparameter_tuning(config, base_model_path, tokenized_path):
    """Run hyperparameter tuning and return optimal parameters."""
    print("\n🔬 HYPERPARAMETER TUNING ENABLED")
    print("=" * 60)

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

    verify_dataset_stratification(tokenized_datasets)

    tuning_config = config.get('hyperparameter_tuning', {})
    sample_percentage = tuning_config.get('sample_percentage', 5)
    train_sample_size = int(len(tokenized_datasets['train']) * sample_percentage / 100)
    eval_sample_size = int(len(tokenized_datasets['validation']) * sample_percentage / 100)

    print(f"\nCreating {sample_percentage}% stratified samples for tuning:")
    print(f"  Training: {train_sample_size} from {len(tokenized_datasets['train'])}")
    print(f"  Validation: {eval_sample_size} from {len(tokenized_datasets['validation'])}")

    train_sampler = StratifiedSampler(tokenized_datasets['train'])
    eval_sampler = StratifiedSampler(tokenized_datasets['validation'])

    tuning_sample = train_sampler.get_proportional_stratified_sample(train_sample_size)
    eval_sample = eval_sampler.get_proportional_stratified_sample(eval_sample_size)

    torch_columns = ["input_ids", "attention_mask"]
    tuning_sample.set_format("torch", columns=torch_columns)
    eval_sample.set_format("torch", columns=torch_columns)

    if os.path.exists(base_model_path):
        print(f"Loading tokenizer from local model: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        print(f"Loading tokenizer from HuggingFace: {config['model']['base_model']}")
        tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tuner = SequentialHyperparameterTuner(
        base_config=config,
        tokenizer=tokenizer,
        base_model_path=base_model_path,
        tuning_sample=tuning_sample,
        eval_sample=eval_sample
    )

    optimal_hyperparams = tuner.tune_sequential()

    if os.path.exists("./temp_tuning"):
        shutil.rmtree("./temp_tuning")

    return optimal_hyperparams