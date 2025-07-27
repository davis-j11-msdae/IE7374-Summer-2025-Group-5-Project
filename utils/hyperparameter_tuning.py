import os
import sys
import copy
import gc
import json
import torch
from datetime import datetime

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import set_cwd, load_datasets

# Get current working directory for path operations
cwd = set_cwd()


def thorough_memory_cleanup():
    """Perform memory cleanup."""
    gc.collect()


def get_gpu_memory_gb():
    """Get GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return 0


def get_tuning_config(target_effective_batch_size=16):
    """Get batch_size and gradient_accumulation for target effective batch size."""
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    if gpu_memory_gb >= 40:  # A100
        return {
            'batch_size': target_effective_batch_size,
            'gradient_accumulation_steps': 1
        }
    else:  # 10GB GPU
        limited_gpu_batch_size = 4
        return {
            'batch_size': limited_gpu_batch_size,
            'gradient_accumulation_steps': target_effective_batch_size // limited_gpu_batch_size
        }



class HyperparameterTuner:
    """Hyperparameter tuning orchestrator driven by config search_spaces."""

    def __init__(self, base_config, base_model_path, tokenized_datasets):
        self.base_config = copy.deepcopy(base_config)
        self.base_model_path = base_model_path
        self.tokenized_datasets = tokenized_datasets
        self.optimization_history = {}

        # Setup persistent storage
        self.results_dir = os.path.join(cwd, "hyperparameter_results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.session_file = os.path.join(self.results_dir, "current_session.json")
        self.results_file = os.path.join(self.results_dir, "trial_results.json")
        self.final_results_file = os.path.join(self.results_dir, "final_optimal_hyperparams.json")

        # Load search spaces from config
        tuning_config = base_config.get('hyperparameter_tuning', {})
        self.search_spaces = tuning_config.get('search_spaces', {})

        if not self.search_spaces:
            raise ValueError("No search_spaces found in hyperparameter_tuning config")

        # Load existing results
        self.trial_results = self._load_trial_results()
        self.session_state = self._load_session_state()

        # Load previous optimization history if resuming
        if self.session_state.get('optimization_history'):
            self.optimization_history = self.session_state['optimization_history']

        gpu_memory_gb = get_gpu_memory_gb()
        gpu_type = "A100" if gpu_memory_gb >= 40 else f"{gpu_memory_gb:.0f}GB GPU"

        print(f"Loaded hyperparameter search spaces for {gpu_type}:")
        for space_name, space_values in self.search_spaces.items():
            print(f"  {space_name}: {space_values}")

        if self.trial_results:
            print(f"\nFound {len(self.trial_results)} previous trial results")
            print(f"Session state: {self.session_state.get('current_phase', 'Not started')}")

    def _load_trial_results(self):
        """Load previous trial results if they exist."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Could not load trial results: {e}")
        return {}

    def _load_session_state(self):
        """Load session state if it exists."""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Could not load session state: {e}")
        return {}

    def _save_trial_result(self, phase, trial_key, hyperparams, score, status="completed"):
        """Save individual trial result."""
        trial_data = {
            'phase': phase,
            'trial_key': trial_key,
            'hyperparams': hyperparams,
            'validation_loss': score if score != float('inf') else None,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }

        self.trial_results[f"{phase}_{trial_key}"] = trial_data

        with open(self.results_file, 'w') as f:
            json.dump(self.trial_results, f, indent=2)

        score_str = 'FAILED' if score == float('inf') else f"{score:.4f}"
        print(f"    Saved trial result: {phase}_{trial_key} -> {score_str}")

    def _save_session_state(self, current_phase):
        """Save current session state."""
        session_data = {
            'current_phase': current_phase,
            'optimization_history': self.optimization_history,
            'timestamp': datetime.now().isoformat(),
            'completed_phases': list(self.optimization_history.keys())
        }

        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

    def _save_final_results(self):
        """Save final optimal hyperparameters."""
        final_results = {
            'optimal_hyperparameters': self.optimization_history,
            'search_spaces_used': self.search_spaces,
            'completion_timestamp': datetime.now().isoformat(),
            'total_trials_run': len(self.trial_results),
            'base_config_hash': hash(str(self.base_config)),
            'model_used': 'mistral-7b-instruct-v0.3',
            'gpu_memory_gb': get_gpu_memory_gb()
        }

        with open(self.final_results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"Final results saved to: {self.final_results_file}")

    def _get_completed_trials(self, phase):
        """Get completed trials for a specific phase."""
        completed = {}
        for trial_id, trial_data in self.trial_results.items():
            if trial_data['phase'] == phase and trial_data['status'] == 'completed':
                trial_key = trial_data['trial_key']
                score = trial_data['validation_loss']
                if score is not None:
                    completed[trial_key] = score
        return completed

    def _create_trial_hyperparams(self, current_param_name, test_value):
        """Create hyperparameters dict for a trial based on config-driven approach."""
        hyperparams = {}

        # Get search spaces directly from config
        search_spaces = self.base_config['hyperparameter_tuning']['search_spaces']

        # Add all parameters from search spaces
        for param_name in search_spaces.keys():
            if param_name in self.optimization_history:
                # Use previously optimized value
                hyperparams[param_name] = self.optimization_history[param_name]
            elif param_name == current_param_name:
                # This is the parameter being tested
                hyperparams[param_name] = test_value
            elif 'lora' in param_name:
                hyperparams[param_name] = self.base_config['lora'][param_name]
            else:
                hyperparams[param_name] = self.base_config['training'][param_name]

        return hyperparams


    def _call_training_for_evaluation(self, hyperparams, trial_name):
        """Call training function for hyperparameter evaluation."""
        print(f"  Testing {trial_name}...")

        try:
            # Import training function
            sys.path.insert(0, cwd)
            import train

            # Create trial output directory
            trial_output_dir = os.path.join(self.results_dir,
                                            f"trial_{trial_name.replace('=', '_').replace('/', '_').replace(' ', '_')}")

            print(f"    Running Mistral 7B training...")

            # Call training function with hyperparameters
            validation_loss = train.run_training_core(
                config=self.base_config,
                hyperparams=hyperparams,
                base_model_path=self.base_model_path,
                tokenized_datasets=self.tokenized_datasets,
                output_dir=trial_output_dir,
                training_state='tuning',
                save_strategy='no',
                save_total_limit=0,
                load_best_model_at_end=False
            )

            print(f"    Validation Loss: {validation_loss:.4f}")

            # Clean up trial directory
            if os.path.exists(trial_output_dir):
                import shutil
                shutil.rmtree(trial_output_dir)

            thorough_memory_cleanup()
            return validation_loss

        except Exception as e:
            print(f"    Trial failed: {str(e)[:500]}...")
            print(f"    Error type: {type(e).__name__}")
            print(f"    Trial name: {trial_name}")
            print(f"    Hyperparams: {hyperparams}")
            
            # Print more detailed error information
            import traceback
            error_details = traceback.format_exc()
            print(f"    Full traceback (last 10 lines):")
            traceback_lines = error_details.split('\n')
            for line in traceback_lines[-12:-1]:  # Last 10 lines plus error
                if line.strip():
                    print(f"      {line}")
            
            # Check for specific common errors
            error_str = str(e).lower()
            if 'cuda' in error_str and 'memory' in error_str:
                print(f"    >> CUDA memory error detected - may need smaller batch size")
            elif 'length' in error_str:
                print(f"    >> Length error detected - likely dataset formatting issue")
            elif 'shape' in error_str or 'size' in error_str:
                print(f"    >> Shape/size mismatch error - tensor dimension problem")
            elif 'device' in error_str:
                print(f"    >> Device error - model/data device mismatch")
            elif 'file' in error_str or 'path' in error_str:
                print(f"    >> File system error - Google Drive mount issue")

        thorough_memory_cleanup()

        return float('inf')

    def _handle_special_parameters(self, param_name, value, hyperparams):
        """Handle special parameter configurations like effective_batch_size and lora_rank."""

        if param_name == 'effective_batch_size':
            # Convert effective batch size to actual batch configuration
            batch_config = get_tuning_config(value)
            hyperparams['batch_size'] = batch_config['batch_size']
            hyperparams['gradient_accumulation_steps'] = batch_config['gradient_accumulation_steps']

        elif param_name == 'lora_rank':
            # Set both rank and alpha (alpha = 2x rank)
            hyperparams['lora_rank'] = value
            hyperparams['lora_alpha'] = value * 2
            print(f"    LoRA rank {value} -> alpha={value * 2}")

    def _format_optimization_result(self, param_name, value, hyperparams):
        """Format the optimization result based on parameter type."""

        if param_name == 'effective_batch_size':
            batch_config = get_tuning_config(value)
            hyperparams['batch_size'] = batch_config['batch_size']
            hyperparams['gradient_accumulation_steps'] = batch_config['gradient_accumulation_steps']
            hyperparams['effective_batch_size'] = value

        elif param_name == 'lora_rank':
            if isinstance(value, dict):
                rank_value = value.get('lora_rank', value.get('r', 16))
            else:
                rank_value = value
            hyperparams['lora_rank'] = rank_value
            hyperparams['lora_alpha'] = rank_value * 2


    def _tune_hyperparameter(self, param_name, search_values):
        """Generic function to tune any hyperparameter."""
        print(f"\nPhase: Tuning {param_name}...")
        # Check if already completed
        if param_name in self.optimization_history:
            print(f"  {param_name} already completed = {self.optimization_history[param_name]}")
            return self.optimization_history[param_name]

        completed_trials = self._get_completed_trials(param_name)
        best_value, best_score = None, float('inf')

        for value in search_values:
            # Create trial key
            if isinstance(value, dict):
                trial_key = "_".join([f"{k}{v}" for k, v in value.items()])
            else:
                trial_key = str(value)

            # Skip if already completed
            if trial_key in completed_trials:
                score = completed_trials[trial_key]
                print(f"  Skipping completed trial: {value} -> {score:.4f}")
                if score < best_score:
                    best_value, best_score = value, score
                continue

            # Create hyperparameters for this trial
            hyperparams = self._create_trial_hyperparams(param_name, value)

            # Handle special parameter configurations
            self._handle_special_parameters(param_name, value, hyperparams)

            # Run trial
            score = self._call_training_for_evaluation(hyperparams, f"{param_name}={value}")

            # Save result
            self._save_trial_result(param_name, trial_key, hyperparams, score)

            if score < best_score:
                best_value, best_score = value, score

            thorough_memory_cleanup()

        # Return the best value directly without formatting
        return best_value

    def tune_sequential(self):
        """Run sequential hyperparameter optimization based on config."""
        print("\nSTARTING SEQUENTIAL HYPERPARAMETER TUNING")
        print("=" * 60)
        print(f"Training samples: {len(self.tokenized_datasets['train'])}")
        gpu_memory_gb = get_gpu_memory_gb()
        print(f"GPU Memory: {gpu_memory_gb:.1f}GB")

        thorough_memory_cleanup()

        # Resume information
        if self.optimization_history:
            print(f"Resuming from previous session...")
            for param, value in self.optimization_history.items():
                print(f"  Already optimized: {param} = {value}")

        # Tune each hyperparameter in order defined by config
        for param_name, search_values in self.search_spaces.items():

            best_value = self._tune_hyperparameter(param_name, search_values)
            self.optimization_history[param_name] = best_value
            self._save_session_state(f"{param_name}_completed")
            print(f"Best {param_name}: {best_value}")

        print("\nHYPERPARAMETER TUNING COMPLETED!")
        print("=" * 60)
        self._print_optimization_summary()

        self._save_final_results()
        thorough_memory_cleanup()

        return self.optimization_history

    def _print_optimization_summary(self):
        """Print summary of optimization results."""
        print("\nOPTIMIZATION SUMMARY:")
        print("-" * 40)

        for param_name, value in self.optimization_history.items():
            if isinstance(value, dict):
                print(f"{param_name.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{param_name.replace('_', ' ').title()}: {value}")

        print(f"\nResults saved to: {self.results_dir}")


def run_hyperparameter_tuning(config, base_model_path, tokenized_path):
    """Run hyperparameter tuning using config-driven approach."""
    print("\nHYPERPARAMETER TUNING ENABLED")
    print("=" * 60)
    print(tokenized_path)
    thorough_memory_cleanup()
    gpu_memory_gb = get_gpu_memory_gb()
    # Check for existing results and prompt user
    results_dir = os.path.join(cwd, "hyperparameter_results")
    final_results_file = os.path.join(results_dir, "final_optimal_hyperparams.json")
    session_file = os.path.join(results_dir, "current_session.json")

    # Check if tuning is already completed
    if os.path.exists(final_results_file):
        print("Hyperparameter tuning already completed.")
        try:
            with open(final_results_file, 'r') as f:
                results = json.load(f)
            completion_time = results.get('completion_timestamp', 'unknown')
            model_used = results.get('model_used', 'unknown')
            gpu_memory = results.get('gpu_memory_gb', 'unknown')
            print(f"Model: {model_used}")
            print(f"GPU: {gpu_memory}GB")
            print(f"Completed at: {completion_time}")
            print("Optimal hyperparameters found:")
            opt_params = results.get('optimal_hyperparameters', {})
            if opt_params:
                for param_name, param_value in opt_params.items():
                    if isinstance(param_value, dict):
                        print(f"  {param_name}:")
                        for sub_key, sub_value in param_value.items():
                            print(f"    {sub_key}: {sub_value}")
                    else:
                        print(f"  {param_name}: {param_value}")
        except Exception as e:
            print(f"Error reading results: {e}")

        response = input("\nRestart hyperparameter tuning from scratch? (y/N): ").strip().lower()
        if response != 'y':
            print("Using existing hyperparameter tuning results.")
            try:
                with open(final_results_file, 'r') as f:
                    results = json.load(f)
                return results.get('optimal_hyperparameters')
            except Exception:
                print("Error loading existing results. Starting fresh.")
        else:
            print("Removing existing results and starting fresh...")
            if os.path.exists(results_dir):
                import shutil
                shutil.rmtree(results_dir)

    # Check if tuning is in progress
    elif os.path.exists(session_file):
        print("Hyperparameter tuning session found (incomplete).")
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            current_phase = session_data.get('current_phase', 'unknown')
            completed_phases = session_data.get('completed_phases', [])
            print(f"Current phase: {current_phase}")
            print(f"Completed phases: {completed_phases}")
        except Exception as e:
            print(f"Error reading session: {e}")

        response = input("\nContinue from where it left off? (Y/n): ").strip().lower()
        if response == 'n':
            print("Removing existing session and starting fresh...")
            if os.path.exists(results_dir):
                import shutil
                shutil.rmtree(results_dir)
        else:
            print("Resuming hyperparameter tuning...")

    else:
        print(f"Starting fresh hyperparameter tuning on {gpu_memory_gb:.1f}GB GPU...")

    print("Setting up config-driven hyperparameter tuning...")

    # Run hyperparameter tuning orchestration
    tuner = HyperparameterTuner(
        base_config=config,
        base_model_path=base_model_path,
        tokenized_datasets=load_datasets(tokenized_path,'sample')
    )

    optimal_hyperparams = tuner.tune_sequential()

    thorough_memory_cleanup()

    print(f"\nHYPERPARAMETER TUNING COMPLETE!")
    print(f"Results stored in: {tuner.results_dir}")

    return optimal_hyperparams