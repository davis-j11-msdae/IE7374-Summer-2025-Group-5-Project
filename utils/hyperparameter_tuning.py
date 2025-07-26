import os
import sys
import copy
import gc
import json
from datetime import datetime

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import set_cwd

# Get current working directory for path operations
cwd = set_cwd()


def thorough_memory_cleanup():
    """Perform memory cleanup."""
    gc.collect()


class HyperparameterTuner:
    """Hyperparameter tuning orchestrator that calls training functions."""

    def __init__(self, base_config, base_model_path, tokenized_path):
        self.base_config = copy.deepcopy(base_config)
        self.base_model_path = base_model_path
        self.tokenized_path = tokenized_path
        self.optimization_history = {}

        # Setup persistent storage
        self.results_dir = os.path.join(cwd, "hyperparameter_results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.session_file = os.path.join(self.results_dir, "current_session.json")
        self.results_file = os.path.join(self.results_dir, "trial_results.json")
        self.final_results_file = os.path.join(self.results_dir, "final_optimal_hyperparams.json")

        tuning_config = base_config.get('hyperparameter_tuning', {})
        self.search_spaces = tuning_config.get('search_spaces', {})

        # Define hyperparameter optimization order and memory efficiency defaults
        self.tuning_order = [
            {
                'name': 'learning_rate',
                'search_key': 'learning_rate',
                'memory_efficient_default': lambda values: min(values),
                'config_path': ['training', 'learning_rate']
            },
            {
                'name': 'lora',
                'search_key': 'lora_rank_alpha',
                'memory_efficient_default': lambda values: min(values, key=lambda x: x['r']),
                'config_paths': [
                    (['lora', 'r'], 'r'),
                    (['lora', 'lora_alpha'], 'alpha')
                ]
            },
            {
                'name': 'batch_size',
                'search_key': 'batch_size',
                'memory_efficient_default': lambda values: min(values),
                'config_path': ['training', 'batch_size']
            },
            {
                'name': 'lora_dropout',
                'search_key': 'lora_dropout',
                'memory_efficient_default': lambda values: max(values),
                'config_path': ['lora', 'lora_dropout']
            },
            {
                'name': 'weight_decay',
                'search_key': 'weight_decay',
                'memory_efficient_default': lambda values: min(values),
                'config_path': ['training', 'weight_decay']
            },
            {
                'name': 'warmup_steps',
                'search_key': 'warmup_steps',
                'memory_efficient_default': lambda values: min(values),
                'config_path': ['training', 'warmup_steps']
            }
        ]

        # Validate search spaces
        for param in self.tuning_order:
            if param['search_key'] not in self.search_spaces:
                raise ValueError(f"Missing search space for {param['search_key']} in hyperparameter_tuning config")

        # Load existing results
        self.trial_results = self._load_trial_results()
        self.session_state = self._load_session_state()

        # Load previous optimization history if resuming
        if self.session_state.get('optimization_history'):
            self.optimization_history = self.session_state['optimization_history']

        print(f"Loaded hyperparameter search spaces for Mistral 7B:")
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
            'model_used': 'mistral-7b-instruct-v0.3'
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

    def _set_nested_config_value(self, config, path, value):
        """Set a nested configuration value using a path list."""
        current = config
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value

    def _create_trial_hyperparams(self, param_config, test_value):
        """Create hyperparameters dict for a trial."""
        hyperparams = {}

        # Add all previously optimized parameters
        for completed_param in self.tuning_order:
            param_name = completed_param['name']

            if param_name in self.optimization_history:
                hyperparams[param_name] = self.optimization_history[param_name]
            elif param_name == param_config['name']:
                # This is the parameter being tested
                hyperparams[param_name] = test_value
            else:
                # Use memory efficient default for parameters not yet optimized
                search_values = self.search_spaces[completed_param['search_key']]
                hyperparams[param_name] = completed_param['memory_efficient_default'](search_values)

        return hyperparams

    def _call_training_for_evaluation(self, hyperparams, trial_name):
        """Call generic training function for hyperparameter evaluation."""
        print(f"  Testing {trial_name}...")

        try:
            # Import training function
            sys.path.insert(0, cwd)
            import train

            # Create trial output directory
            trial_output_dir = os.path.join(self.results_dir,
                                            f"trial_{trial_name.replace('=', '_').replace('/', '_').replace(' ', '_')}")

            print(f"    Running Mistral 7B training...")

            # Call generic training function with tuning indicator
            validation_loss = train.run_training_generic(
                config=self.base_config,
                hyperparams=hyperparams,
                base_model_path=self.base_model_path,
                tokenized_path=self.tokenized_path,
                output_dir=trial_output_dir,
                tuning_ind=1  # Indicates this is a tuning trial
            )

            print(f"    Validation Loss: {validation_loss:.4f}")

            # Clean up trial directory
            if os.path.exists(trial_output_dir):
                import shutil
                shutil.rmtree(trial_output_dir)

            thorough_memory_cleanup()
            return validation_loss

        except SyntaxError as e:
            print(f"    Syntax error in train.py: {e}")
            thorough_memory_cleanup()
            return float('inf')
        except ImportError as e:
            print(f"    Import error: {e}")
            thorough_memory_cleanup()
            return float('inf')
        except Exception as e:
            print(f"    Trial failed: {str(e)[:100]}...")
            thorough_memory_cleanup()
            return float('inf')

    def _tune_hyperparameter(self, param_config):
        """Generic function to tune any hyperparameter."""
        param_name = param_config['name']
        search_key = param_config['search_key']
        phase = param_name

        print(f"\nPhase: Tuning {param_name} for Mistral 7B...")

        # Check if already completed
        if param_name in self.optimization_history:
            print(f"  {param_name} already completed = {self.optimization_history[param_name]}")
            return self.optimization_history[param_name]

        completed_trials = self._get_completed_trials(phase)
        best_value, best_score = None, float('inf')
        search_values = self.search_spaces[search_key]

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
            hyperparams = self._create_trial_hyperparams(param_config, value)

            # Run trial
            score = self._call_training_for_evaluation(hyperparams, f"{param_name}={value}")

            # Save result
            self._save_trial_result(phase, trial_key, hyperparams, score)

            if score < best_score:
                best_value, best_score = value, score

            thorough_memory_cleanup()

        return best_value

    def tune_sequential(self):
        """Run sequential hyperparameter optimization for Mistral 7B."""
        print("\nSTARTING SEQUENTIAL HYPERPARAMETER TUNING FOR MISTRAL 7B")
        print("=" * 60)

        thorough_memory_cleanup()

        # Resume information
        if self.optimization_history:
            print(f"Resuming from previous session...")
            for param, value in self.optimization_history.items():
                print(f"  Already optimized: {param} = {value}")

        # Tune each hyperparameter in order
        for param_config in self.tuning_order:
            param_name = param_config['name']

            best_value = self._tune_hyperparameter(param_config)
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
        print("\nOPTIMIZATION SUMMARY (Mistral 7B):")
        print("-" * 40)
        print(f"Learning Rate:     {self.optimization_history['learning_rate']}")
        print(f"LoRA Rank:         {self.optimization_history['lora']['r']}")
        print(f"LoRA Alpha:        {self.optimization_history['lora']['alpha']}")
        print(f"Batch Size:        {self.optimization_history['batch_size']}")
        print(f"LoRA Dropout:      {self.optimization_history['lora_dropout']}")
        print(f"Weight Decay:      {self.optimization_history['weight_decay']}")
        print(f"Warmup Steps:      {self.optimization_history['warmup_steps']}")
        print(f"\nResults saved to: {self.results_dir}")


def run_hyperparameter_tuning(config, base_model_path, tokenized_path):
    """Run hyperparameter tuning using training function calls for Mistral 7B."""
    print("\nHYPERPARAMETER TUNING ENABLED FOR MISTRAL 7B")
    print("=" * 60)

    thorough_memory_cleanup()

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
            print(f"Model: {model_used}")
            print(f"Completed at: {completion_time}")
            print("Optimal hyperparameters found:")
            opt_params = results.get('optimal_hyperparameters', {})
            if opt_params:
                print(f"  Learning Rate: {opt_params.get('learning_rate', 'N/A')}")
                if 'lora' in opt_params:
                    print(f"  LoRA Rank: {opt_params['lora'].get('r', 'N/A')}")
                    print(f"  LoRA Alpha: {opt_params['lora'].get('alpha', 'N/A')}")
                print(f"  Batch Size: {opt_params.get('batch_size', 'N/A')}")
                print(f"  LoRA Dropout: {opt_params.get('lora_dropout', 'N/A')}")
                print(f"  Weight Decay: {opt_params.get('weight_decay', 'N/A')}")
                print(f"  Warmup Steps: {opt_params.get('warmup_steps', 'N/A')}")
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
        print("Starting fresh hyperparameter tuning for Mistral 7B...")

    print("Setting up hyperparameter tuning orchestration...")

    # Run hyperparameter tuning orchestration
    tuner = HyperparameterTuner(
        base_config=config,
        base_model_path=base_model_path,
        tokenized_path=tokenized_path
    )

    optimal_hyperparams = tuner.tune_sequential()

    thorough_memory_cleanup()

    print(f"\nHYPERPARAMETER TUNING COMPLETE FOR MISTRAL 7B!")
    print(f"Results stored in: {tuner.results_dir}")

    return optimal_hyperparams