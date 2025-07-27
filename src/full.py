#!/usr/bin/env python3
"""
Main control script for the Personalized Storytelling System.
Provides authentication and role-based menu access.
"""

import sys
import os
import pandas as pd
import importlib.util
import json

if importlib.util.find_spec("google.colab") is not None:
    cwd = os.getcwd()
else:
    cwd = os.getcwd().rstrip(r"\src")
os.chdir(cwd)
import warnings

# Suppress TensorFlow warnings - these come from detoxify using TensorFlow backend
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for consistent results
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow INFO/WARNING messages
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizer warnings

# Filter specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tf_keras")
warnings.filterwarnings("ignore", message=".*tf.losses.sparse_softmax_cross_entropy.*")
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*use_reentrant.*")
# Add utils to path for imports
sys.path.append(os.path.join(cwd, 'utils'))
from helpers import log_operation_status, load_config

config = load_config()
base_model_path = os.path.join(config['paths']['models'], 'mistral-7b-base')
tokenized_path = os.path.join(config['paths']['data_tokenized'], 'datasets')

def authenticate_user():
    """Authenticate user and return user info."""
    users_file = os.path.join(cwd, "data", "users", "users.txt")

    if not os.path.exists(users_file):
        print("Users file not found. Please run generate_users.py first.")
        return None

    try:
        users_df = pd.read_csv(users_file)
    except Exception as e:
        print(f"Error loading users file: {e}")
        return None

    print("\nUSER AUTHENTICATION")
    print("=" * 30)

    username = input("Username: ").strip()
    password = input("Password: ").strip()

    user_row = users_df[users_df['username'] == username]

    if user_row.empty:
        print("Invalid username")
        return None

    user_data = user_row.iloc[0]

    if user_data['password'] != password:
        print("Invalid password")
        return None

    user_info = {
        'username': username,
        'age': int(user_data['age']),
        'admin': int(user_data.get('admin', 0))
    }

    role = "Admin" if user_info['admin'] else "User"
    print(f"Welcome {username}! (Age: {user_info['age']}, Role: {role})")

    return user_info


def admin_menu():
    """Display admin menu and handle selection."""
    while True:
        print("\n" + "=" * 60)
        print("ADMIN MENU - PERSONALIZED STORYTELLING SYSTEM")
        print("=" * 60)
        print("1. Check Environment")
        print("2. Download Data and Models")
        print("3. Process and Evaluate Data")
        print("4. Tokenize Data")
        print("5. Hyperparameter Tuning")
        print("6. Train Model")
        print("7. Process Sample Stories")
        print("8. Interactive Story Creation")
        print("9. Generate Users File")
        print("10. Manual Evaluation Only")
        print("11. Exit")
        print("=" * 60)

        choice = input("Select option (1-11): ").strip()

        if choice == "1":
            check_environment()
        elif choice == "2":
            download_data()
        elif choice == "3":
            process_data()
        elif choice == "4":
            tokenize_data()
        elif choice == "5":
            run_hyperparameter_tuning()
        elif choice == "6":
            train_model()
        elif choice == "7":
            process_samples()
        elif choice == "8":
            interactive_stories()
        elif choice == "9":
            generate_users()
        elif choice == "10":
            evaluate_data()
        elif choice == "11":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")


def check_environment():
    """Check system environment and dependencies."""
    log_operation_status("Environment check")

    from environment_check import run_full_environment_check, print_environment_report

    print("\nChecking system environment...")
    results = run_full_environment_check()
    print_environment_report(results)

    input("\nPress Enter to continue...")


def download_data():
    """Download datasets and models."""
    log_operation_status("Data download")

    from download_data import main as download_main

    print("\nStarting data and model download...")
    print("This will download:")
    print("  - Children's stories dataset from Project Gutenberg")
    print("  - Sci-fi and fantasy stories from Project Gutenberg")
    print("  - Adventure and fairy tale stories from Project Gutenberg")
    print("  - Mistral 7B Instruct v0.3 base model from Hugging Face")
    print("\nNote: This may take 15-30 minutes and requires ~15GB storage")

    confirm = input("\nProceed with download? (y/N): ").strip().lower()
    if confirm == 'y':
        download_main()
    else:
        print("Download cancelled.")

    input("\nPress Enter to continue...")


def process_data():
    """Process raw data files with integrated evaluation."""
    log_operation_status("Data processing with evaluation")

    from data_loader import main as loader_main

    print("\nProcessing and evaluating raw data files...")
    print("This will:")
    print("  - Extract stories from downloaded Project Gutenberg files")
    print("  - Clean and filter content")
    print("  - Assign age groups to stories")
    print("  - Evaluate stories for quality, safety, and appropriateness using Mistral")
    print("  - Filter out toxic or inappropriate content")
    print("  - Save processed and evaluated datasets")
    print("\nNote: Uses Mistral model for evaluation (no API keys needed)")

    loader_main()

    input("\nPress Enter to continue...")


def tokenize_data():
    """Tokenize processed datasets."""
    log_operation_status("Data tokenization")

    from data_tokenizer import main as tokenizer_main

    print("\nTokenizing processed datasets...")
    print("This will:")
    print("  - Load processed and evaluated story datasets")
    print("  - Format stories with age-appropriate instructions using Mistral chat format")
    print("  - Tokenize using Mistral tokenizer")
    print("  - Create train/validation/test splits")
    print("  - Select samples for hyperparameter tuning")

    tokenizer_main()

    input("\nPress Enter to continue...")


def run_hyperparameter_tuning():
    """Run hyperparameter tuning separately."""
    log_operation_status("Hyperparameter tuning")

    print("\nHyperparameter Tuning...")
    print("This will:")
    print("  - Use pre-selected tuning samples from tokenized data")
    print("  - Sequentially optimize learning rate, LoRA params, batch size, etc.")
    print("  - Save optimal hyperparameters for training")
    print("  - Resume capability if interrupted")
    print("\nNote: This process can take 1-2 hours depending on GPU")

    confirm = input("\nProceed with hyperparameter tuning? (y/N): ").strip().lower()
    if confirm == 'y':
        try:
            from hyperparameter_tuning import run_hyperparameter_tuning

            if not os.path.exists(tokenized_path):
                print(f"Tokenized datasets not found at: {tokenized_path}")
                print("Please run tokenization first (option 4).")
                input("\nPress Enter to continue...")
                returnad

            optimal_hyperparams = run_hyperparameter_tuning(config, base_model_path, tokenized_path)

            if optimal_hyperparams:
                print("\nHyperparameter tuning completed successfully!")
                print("Results saved for use in training")
            else:
                print("\nHyperparameter tuning failed or was cancelled")

        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Hyperparameter tuning cancelled.")

    input("\nPress Enter to continue...")


def load_optimal_hyperparameters():
    """Load optimal hyperparameters from tuning results."""
    results_dir = os.path.join(cwd, "hyperparameter_results")
    final_results_file = os.path.join(results_dir, "final_optimal_hyperparams.json")

    if os.path.exists(final_results_file):
        try:
            with open(final_results_file, 'r') as f:
                results = json.load(f)
            return results.get('optimal_hyperparameters')
        except Exception as e:
            print(f"Could not load optimal hyperparameters: {e}")

    return None


def train_model():
    """Train the storytelling model with optimal hyperparameters."""
    log_operation_status("Model training")

    print("\nTraining storytelling model...")

    # Define paths
    from helpers import load_config,load_datasets
    config = load_config()
    output_dir = os.path.join(config['paths']['models'], 'tuned_story_llm')
    tokenized_path = os.path.join(config['paths']['data_tokenized'], 'datasets')

    # Check if tokenized data exists
    if not os.path.exists(tokenized_path):
        print(f"Tokenized datasets not found at: {tokenized_path}")
        print("Please run tokenization first (option 4).")
        input("\nPress Enter to continue...")
        return

    # Check for existing trained model
    if os.path.exists(output_dir):
        # Import training state check function
        sys.path.insert(0, cwd)
        import train
        training_state, training_info = train.check_training_state(output_dir)

        if training_state == "completed" or training_state == "early_stopped":
            print("Model training already completed.")
            if training_state == "early_stopped":
                print("Previous training ended due to early stopping.")

            restart = input("\nRestart model training from scratch? (y/N): ").strip().lower()
            if restart != 'y':
                print("Training cancelled.")
                input("\nPress Enter to continue...")
                return
            else:
                print("Removing existing model to restart training...")
                import shutil
                shutil.rmtree(output_dir)
                training_state = "new"

        elif training_state == "resumable":
            print("Incomplete training found.")
            resume = input("\nResume training for additional 3 epochs? (Y/n): ").strip().lower()
            if resume == 'n':
                print("Training cancelled.")
                input("\nPress Enter to continue...")
                return
    else:
        training_state = "new"

    # Check for optimal hyperparameters if starting new training
    optimal_hyperparams = None
    if training_state == "new":
        optimal_hyperparams = load_optimal_hyperparameters()

        if optimal_hyperparams:
            print("Found optimal hyperparameters from tuning:")
            for param, value in optimal_hyperparams.items():
                if isinstance(value, dict):
                    print(f"  {param.replace('_', ' ').title()}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {param.replace('_', ' ').title()}: {value}")

            use_tuned = input("\nUse these optimized hyperparameters? (Y/n): ").strip().lower()
            if use_tuned == 'n':
                optimal_hyperparams = None
                print("Will use default hyperparameters from config file")
        else:
            print("WARNING: No optimal hyperparameters found.")
            print("For best results, run hyperparameter tuning first (option 5).")

            proceed = input("\nProceed with default hyperparameters? (y/N): ").strip().lower()
            if proceed != 'y':
                print("Training cancelled.")
                input("\nPress Enter to continue...")
                return

    print("\nThis will:")
    print("  - Load Mistral 7B Instruct v0.3 base model")
    print("  - Fine-tune on processed story datasets")
    print("  - Use optimized memory settings for 10GB VRAM")
    if training_state == "resumable":
        print("  - Resume from previous checkpoint for 3 additional epochs")
    elif optimal_hyperparams:
        print("  - Use optimized hyperparameters")
    else:
        print("  - Use default hyperparameters")
    print("  - Save fine-tuned model")
    print("\nNote: Requires 8-10GB GPU memory and time (2-3 hours)")

    confirm = input("\nProceed with training? (y/N): ").strip().lower()
    if confirm == 'y':
        try:
            # Import and run the training script
            sys.path.insert(0, cwd)
            import train
            train.run_training_core(
                config=config,
                hyperparams=optimal_hyperparams,
                base_model_path=base_model_path,
                tokenized_datasets=load_datasets(tokenized_path,training_state),
                output_dir=output_dir,
                training_state=training_state,
                save_strategy='epoch',
                save_total_limit=5,
                load_best_model_at_end=True
            )
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Training cancelled.")

    input("\nPress Enter to continue...")


def evaluate_data():
    """Manually evaluate processed datasets."""
    log_operation_status("Manual data evaluation")

    from eval import main as eval_main

    print("\nManually evaluating processed datasets...")
    print("This will:")
    print("  - Analyze text quality (grammar, coherence) using Mistral")
    print("  - Calculate readability scores")
    print("  - Check content safety (toxicity)")
    print("  - Generate evaluation statistics")
    print("\nNote: This is only needed if evaluation wasn't done during processing")
    print("      or if you want to re-evaluate with different parameters")

    confirm = input("\nProceed with manual evaluation? (y/N): ").strip().lower()
    if confirm == 'y':
        eval_main()
    else:
        print("Manual evaluation cancelled.")

    input("\nPress Enter to continue...")


def process_samples():
    """Process sample stories."""
    log_operation_status("Sample processing")

    from samples import main as samples_main

    print("\nProcessing sample stories...")
    print("This will:")
    print("  - Generate stories for 10 sample prompts")
    print("  - Test all age groups (child, kid, teen, adult)")
    print("  - Include story continuation examples")
    print("  - Save stories to user history")
    print("  - Generate comprehensive evaluation report")

    samples_main()

    input("\nPress Enter to continue...")


def interactive_stories():
    """Run interactive story creation."""
    log_operation_status("Interactive story session")

    from model_runner import StoryModelRunner

    print("\nStarting interactive story creation...")
    print("This will:")
    print("  - Generate personalized stories using fine-tuned Mistral")
    print("  - Support story continuation")
    print("  - Manage story history with Mistral-based summarization")

    runner = StoryModelRunner()

    if runner.users_df.empty:
        print("No users found. Please run generate_users.py first.")
        input("\nPress Enter to continue...")
        return

    if not runner.load_model():
        print("Failed to load model. Please run training first.")
        input("\nPress Enter to continue...")
        return

    runner.story_session_authenticated()

    input("\nPress Enter to continue...")


def generate_users():
    """Generate users file."""
    log_operation_status("User generation")

    from generate_users import generate_users

    print("\nGenerating users file...")
    print("This will create/update users with credentials:")
    print("  - Standard users: child_1, kid_1, teen_1, adult_1, etc.")
    print("  - Admin user: admin (password: admin)")
    print("  - Preserves existing users in file")

    generate_users()

    input("\nPress Enter to continue...")


def setup_directories():
    """Create required directory structure."""
    from helpers import ensure_dir_exists, load_config

    config = load_config()
    paths = config['paths']

    directories = [
        paths['data_root'],
        paths['data_raw'],
        paths['data_processed'],
        paths['data_tokenized'],
        paths['data_evaluated'],
        paths['models'],
        paths['outputs'],
        paths['user_history'],
        paths['samples'],
        paths['users'],
        "logs",
        "hyperparameter_results"
    ]

    for directory in directories:
        ensure_dir_exists(directory)


def display_welcome():
    """Display welcome message and system information."""
    print("PERSONALIZED STORYTELLING SYSTEM")
    print("=" * 60)
    print("An AI-powered storytelling system using Mistral 7B Instruct v0.3")
    print("Features:")
    print("  • Age-appropriate story generation (child, kid, teen, adult)")
    print("  • Integrated quality evaluation and safety filtering")
    print("  • Story history and continuation")
    print("  • Interactive user sessions")
    print("  • Hyperparameter optimization")
    print("  • Memory-optimized for 10GB VRAM")
    print("=" * 60)


def main():
    """Main entry point with authentication."""
    setup_directories()
    display_welcome()

    config_file = os.path.join(cwd, "configs", "model_config.yaml")
    if not os.path.exists(config_file):
        print("\nConfiguration file not found!")
        print("Please ensure configs/model_config.yaml exists.")
        return

    user_info = authenticate_user()
    if not user_info:
        print("Authentication failed")
        return

    if user_info['admin']:
        print("\nRECOMMENDED WORKFLOW:")
        print("1. Check Environment → 2. Download Data → 9. Generate Users →")
        print("3. Process Data → 4. Tokenize → 5. Hyperparameter Tuning → 6. Train → 7. Test Samples → 8. Interactive")
        admin_menu()
    else:
        print(f"\nWelcome to Interactive Story Creation!")
        interactive_stories()


if __name__ == "__main__":
    main()