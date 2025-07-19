#!/usr/bin/env python3
"""
Main control script for the Personalized Storytelling System.
Provides a menu-driven interface for all system operations.
"""

import sys
from pathlib import Path
from utils.helpers import log_operation_status


def main_menu():
    """Display main menu and handle user selection."""
    while True:
        print("\n" + "=" * 60)
        print("🎭 PERSONALIZED STORYTELLING SYSTEM")
        print("=" * 60)
        print("1. Check Environment")
        print("2. Download Data and Models")
        print("3. Process and Evaluate Data")
        print("4. Tokenize Data")
        print("5. Train Model")
        print("6. Process Sample Stories")
        print("7. Interactive Story Creation")
        print("8. Generate Users File")
        print("9. Manual Evaluation Only")
        print("10. Exit")
        print("=" * 60)

        choice = input("Select option (1-10): ").strip()

        if choice == "1":
            check_environment()
        elif choice == "2":
            download_data()
        elif choice == "3":
            process_data()
        elif choice == "4":
            tokenize_data()
        elif choice == "5":
            train_model()
        elif choice == "6":
            process_samples()
        elif choice == "7":
            interactive_stories()
        elif choice == "8":
            generate_users()
        elif choice == "9":
            evaluate_data()
        elif choice == "10":
            print("👋 Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please try again.")


def check_environment():
    """Check system environment and dependencies."""
    log_operation_status("Environment check")

    try:
        from src.environment_check import run_full_environment_check, print_environment_report

        print("\n🔍 Checking system environment...")
        results = run_full_environment_check()
        print_environment_report(results)

    except ImportError as e:
        print(f"❌ Error importing environment check: {e}")
    except Exception as e:
        print(f"❌ Environment check failed: {e}")

    input("\nPress Enter to continue...")


def download_data():
    """Download datasets and models."""
    log_operation_status("Data download")

    try:
        from src.download_data import main as download_main

        print("\n📦 Starting data and model download...")
        print("This will download:")
        print("  - Children's stories dataset from Project Gutenberg")
        print("  - Sci-fi and fantasy stories from Project Gutenberg")
        print("  - Adventure and fairy tale stories from Project Gutenberg")
        print("  - Mixtral 8x7B base model from Hugging Face")
        print("\nNote: This may take 30-60 minutes and requires ~50GB storage")

        confirm = input("\nProceed with download? (y/N): ").strip().lower()
        if confirm == 'y':
            download_main()
        else:
            print("Download cancelled.")

    except ImportError as e:
        print(f"❌ Error importing download module: {e}")
    except Exception as e:
        print(f"❌ Download failed: {e}")

    input("\nPress Enter to continue...")


def process_data():
    """Process raw data files with integrated evaluation."""
    log_operation_status("Data processing with evaluation")

    try:
        from src.data_loader import main as loader_main

        print("\n⚙️ Processing and evaluating raw data files...")
        print("This will:")
        print("  - Extract stories from downloaded Project Gutenberg files")
        print("  - Clean and filter content")
        print("  - Assign age groups to stories")
        print("  - Evaluate stories for quality, safety, and appropriateness")
        print("  - Filter out toxic or inappropriate content")
        print("  - Save processed and evaluated datasets")
        print("\nNote: Requires OpenAI API key for grammar/coherence evaluation")

        loader_main()

    except ImportError as e:
        print(f"❌ Error importing data loader: {e}")
    except Exception as e:
        print(f"❌ Data processing failed: {e}")

    input("\nPress Enter to continue...")


def tokenize_data():
    """Tokenize processed datasets."""
    log_operation_status("Data tokenization")

    try:
        from src.data_tokenizer import main as tokenizer_main

        print("\n🔤 Tokenizing processed datasets...")
        print("This will:")
        print("  - Load processed and evaluated story datasets")
        print("  - Format stories with age-appropriate instructions")
        print("  - Tokenize using Mixtral tokenizer")
        print("  - Create train/validation/test splits")

        tokenizer_main()

    except ImportError as e:
        print(f"❌ Error importing tokenizer: {e}")
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")

    input("\nPress Enter to continue...")


def evaluate_data():
    """Manually evaluate processed datasets (if not done during processing)."""
    log_operation_status("Manual data evaluation")

    try:
        from src.eval import main as eval_main

        print("\n📊 Manually evaluating processed datasets...")
        print("This will:")
        print("  - Analyze text quality (grammar, coherence)")
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

    except ImportError as e:
        print(f"❌ Error importing evaluation module: {e}")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

    input("\nPress Enter to continue...")


def train_model():
    """Train the storytelling model."""
    log_operation_status("Model training")

    try:
        from src.train import main as train_main

        print("\n🚀 Training storytelling model...")
        print("This will:")
        print("  - Load Mixtral 8x7B base model")
        print("  - Fine-tune on processed story datasets")
        print("  - Use DeepSpeed for memory optimization")
        print("  - Save fine-tuned model")
        print("\nNote: Requires significant GPU memory and time (1-3 hours)")

        confirm = input("\nProceed with training? (y/N): ").strip().lower()
        if confirm == 'y':
            train_main()
        else:
            print("Training cancelled.")

    except ImportError as e:
        print(f"❌ Error importing training module: {e}")
    except Exception as e:
        print(f"❌ Training failed: {e}")

    input("\nPress Enter to continue...")


def process_samples():
    """Process sample stories."""
    log_operation_status("Sample processing")

    try:
        from src.samples import main as samples_main

        print("\n📝 Processing sample stories...")
        print("This will:")
        print("  - Generate stories for 10 sample prompts")
        print("  - Test all age groups (child, kid, teen, adult)")
        print("  - Include story continuation examples")
        print("  - Save stories to user history")
        print("  - Generate comprehensive evaluation report")

        samples_main()

    except ImportError as e:
        print(f"❌ Error importing samples module: {e}")
    except Exception as e:
        print(f"❌ Sample processing failed: {e}")

    input("\nPress Enter to continue...")


def interactive_stories():
    """Run interactive story creation."""
    log_operation_status("Interactive story session")

    try:
        from src.model_runner import main as runner_main

        print("\n🎭 Starting interactive story creation...")
        print("This will:")
        print("  - Authenticate user credentials")
        print("  - Generate personalized stories")
        print("  - Support story continuation")
        print("  - Manage story history")

        runner_main()

    except ImportError as e:
        print(f"❌ Error importing model runner: {e}")
    except Exception as e:
        print(f"❌ Interactive session failed: {e}")

    input("\nPress Enter to continue...")


def generate_users():
    """Generate users file."""
    log_operation_status("User generation")

    try:
        from generate_users import generate_users

        print("\n👥 Generating users file...")
        print("This will create 20 users (5 per age group) with credentials:")
        print("  - Usernames: child_1, kid_1, teen_1, adult_1, etc.")
        print("  - Password: 'test' for all users")
        print("  - Ages distributed across age groups")

        generate_users()

    except ImportError as e:
        print(f"❌ Error importing user generator: {e}")
    except Exception as e:
        print(f"❌ User generation failed: {e}")

    input("\nPress Enter to continue...")


def setup_directories():
    """Create required directory structure."""
    from utils.helpers import ensure_dir_exists, load_config

    try:
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
            "logs"
        ]

        for directory in directories:
            ensure_dir_exists(directory)

    except Exception as e:
        print(f"⚠️ Warning: Could not create directories: {e}")


def display_welcome():
    """Display welcome message and system information."""
    print("🎭 PERSONALIZED STORYTELLING SYSTEM")
    print("=" * 60)
    print("An AI-powered storytelling system using Mixtral 8x7B")
    print("Features:")
    print("  • Age-appropriate story generation (child, kid, teen, adult)")
    print("  • Integrated quality evaluation and safety filtering")
    print("  • Story history and continuation")
    print("  • Interactive user sessions")
    print("=" * 60)


def main():
    """Main entry point."""
    # Setup environment
    setup_directories()

    # Display welcome message
    display_welcome()

    # Check if this is first run
    config_file = Path("configs/model_config.yaml")
    users_file = Path("data/users/users.txt")

    if not config_file.exists():
        print("\n⚠️ Configuration file not found!")
        print("Please ensure configs/model_config.yaml exists.")
        return

    if not users_file.exists():
        print("\n💡 Users file not found. Run option 8 to generate users first.")

    print("\n🔄 RECOMMENDED WORKFLOW:")
    print("1. Check Environment → 2. Download Data → 8. Generate Users →")
    print("3. Process Data → 4. Tokenize → 5. Train → 6. Test Samples → 7. Interactive")

    # Start main menu loop
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 System interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()