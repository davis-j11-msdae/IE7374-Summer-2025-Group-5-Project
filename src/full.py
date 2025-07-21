#!/usr/bin/env python3
"""
Main control script for the Personalized Storytelling System.
Provides authentication and role-based menu access.
"""

import sys
import os
import pandas as pd
import importlib.util


if importlib.util.find_spec("google.colab") is not None:
    cwd = os.getcwd()
else:
    cwd = os.getcwd().rstrip(r"\src")
os.chdir(cwd)

# Add utils to path for imports
sys.path.append(os.path.join(cwd, 'utils'))
from helpers import log_operation_status


def authenticate_user():
    """Authenticate user and return user info."""
    users_file = os.path.join(cwd, "data", "users", "users.txt")

    if not os.path.exists(users_file):
        print("❌ Users file not found. Please run generate_users.py first.")
        return None

    try:
        users_df = pd.read_csv(users_file)
    except Exception as e:
        print(f"❌ Error loading users file: {e}")
        return None

    print("\n🔐 USER AUTHENTICATION")
    print("=" * 30)

    username = input("Username: ").strip()
    password = input("Password: ").strip()

    user_row = users_df[users_df['username'] == username]

    if user_row.empty:
        print("❌ Invalid username")
        return None

    user_data = user_row.iloc[0]

    if user_data['password'] != password:
        print("❌ Invalid password")
        return None

    user_info = {
        'username': username,
        'age': int(user_data['age']),
        'admin': int(user_data.get('admin', 0))
    }

    role = "Admin" if user_info['admin'] else "User"
    print(f"✅ Welcome {username}! (Age: {user_info['age']}, Role: {role})")

    return user_info


def admin_menu():
    """Display admin menu and handle selection."""
    while True:
        print("\n" + "=" * 60)
        print("🛠️ ADMIN MENU - PERSONALIZED STORYTELLING SYSTEM")
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

    from environment_check import run_full_environment_check, print_environment_report

    print("\n🔍 Checking system environment...")
    results = run_full_environment_check()
    print_environment_report(results)

    input("\nPress Enter to continue...")


def download_data():
    """Download datasets and models."""
    log_operation_status("Data download")

    from download_data import main as download_main

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

    input("\nPress Enter to continue...")


def process_data():
    """Process raw data files with integrated evaluation."""
    log_operation_status("Data processing with evaluation")

    from data_loader import main as loader_main

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

    input("\nPress Enter to continue...")


def tokenize_data():
    """Tokenize processed datasets."""
    log_operation_status("Data tokenization")

    from data_tokenizer import main as tokenizer_main

    print("\n🔤 Tokenizing processed datasets...")
    print("This will:")
    print("  - Load processed and evaluated story datasets")
    print("  - Format stories with age-appropriate instructions")
    print("  - Tokenize using Mixtral tokenizer")
    print("  - Create train/validation/test splits")

    tokenizer_main()

    input("\nPress Enter to continue...")


def evaluate_data():
    """Manually evaluate processed datasets."""
    log_operation_status("Manual data evaluation")

    from eval import main as eval_main

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

    input("\nPress Enter to continue...")


def train_model():
    """Train the storytelling model."""
    log_operation_status("Model training")

    from train import main as train_main

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

    input("\nPress Enter to continue...")


def process_samples():
    """Process sample stories."""
    log_operation_status("Sample processing")

    from samples import main as samples_main

    print("\n📝 Processing sample stories...")
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

    print("\n🎭 Starting interactive story creation...")
    print("This will:")
    print("  - Generate personalized stories")
    print("  - Support story continuation")
    print("  - Manage story history")

    runner = StoryModelRunner()

    if runner.users_df.empty:
        print("❌ No users found. Please run generate_users.py first.")
        input("\nPress Enter to continue...")
        return

    if not runner.load_model():
        print("❌ Failed to load model. Please run training first.")
        input("\nPress Enter to continue...")
        return

    runner.story_session_authenticated()

    input("\nPress Enter to continue...")


def generate_users():
    """Generate users file."""
    log_operation_status("User generation")

    from generate_users import generate_users

    print("\n👥 Generating users file...")
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
        "logs"
    ]

    for directory in directories:
        ensure_dir_exists(directory)


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
    """Main entry point with authentication."""
    setup_directories()
    display_welcome()


    config_file = os.path.join(cwd, "configs", "model_config.yaml")
    if not os.path.exists(config_file):
        print("\n⚠️ Configuration file not found!")
        print("Please ensure configs/model_config.yaml exists.")
        return

    user_info = authenticate_user()
    if not user_info:
        print("❌ Authentication failed")
        return

    if user_info['admin']:
        print("\n🔄 ADMIN WORKFLOW:")
        print("1. Check Environment → 2. Download Data → 8. Generate Users →")
        print("3. Process Data → 4. Tokenize → 5. Train → 6. Test Samples → 7. Interactive")
        admin_menu()
    else:
        print(f"\n🎭 Welcome to Interactive Story Creation!")
        interactive_stories()


if __name__ == "__main__":
    main()