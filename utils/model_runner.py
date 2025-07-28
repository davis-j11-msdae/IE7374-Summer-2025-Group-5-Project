import torch
import pandas as pd
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional
from helpers import (
    load_config, get_age_group, log_operation_status, set_cwd)
from history import StoryHistoryManager
from eval import evaluate_single_text, load_evaluation_models

cwd=set_cwd()
sys.path.append(os.path.join(cwd, 'utils'))

class StoryModelRunner:
    """Runs the fine-tuned storytelling model with user authentication and history."""

    def __init__(self):
        self.config = load_config()
        self.model = None
        self.tokenizer = None
        self.history_manager = StoryHistoryManager()
        self.evaluation_models = None
        self.users_df = self._load_users()

    def _load_users(self) -> pd.DataFrame:
        """Load user authentication data."""
        users_path = os.path.join(self.config['paths']['users'], "users.txt")

        if not os.path.exists(users_path):
            print(f"Users file not found at {users_path}")
            print("Please run generate_users.py first.")
            return pd.DataFrame()

        return pd.read_csv(users_path)

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials."""
        if self.users_df.empty:
            return None

        user_row = self.users_df[self.users_df['username'] == username]

        if user_row.empty:
            return None

        if user_row.iloc[0]['password'] != password:
            return None

        return {
            'username': username,
            'age': int(user_row.iloc[0]['age']),
            'age_group': get_age_group(int(user_row.iloc[0]['age'])),
            'admin': int(user_row.iloc[0].get('admin', 0))
        }

    def load_model(self) -> bool:
        """Load the fine-tuned storytelling model."""
        models_path = self.config['paths']['models']
        model_path = os.path.join(models_path, "tuned_story_llm")

        if not os.path.exists(model_path):
            print(f"Fine-tuned model not found at {model_path}")
            print("Please run train.py first.")
            return False

        log_operation_status("Loading fine-tuned Mistral 7B model")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        try:
            # First try loading without device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            # Move to appropriate device
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if gpu_memory_gb >= 10:  # If we have enough VRAM, try GPU
                    try:
                        self.model = self.model.to('cuda')
                        print("Model loaded on GPU")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print("GPU out of memory, keeping model on CPU")
                            self.model = self.model.to('cpu')
                        else:
                            raise e
                else:
                    self.model = self.model.to('cpu')
                    print("Model loaded on CPU due to limited VRAM")
            else:
                self.model = self.model.to('cpu')
                print("Model loaded on CPU (no CUDA available)")

            self.model.eval()
            print("Model loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def load_evaluation_models(self):
        """Load models for story evaluation."""
        if self.evaluation_models is None:
            log_operation_status("Loading evaluation models")
            self.evaluation_models = load_evaluation_models()
            print("Evaluation models loaded")

    def format_story_prompt(self, prompt: str, age: int, history_context: str = None) -> str:
        """Format prompt with age-appropriate instructions using Mistral's chat format."""
        age_group = get_age_group(age)

        age_instructions = {
            'child': 'Write a simple, wholesome story for young children with easy words and short sentences.',
            'kid': 'Write an engaging story for children with age-appropriate vocabulary and themes.',
            'teen': 'Write a compelling story for teenagers with more complex vocabulary and themes.',
            'adult': 'Write a sophisticated story for adults with mature themes and advanced vocabulary.'
        }

        instruction = age_instructions[age_group]

        # Use Mistral's chat format
        if history_context:
            system_msg = f"{instruction}\n\nPrevious story context: {history_context}"
        else:
            system_msg = instruction

        # Mistral chat template format
        formatted_prompt = f"<s>[INST] {system_msg}\n\nPrompt: {prompt}\n\nPlease write a story based on this prompt. [/INST]"

        return formatted_prompt

    def generate_story(self, prompt: str, user_info: Dict[str, Any],
                       history_context: str = None, max_length: int = 512) -> str:
        """Generate a story using the fine-tuned model."""
        formatted_prompt = self.format_story_prompt(
            prompt, user_info['age'], history_context
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # Increased for Mistral's longer context
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=self.config['model']['temperature'],
                top_p=self.config['model']['top_p'],
                top_k=self.config['model']['top_k'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1  # Help prevent repetition
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def evaluate_story(self, story: str, user_age: int) -> Dict[str, Any]:
        """Evaluate generated story for quality and appropriateness."""
        if self.evaluation_models is None:
            self.load_evaluation_models()

        return evaluate_single_text(story, user_age, self.evaluation_models)

    def check_story_appropriateness(self, story: str, user_age: int) -> Dict[str, Any]:
        """Check if story is appropriate for user's age and non-toxic."""
        evaluation = self.evaluate_story(story, user_age)

        user_age_group = get_age_group(user_age)
        predicted_age_group = evaluation['predicted_age_group']

        age_hierarchy = {'child': 0, 'kid': 1, 'teen': 2, 'adult': 3}

        is_age_appropriate = (
                age_hierarchy[predicted_age_group] <= age_hierarchy[user_age_group]
        )

        return {
            'is_appropriate': is_age_appropriate and not evaluation['is_toxic'],
            'is_toxic': evaluation['is_toxic'],
            'is_age_appropriate': is_age_appropriate,
            'predicted_age_group': predicted_age_group,
            'user_age_group': user_age_group,
            'evaluation': evaluation
        }

    def generate_appropriate_story(self, prompt: str, user_info: Dict[str, Any],
                                   history_context: str = None, max_attempts: int = 3) -> Dict[str, Any]:
        """Generate story with appropriateness checking and retry logic."""
        log_operation_status(f"Generating story for {user_info['username']}")

        for attempt in range(max_attempts):

            story = self.generate_story(prompt, user_info, history_context)

            if not story.strip():
                continue

            appropriateness = self.check_story_appropriateness(story, user_info['age'])

            if appropriateness['is_appropriate']:
                return {
                    'success': True,
                    'story': story,
                    'evaluation': appropriateness['evaluation'],
                    'attempts': attempt + 1
                }

            if appropriateness['is_toxic']:
                prompt = f"Write a completely wholesome, non-toxic story: {prompt}"

            if not appropriateness['is_age_appropriate']:
                age_group = user_info['age_group']
                prompt = f"Write a simple story appropriate for {age_group} readers: {prompt}"

        return {
            'success': False,
            'error': 'Could not generate appropriate story after maximum attempts',
            'story': None,
            'attempts': max_attempts
        }

    def interactive_story_session(self):
        """Run interactive story generation session with authentication."""
        print("\nPERSONALIZED STORYTELLING SYSTEM (Mistral 7B)")
        print("=" * 50)

        username = input("Username: ").strip()
        password = input("Password: ").strip()

        user_info = self.authenticate_user(username, password)
        if not user_info:
            print("Invalid credentials")
            return

        print(f"Welcome {username}! (Age: {user_info['age']}, Group: {user_info['age_group']})")

        if self.model is None:
            if not self.load_model():
                return

        self._run_story_session(user_info)

    def story_session_authenticated(self):
        """Run story session for already authenticated user (from main menu)."""
        print("\nEnter your username to continue:")
        username = input("Username: ").strip()

        user_row = self.users_df[self.users_df['username'] == username]
        if user_row.empty:
            print("User not found")
            return

        user_data = user_row.iloc[0]
        user_info = {
            'username': username,
            'age': int(user_data['age']),
            'age_group': get_age_group(int(user_data['age'])),
            'admin': int(user_data.get('admin', 0))
        }

        print(f"Starting story session for {username}! (Age: {user_info['age']}, Group: {user_info['age_group']})")

        if self.model is None:
            if not self.load_model():
                return

        self._run_story_session(user_info)

    def _run_story_session(self, user_info: Dict[str, Any]):
        """Run the actual story session."""
        while True:
            print(f"\nStory Options:")
            print("1. Create new story")
            print("2. Continue existing story")
            print("3. View story history")
            print("4. Delete story")
            print("5. Exit")

            choice = input("\nSelect option (1-5): ").strip()

            if choice == "1":
                self._create_new_story(user_info)
            elif choice == "2":
                self._continue_existing_story(user_info)
            elif choice == "3":
                self._view_story_history(user_info)
            elif choice == "4":
                self._delete_story(user_info)
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

    def _create_new_story(self, user_info: Dict[str, Any]):
        """Create a new story interactively."""
        prompt = input("\nEnter your story prompt: ").strip()
        if not prompt:
            print("Please enter a valid prompt")
            return

        print("Generating story...")

        result = self.generate_appropriate_story(prompt, user_info)

        if result['success']:
            story = result['story']
            evaluation = result['evaluation']

            print(f"\nGenerated Story:")
            print("=" * 50)
            print(story)
            print("=" * 50)

            print(f"\nStory Statistics:")
            print(f"  Length: {len(story)} characters")
            print(f"  Reading Level: {evaluation.get('flesch_kincaid_score', 'N/A')}")
            print(f"  Predicted Age Group: {evaluation.get('predicted_age_group', 'N/A')}")
            print(f"  Quality Scores: Grammar {evaluation.get('grammar_score', 0):.1f}/100, Coherence {evaluation.get('coherence_score', 0):.1f}/100")

            save_response = input("\nSave this story to your history? (y/N): ").strip().lower()

            if save_response == 'y':
                success = self.history_manager.save_story(
                    username=user_info['username'],
                    story=story,
                    age=user_info['age'],
                    prompt=prompt
                )

                if not success:
                    print("Failed to save story")
        else:
            print(f"Story generation failed: {result['error']}")
            print("Please try a different prompt.")

    def _continue_existing_story(self, user_info: Dict[str, Any]):
        """Continue an existing story."""
        titles = self.history_manager.get_user_titles(user_info['username'])

        if not titles:
            print("No stories found in your history")
            return

        print(f"\nYour Stories:")
        for i, title in enumerate(titles, 1):
            print(f"  {i}. {title}")

        try:
            choice = int(input(f"\nSelect story to continue (1-{len(titles)}): "))

            if 1 <= choice <= len(titles):
                selected_title = titles[choice - 1]

                story_entry = self.history_manager.get_story_by_title(
                    user_info['username'], selected_title
                )

                print(f"\nStory Summary: {story_entry['summary']}")

                continuation_prompt = input("\nHow should the story continue? ").strip()

                if not continuation_prompt:
                    print("Please enter a valid continuation prompt")
                    return

                print("Generating continuation...")

                result = self.generate_appropriate_story(
                    continuation_prompt,
                    user_info,
                    history_context=story_entry['summary']
                )

                if result['success']:
                    new_content = result['story']

                    print(f"\nStory Continuation:")
                    print("=" * 50)
                    print(new_content)
                    print("=" * 50)

                    print(f"\nSave Options:")
                    print("1. Update original story")
                    print("2. Save as new story")
                    print("3. Don't save")

                    save_choice = input("Select option (1-3): ").strip()

                    if save_choice == "1":
                        success = self.history_manager.continue_story(
                            user_info['username'],
                            selected_title,
                            new_content,
                            save_as_new=False
                        )
                    elif save_choice == "2":
                        success = self.history_manager.continue_story(
                            user_info['username'],
                            selected_title,
                            new_content,
                            save_as_new=True
                        )
                    else:
                        success = True

                    if not success:
                        print("Failed to save continuation")
                else:
                    print(f"Continuation generation failed: {result['error']}")
            else:
                print("Invalid selection")

        except ValueError:
            print("Please enter a valid number")

    def _view_story_history(self, user_info: Dict[str, Any]):
        """View user's story history."""
        titles = self.history_manager.get_user_titles(user_info['username'])

        if not titles:
            print("No stories found in your history")
            return

        print(f"\nYour Story History:")

        for i, title in enumerate(titles, 1):
            story_entry = self.history_manager.get_story_by_title(
                user_info['username'], title
            )

            print(f"\n{i}. {title}")
            print(f"   Created: {story_entry['timestamp'][:10]}")
            print(f"   Length: {story_entry['word_count']} words")
            print(f"   Summary: {story_entry['summary']}")

        stats = self.history_manager.get_user_statistics(user_info['username'])
        print(f"\nYour Statistics:")
        print(f"  Total Stories: {stats['total_stories']}")
        print(f"  Total Words: {stats['total_words']:,}")
        print(f"  Average Length: {stats['average_story_length']:.0f} words")

    def _delete_story(self, user_info: Dict[str, Any]):
        """Delete a story from history."""
        titles = self.history_manager.get_user_titles(user_info['username'])

        if not titles:
            print("No stories found in your history")
            return

        print(f"\nYour Stories:")
        for i, title in enumerate(titles, 1):
            print(f"  {i}. {title}")

        try:
            choice = int(input(f"\nSelect story to delete (1-{len(titles)}): "))

            if 1 <= choice <= len(titles):
                selected_title = titles[choice - 1]

                confirm = input(f"Delete '{selected_title}'? This cannot be undone. (y/N): ").strip().lower()

                if confirm == 'y':
                    success = self.history_manager.delete_story(
                        user_info['username'], selected_title
                    )

                    if not success:
                        print("Failed to delete story")
                else:
                    print("Deletion cancelled")
            else:
                print("Invalid selection")

        except ValueError:
            print("Please enter a valid number")


def main():
    """Main function to run the story generation system."""
    runner = StoryModelRunner()

    if runner.users_df.empty:
        print("No users found. Please run generate_users.py first.")
        return

    runner.interactive_story_session()


if __name__ == "__main__":
    main()