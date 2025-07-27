import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from detoxify import Detoxify
from utils.helpers import (
    load_config, ensure_dir_exists, save_json, load_json,
    log_operation_status, set_cwd)

# Get current working directory for path operations
cwd = set_cwd()


class MistralSummarizer:
    """Summarizer using Mistral model for story summarization."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.history_config = config['history']
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load Mistral model for summarization."""
        print("Loading Mistral model for summarization...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print("Mistral summarizer loaded")

    def generate_summary(self, story: str) -> str:
        """Generate a summary using Mistral model."""
        # Ensure story is long enough for summarization
        if len(story.split()) < 50:
            return story[:self.history_config['max_summary_length']]

        prompt = f"""Summarize the following story in 2-3 sentences. Focus on the main plot points and characters. Keep it concise and engaging.

Story: {story}

Summary:"""

        try:
            # Format using Mistral's chat template
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"

            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536  # Leave room for summary
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.history_config['mistral_summary_max_tokens'],
                    temperature=self.history_config['mistral_summary_temperature'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            summary = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Clean up summary
            summary = summary.strip()
            if not summary.endswith('.'):
                summary += '.'

            # Truncate if too long
            if len(summary) > self.history_config['max_summary_length']:
                sentences = summary.split('.')
                summary = '. '.join(sentences[:-1]) + '.'
                if len(summary) > self.history_config['max_summary_length']:
                    summary = summary[:self.history_config['max_summary_length'] - 3] + '...'

            return summary

        except Exception as e:
            print(f"Summary generation failed: {e}")
            # Fallback: return first part of story
            words = story.split()
            max_words = min(30, len(words))
            return ' '.join(words[:max_words]) + '...'


class StoryHistoryManager:
    """Manages user story history with Mistral-based summarization and toxicity checking."""

    def __init__(self):
        self.config = load_config()
        self.history_config = self.config['history']
        self.paths = self.config['paths']

        # Determine model path for summarization
        models_path = self.config['paths']['models']
        tuned_model_path = os.path.join(models_path, "tuned_story_llm")
        base_model_path = os.path.join(models_path, "mistral-7b-base")

        if os.path.exists(tuned_model_path):
            model_path = tuned_model_path
        elif os.path.exists(base_model_path):
            model_path = base_model_path
        else:
            model_path = self.config['model']['base_model']

        # Initialize Mistral summarizer
        self.summarizer = MistralSummarizer(model_path, self.config)

        # Initialize toxicity detector for title validation
        self.detoxify = Detoxify('original')

        # Ensure history directory exists
        ensure_dir_exists(self.paths['user_history'])

    def _get_user_history_path(self, username: str) -> Path:
        """Get the file path for user's history."""
        return Path(self.paths['user_history']) / f"{username}_history.json"

    def load_user_history(self, username: str) -> List[Dict[str, Any]]:
        """Load user's story history."""
        history_path = self._get_user_history_path(username)

        if not history_path.exists():
            return []

        try:
            return load_json(history_path)
        except Exception as e:
            print(f"Error loading history for {username}: {e}")
            return []

    def save_user_history(self, username: str, history: List[Dict[str, Any]]) -> bool:
        """Save user's story history."""
        history_path = self._get_user_history_path(username)

        try:
            save_json(history, history_path)
            return True
        except Exception as e:
            print(f"Error saving history for {username}: {e}")
            return False

    def generate_summary(self, story: str) -> str:
        """Generate a summary of the story using Mistral."""
        return self.summarizer.generate_summary(story)

    def validate_title(self, title: str) -> bool:
        """Validate that title is not toxic."""
        try:
            # Check length
            if len(title) > self.history_config['title_max_length']:
                return False

            # Check toxicity
            results = self.detoxify.predict(title)
            toxicity_score = results['toxicity']

            return toxicity_score < self.config['evaluation']['toxicity_threshold']

        except Exception as e:
            print(f"Error validating title: {e}")
            return False

    def suggest_titles(self, story_summary: str) -> List[str]:
        """Suggest titles based on story summary using Mistral."""
        prompt = f"""Based on this story summary, suggest 3 short, creative titles (each under 50 characters). Respond with only the titles, one per line:

Summary: {story_summary}

Titles:"""

        try:
            # Format using Mistral's chat template
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"

            inputs = self.summarizer.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.summarizer.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    temperature=0.7,  # Slightly higher for creativity
                    do_sample=True,
                    pad_token_id=self.summarizer.tokenizer.eos_token_id,
                    eos_token_id=self.summarizer.tokenizer.eos_token_id
                )

            response = self.summarizer.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Extract titles from response
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            suggested_titles = []

            for line in lines[:3]:  # Take first 3 lines
                # Clean up the title (remove numbers, quotes, etc.)
                title = re.sub(r'^\d+\.?\s*', '', line)  # Remove leading numbers
                title = title.strip('"\'')  # Remove quotes
                title = title[:self.history_config['title_max_length']]  # Truncate if needed

                if title and self.validate_title(title):
                    suggested_titles.append(title)

            # Add fallback titles if needed
            if len(suggested_titles) < 3:
                fallback_titles = ["An Amazing Adventure", "A Wonderful Tale", "The Journey Begins"]
                for fallback in fallback_titles:
                    if len(suggested_titles) < 3 and self.validate_title(fallback):
                        suggested_titles.append(fallback)

            return suggested_titles

        except Exception as e:
            print(f"Title suggestion failed: {e}")
            # Fallback to simple extraction
            words = story_summary.split()
            if len(words) > 3:
                return ["An Amazing Adventure", "A Wonderful Tale", "The Journey Begins"]
            return ["My Story"]

    def get_user_titles(self, username: str) -> List[str]:
        """Get list of story titles for a user."""
        history = self.load_user_history(username)
        return [entry['title'] for entry in history]

    def get_story_by_title(self, username: str, title: str) -> Optional[Dict[str, Any]]:
        """Get a specific story by title."""
        history = self.load_user_history(username)

        for entry in history:
            if entry['title'] == title:
                return entry

        return None

    def save_story(self, username: str, story: str, title: str = None,
                   age: int = None, prompt: str = "") -> bool:
        """Save a story to user history."""
        log_operation_status(f"Saving story for {username}")

        # Generate summary
        summary = self.generate_summary(story)

        # Handle title
        if title is None:
            # Suggest titles using Mistral
            suggestions = self.suggest_titles(summary)

            if suggestions:
                print("\nSuggested titles:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")

                print(f"  {len(suggestions) + 1}. Enter custom title")

                try:
                    choice = int(input(f"\nSelect title (1-{len(suggestions) + 1}): "))

                    if 1 <= choice <= len(suggestions):
                        title = suggestions[choice - 1]
                    else:
                        title = input("Enter custom title: ").strip()
                except ValueError:
                    title = input("Enter custom title: ").strip()
            else:
                title = input("Enter title for this story: ").strip()

        # Validate title
        if not self.validate_title(title):
            print("Title contains inappropriate content or is too long.")
            return False

        # Check if title already exists
        existing_titles = self.get_user_titles(username)
        if title in existing_titles:
            response = input(f"Title '{title}' already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                return False

            # Remove existing entry
            history = self.load_user_history(username)
            history = [entry for entry in history if entry['title'] != title]
        else:
            history = self.load_user_history(username)

        # Create new entry
        entry = {
            'title': title,
            'summary': summary,
            'full_story': story,
            'timestamp': datetime.now().isoformat(),
            'age': age,
            'prompt': prompt,
            'word_count': len(story.split()),
            'character_count': len(story)
        }

        # Add to history
        history.append(entry)

        # Maintain maximum history length
        max_length = self.history_config['max_history_length']
        if len(history) > max_length:
            history = history[-max_length:]  # Keep most recent

        # Save updated history
        success = self.save_user_history(username, history)

        if success:
            print(f"Story saved as '{title}'")

        return success

    def continue_story(self, username: str, title: str, new_content: str,
                       save_as_new: bool = False, new_title: str = None) -> bool:
        """Continue an existing story."""
        existing_story = self.get_story_by_title(username, title)

        if not existing_story:
            print(f"Story '{title}' not found.")
            return False

        # Combine stories
        combined_story = existing_story['full_story'] + "\n\n" + new_content
        combined_summary = self.generate_summary(combined_story)

        if save_as_new:
            # Save as new story
            if new_title is None:
                new_title = input("Enter title for continued story: ").strip()

            return self.save_story(
                username=username,
                story=combined_story,
                title=new_title,
                age=existing_story.get('age'),
                prompt=f"Continuation of '{title}'"
            )
        else:
            # Update existing story
            history = self.load_user_history(username)

            for entry in history:
                if entry['title'] == title:
                    entry['full_story'] = combined_story
                    entry['summary'] = combined_summary
                    entry['timestamp'] = datetime.now().isoformat()
                    entry['word_count'] = len(combined_story.split())
                    entry['character_count'] = len(combined_story)
                    break

            success = self.save_user_history(username, history)

            if success:
                print(f"Story '{title}' updated with continuation")

            return success

    def delete_story(self, username: str, title: str) -> bool:
        """Delete a story from user history."""
        history = self.load_user_history(username)
        original_length = len(history)

        history = [entry for entry in history if entry['title'] != title]

        if len(history) == original_length:
            print(f"Story '{title}' not found.")
            return False

        success = self.save_user_history(username, history)

        if success:
            print(f"Story '{title}' deleted")

        return success

    def get_user_statistics(self, username: str) -> Dict[str, Any]:
        """Get statistics about user's story history."""
        history = self.load_user_history(username)

        if not history:
            return {
                'total_stories': 0,
                'total_words': 0,
                'average_story_length': 0,
                'first_story_date': None,
                'last_story_date': None
            }

        total_stories = len(history)
        total_words = sum(entry['word_count'] for entry in history)
        average_length = total_words / total_stories

        dates = [entry['timestamp'] for entry in history]
        first_date = min(dates)
        last_date = max(dates)

        return {
            'total_stories': total_stories,
            'total_words': total_words,
            'average_story_length': average_length,
            'first_story_date': first_date,
            'last_story_date': last_date
        }


def main():
    """Demo function for history manager."""
    history_manager = StoryHistoryManager()

    # Test with demo user
    demo_user = "demo_user"
    demo_story = """
    Once upon a time, in a magical forest filled with talking animals, 
    there lived a young fox named Felix. Felix was known throughout the 
    forest for his curiosity and kind heart. One day, while exploring 
    a part of the forest he had never seen before, Felix discovered a 
    hidden cave with glowing crystals.
    """

    # Save story
    success = history_manager.save_story(
        username=demo_user,
        story=demo_story,
        age=8,
        prompt="A magical adventure in a forest"
    )

    if success:
        # Show user titles
        titles = history_manager.get_user_titles(demo_user)
        print(f"\nUser stories: {titles}")

        # Show statistics
        stats = history_manager.get_user_statistics(demo_user)
        print(f"\nUser statistics: {stats}")


if __name__ == "__main__":
    main()