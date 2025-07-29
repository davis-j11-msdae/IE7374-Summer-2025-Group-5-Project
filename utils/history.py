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


# Fix for MistralSummarizer to use lazy loading
class MistralSummarizer:
    """Summarizer using shared Mistral model for story summarization."""

    def __init__(self, shared_model=None, shared_tokenizer=None, config=None):
        self.shared_model = shared_model
        self.shared_tokenizer = shared_tokenizer
        self.config = config
        self.history_config = config['history'] if config else {}
        print("Mistral summarizer initialized (using shared model)")

    def _ensure_model_loaded(self):
        """Ensure shared model is available."""
        if self.shared_model is None or self.shared_tokenizer is None:
            raise RuntimeError("Shared model not provided to summarizer")

    def generate_summary(self, story: str, original_prompt: str = None) -> str:
        """Generate a summary using shared Mistral model, including original prompt."""
        self._ensure_model_loaded()

        # Ensure story is long enough for summarization
        if len(story.split()) < 50:
            summary_text = story[:self.history_config['max_summary_length']]
            if original_prompt:
                return f"Prompt: {original_prompt}. {summary_text}"
            return summary_text

        # Include prompt in summarization request
        if original_prompt:
            prompt = f"""Summarize the following story in 2-3 sentences. Include the original prompt at the beginning. Focus on the main plot points and characters. Keep it concise and engaging.

    Original Prompt: {original_prompt}

    Story: {story}

    Summary (include the prompt):"""
        else:
            prompt = f"""Summarize the following story in 2-3 sentences. Focus on the main plot points and characters. Keep it concise and engaging.

    Story: {story}

    Summary:"""

        try:
            # Format using Mistral's chat template
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"

            inputs = self.shared_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536  # Leave room for summary
            )

            # Move inputs to same device as model
            if next(self.shared_model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.shared_model.generate(
                    **inputs,
                    max_new_tokens=self.history_config['mistral_summary_max_tokens'],
                    temperature=self.history_config['mistral_summary_temperature'],
                    do_sample=True,
                    pad_token_id=self.shared_tokenizer.eos_token_id,
                    eos_token_id=self.shared_tokenizer.eos_token_id
                )

            summary = self.shared_tokenizer.decode(
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
            # Fallback: return first part of story with prompt if available
            words = story.split()
            max_words = min(30, len(words))
            fallback_summary = ' '.join(words[:max_words]) + '...'

            if original_prompt:
                return f"Prompt: {original_prompt}. {fallback_summary}"
            return fallback_summary


class StoryHistoryManager:
    def __init__(self, shared_model=None, shared_tokenizer=None):
        self.config = load_config()
        self.history_config = self.config['history']
        self.paths = self.config['paths']

        # Initialize Mistral summarizer
        self.summarizer = MistralSummarizer(
            shared_model=shared_model,
            shared_tokenizer=shared_tokenizer,
            config=self.config
        )

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

    def generate_summary(self, story: str, original_prompt: str = None) -> str:
        """Generate a summary of the story using Mistral, including original prompt."""
        return self.summarizer.generate_summary(story, original_prompt)

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
        # Extract key words from summary for simple title generation
        summary_words = story_summary.split()[:20]  # First 20 words only
        key_summary = ' '.join(summary_words)

        prompt = f"""Given this story summary, suggest ONE short title (maximum 5 words):

    Summary: {key_summary}

    Title:"""

        suggested_titles = []

        # Try to generate 3 titles with separate calls
        for attempt in range(3):
            try:
                self.summarizer._ensure_model_loaded()

                formatted_prompt = f"<s>[INST] {prompt} [/INST]"

                inputs = self.summarizer.shared_tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512  # Shorter context
                )

                if next(self.summarizer.shared_model.parameters()).is_cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.summarizer.shared_model.generate(
                        **inputs,
                        max_new_tokens=10,  # Very limited tokens
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.summarizer.shared_tokenizer.eos_token_id,
                        eos_token_id=self.summarizer.shared_tokenizer.eos_token_id,
                        repetition_penalty=1.2
                    )

                response = self.summarizer.shared_tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                # Aggressive cleaning of the response
                title = response.split('.')[0].split('\n')[0].strip()

                # Remove markdown formatting and special characters
                title = re.sub(r'\*+', '', title)  # Remove asterisks
                title = re.sub(r'_+', '', title)  # Remove underscores
                title = re.sub(r'#+', '', title)  # Remove hash symbols
                title = re.sub(r'\[.*?\]', '', title)  # Remove brackets
                title = re.sub(r'\(.*?\)', '', title)  # Remove parentheses
                title = re.sub(r'[^\w\s\-\'"]', ' ', title)  # Replace special chars with spaces

                # Clean up multiple spaces and trim
                title = re.sub(r'\s+', ' ', title).strip()
                title = title.strip('"\'')

                # Remove common prefixes/suffixes that might appear
                prefixes_to_remove = ['title:', 'story:', 'the story of', 'a tale of']
                for prefix in prefixes_to_remove:
                    if title.lower().startswith(prefix):
                        title = title[len(prefix):].strip()

                # Limit to first 5 words maximum
                words = title.split()
                if len(words) > 5:
                    title = ' '.join(words[:5])

                # Final cleanup - ensure proper capitalization
                if title:
                    title = ' '.join(word.capitalize() for word in title.split())

                # Check if it's reasonable length and validate
                if (title and
                        5 <= len(title) <= self.history_config['title_max_length'] and
                        self.validate_title(title) and
                        title not in suggested_titles):
                    suggested_titles.append(title)

            except Exception as e:
                print(f"Title generation attempt {attempt + 1} failed: {e}")
                continue

        # Fill with safe fallbacks if needed
        safe_fallbacks = [
            "A Wonderful Tale",
            "An Amazing Adventure",
            "The Great Story",
            "A Magical Journey",
            "The Special Day"
        ]

        for fallback in safe_fallbacks:
            if len(suggested_titles) >= 3:
                break
            if fallback not in suggested_titles:
                suggested_titles.append(fallback)

        return suggested_titles[:3]

    def _ensure_model_loaded(self):
        """Ensure shared model is available."""
        if self.shared_model is None or self.shared_tokenizer is None:
            raise RuntimeError("Shared model not provided to summarizer")

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

        # Generate summary including the original prompt
        summary = self.generate_summary(story, prompt)

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

        # Use the original prompt for summary generation
        original_prompt = existing_story.get('prompt', '')
        combined_summary = self.generate_summary(combined_story, original_prompt)

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