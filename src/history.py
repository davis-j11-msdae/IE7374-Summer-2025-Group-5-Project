import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import pipeline
from detoxify import Detoxify
from utils.helpers import (
    load_config, ensure_dir_exists, save_json, load_json,
    log_operation_status
)


class StoryHistoryManager:
    """Manages user story history with summarization and toxicity checking."""

    def __init__(self):
        self.config = load_config()
        self.history_config = self.config['history']
        self.paths = self.config['paths']

        # Initialize summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model=self.history_config['summary_model'],
            tokenizer=self.history_config['summary_model']
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

    def generate_summary(self, story: str) -> str:
        """Generate a summary of the story."""
        try:
            # Ensure story is long enough for summarization
            if len(story.split()) < 50:
                return story[:self.history_config['max_summary_length']]

            # Generate summary
            summary_result = self.summarizer(
                story,
                max_length=self.history_config['max_summary_length'],
                min_length=30,
                do_sample=False
            )

            summary = summary_result[0]['summary_text']

            # Clean up summary
            summary = summary.strip()
            if not summary.endswith('.'):
                summary += '.'

            return summary

        except Exception as e:
            print(f"Error generating summary: {e}")
            # Fallback: return first part of story
            words = story.split()
            max_words = min(30, len(words))
            return ' '.join(words[:max_words]) + '...'

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
        """Suggest titles based on story summary."""
        # Extract key words and phrases
        words = story_summary.split()

        # Simple title suggestions based on content
        suggestions = []

        # Look for character names (capitalized words)
        characters = [word.strip('.,!?') for word in words if word[0].isupper() and len(word) > 2]

        # Look for key themes
        themes = []
        theme_keywords = {
            'adventure': ['journey', 'quest', 'travel', 'explore'],
            'magic': ['magic', 'magical', 'wizard', 'spell'],
            'friendship': ['friend', 'friends', 'together'],
            'mystery': ['mystery', 'secret', 'hidden'],
            'space': ['space', 'planet', 'star', 'galaxy']
        }

        for theme, keywords in theme_keywords.items():
            if any(keyword in story_summary.lower() for keyword in keywords):
                themes.append(theme)

        # Generate suggestions
        if characters:
            suggestions.append(f"The Tale of {characters[0]}")
            if len(characters) > 1:
                suggestions.append(f"{characters[0]} and {characters[1]}")

        if themes:
            suggestions.append(f"A {themes[0].title()} Story")
            suggestions.append(f"The Great {themes[0].title()}")

        # Generic suggestions
        suggestions.extend([
            "An Amazing Adventure",
            "A Wonderful Tale",
            "The Journey Begins"
        ])

        # Validate and return non-toxic titles
        valid_suggestions = []
        for title in suggestions[:5]:  # Limit to 5 suggestions
            if self.validate_title(title):
                valid_suggestions.append(title)

        return valid_suggestions

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
            # Suggest titles
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
            print("❌ Title contains inappropriate content or is too long.")
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
            print(f"✅ Story saved as '{title}'")

        return success

    def continue_story(self, username: str, title: str, new_content: str,
                       save_as_new: bool = False, new_title: str = None) -> bool:
        """Continue an existing story."""
        existing_story = self.get_story_by_title(username, title)

        if not existing_story:
            print(f"❌ Story '{title}' not found.")
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
                print(f"✅ Story '{title}' updated with continuation")

            return success

    def delete_story(self, username: str, title: str) -> bool:
        """Delete a story from user history."""
        history = self.load_user_history(username)
        original_length = len(history)

        history = [entry for entry in history if entry['title'] != title]

        if len(history) == original_length:
            print(f"❌ Story '{title}' not found.")
            return False

        success = self.save_user_history(username, history)

        if success:
            print(f"✅ Story '{title}' deleted")

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