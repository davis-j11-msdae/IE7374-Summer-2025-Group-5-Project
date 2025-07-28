from pathlib import Path
from typing import Dict, List, Any
from utils.helpers import (
    load_config, ensure_dir_exists, save_json,
    log_operation_status, set_cwd)
from utils.model_runner import StoryModelRunner
# from utils.history import StoryHistoryManager

# Get current working directory for path operations
cwd = set_cwd()

# Sample prompts for different age groups
SAMPLE_PROMPTS = {
    'child_1': {
        'age': 4,
        'prompts': [
            "A friendly puppy finds a lost toy in the park"
        ]
    },
    'child_2': {
        'age': 5,
        'prompts': [
            "A colorful butterfly visits different flowers"
        ]
    },
    'kid_1': {
        'age': 8,
        'prompts': [
            "A brave mouse goes on a quest to find magical cheese"
        ]
    },
    'kid_2': {
        'age': 10,
        'prompts': [
            "Children discover a secret garden behind their school"
        ]
    },
    'teen_1': {
        'age': 15,
        'prompts': [
            "A teenager discovers they can communicate with animals"
        ]
    },
    'teen_2': {
        'age': 16,
        'prompts': [
            "High school students start a band and face their first competition"
        ]
    },
    'adult_1': {
        'age': 25,
        'prompts': [
            "A young detective solves their first mysterious case"
        ]
    },
    'adult_2': {
        'age': 35,
        'prompts': [
            "An astronaut discovers an ancient civilization on Mars"
        ]
    }
    # ,
    # # Users with history continuation
    # 'child_3': {
    #     'age': 6,
    #     'prompts': [
    #         "A little cat explores a big house",
    #         "The cat finds a magical door that leads to a candy kingdom"  # Continuation
    #     ]
    # },
    # 'adult_3': {
    #     'age': 42,
    #     'prompts': [
    #         "A marine biologist discovers a new underwater species",
    #         "The scientist returns to study the creature's advanced society"  # Continuation
    #     ]
    # }
}


class SampleProcessor:
    """Processes sample prompts and generates evaluation results using Mistral 7B."""

    def __init__(self):
        self.config = load_config()
        self.model_runner = StoryModelRunner()
        # self.history_manager = StoryHistoryManager()
        self.results = []

        # Ensure samples directory exists
        ensure_dir_exists(self.config['paths']['samples'])

    def setup_sample_users(self):
        """Create temporary authentication for sample users."""
        import pandas as pd

        # Create sample user data
        sample_users = []
        for username, user_data in SAMPLE_PROMPTS.items():
            sample_users.append({
                'username': username,
                'age': user_data['age'],
                'password': 'test'
            })

        # Temporarily add to users dataframe
        sample_df = pd.DataFrame(sample_users)
        self.model_runner.users_df = pd.concat([
            self.model_runner.users_df,
            sample_df
        ], ignore_index=True)

    def process_single_sample(self, username: str, user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process all prompts for a single sample user using Mistral 7B."""
        log_operation_status(f"Processing samples for {username}")

        # Import get_age_group from helpers
        from helpers import get_age_group

        user_info = {
            'username': username,
            'age': user_data['age'],
            'age_group': get_age_group(user_data['age'])  # Use imported function
        }

        user_results = []

        for i, prompt in enumerate(user_data['prompts']):
            print(f"  Prompt {i + 1}: {prompt[:50]}...")

            # Determine if this is a continuation
            is_continuation = i > 0
            history_context = None

            if is_continuation:
                # Get previous story as context
                temp_val=0
                # titles = self.history_manager.get_user_titles(username)
                # if titles:
                #     previous_story = self.history_manager.get_story_by_title(username, titles[-1])
                #     history_context = previous_story['summary']

            # Generate story using Mistral 7B (without evaluation)
            story = self.model_runner.generate_story(
                prompt,
                user_info,
                history_context
            )

            sample_result = {
                'username': username,
                'age': user_data['age'],
                'prompt_number': i + 1,
                'prompt': prompt,
                'is_continuation': is_continuation,
                'model_used': 'mistral-7b-instruct-v0.3'
            }

            if story and story.strip() :
                print(f"    Generated ({len(story)} chars)")

                # Save to history with specific save behavior
                # if username == 'child_3' and is_continuation:
                #     # Update original story for child
                #     titles = self.history_manager.get_user_titles(username)
                #     if titles:
                #         self.history_manager.continue_story(
                #             username, titles[-1], story, save_as_new=False
                #         )
                # elif username == 'adult_3' and is_continuation:
                #     # Save as new story for adult
                #     self.history_manager.save_story(
                #         username, story,
                #         title=f"Continuation: {prompt[:30]}...",
                #         age=user_data['age'],
                #         prompt=prompt
                #     )
                # else:
                #     # Regular save
                #     self.history_manager.save_story(
                #         username, story,
                #         age=user_data['age'],
                #         prompt=prompt
                #     )

                sample_result.update({
                    'story': story,
                    'story_length': len(story),
                    'word_count': len(story.split()),
                    'saved_to_history': True,
                    'generation_success': True
                })
            else:
                print(f"    Failed: No story generated")
                sample_result.update({
                    'story': None,
                    'story_length': 0,
                    'word_count': 0,
                    'saved_to_history': False,
                    'generation_success': False,
                    'error': 'No story generated'
                })

            user_results.append(sample_result)

        return user_results

    def process_all_samples(self) -> Dict[str, Any]:
        """Process all sample prompts and generate stories using Mistral 7B."""
        log_operation_status("Sample story generation using Mistral 7B")

        # Load model
        if not self.model_runner.load_model():
            print("Failed to load Mistral 7B model")
            return {}

        # Setup sample users
        self.setup_sample_users()

        # Process each sample user (generation only)
        all_results = {}
        summary_stats = {
            'total_users': len(SAMPLE_PROMPTS),
            'total_prompts': sum(len(data['prompts']) for data in SAMPLE_PROMPTS.values()),
            'successful_generations': 0,
            'failed_generations': 0,
            'age_group_breakdown': {'child': 0, 'kid': 0, 'teen': 0, 'adult': 0},
            'continuation_stats': {'total_continuations': 0, 'successful_continuations': 0},
            'model_used': 'mistral-7b-instruct-v0.3'
        }

        for username, user_data in SAMPLE_PROMPTS.items():
            user_results = self.process_single_sample(username, user_data)
            all_results[username] = user_results

            # Update summary statistics (generation only)
            for result in user_results:
                if result['generation_success']:
                    summary_stats['successful_generations'] += 1

                    # # Age group stats
                    # age_group = self.model_runner.get_age_group(user_data['age'])
                    # summary_stats['age_group_breakdown'][age_group] += 1
                    #
                    # # Continuation stats
                    # if result['is_continuation']:
                    #     summary_stats['continuation_stats']['total_continuations'] += 1
                    #     summary_stats['continuation_stats']['successful_continuations'] += 1
                else:
                    summary_stats['failed_generations'] += 1

                    if result['is_continuation']:
                        summary_stats['continuation_stats']['total_continuations'] += 1

        # Compile final results without evaluation
        final_results = {
            'timestamp': self.get_timestamp(),
            'summary': summary_stats,
            'detailed_results': all_results,
            'sample_prompts_used': SAMPLE_PROMPTS,
            'model_info': {
                'name': 'mistral-7b-instruct-v0.3',
                'type': 'fine-tuned',
                'base_model': self.config['model']['base_model']
            },
            'evaluated': False  # Mark as not evaluated
        }

        log_operation_status("Sample story generation using Mistral 7B", "completed")
        return final_results

    def evaluate_samples(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate already generated samples."""
        log_operation_status("Evaluating generated samples")

        # Load evaluation models
        self.model_runner.load_evaluation_models()

        # Add evaluation metrics to existing results
        quality_scores = {'grammar': [], 'coherence': [], 'flesch_kincaid': []}
        safety_stats = {'safe_stories': 0, 'toxic_stories': 0}

        for username, user_results in results['detailed_results'].items():
            for result in user_results:
                if result['generation_success'] and result['story']:
                    story = result['story']
                    user_age = result['age']

                    print(f"  Evaluating {username} prompt {result['prompt_number']}...")

                    # Evaluate the story
                    evaluation = self.model_runner.evaluate_story(story, user_age)

                    # Add evaluation to result
                    result['evaluation'] = evaluation
                    result['evaluated'] = True

                    # Update quality scores
                    quality_scores['grammar'].append(evaluation.get('grammar_score', 75))
                    quality_scores['coherence'].append(evaluation.get('coherence_score', 75))
                    quality_scores['flesch_kincaid'].append(evaluation.get('flesch_kincaid_score', 8))

                    # Update safety stats
                    if evaluation.get('is_toxic', False):
                        safety_stats['toxic_stories'] += 1
                    else:
                        safety_stats['safe_stories'] += 1

                    print(
                        f"    Quality: Grammar {evaluation.get('grammar_score', 0):.1f}/100, Coherence {evaluation.get('coherence_score', 0):.1f}/100")
                    print(f"    Age Group: {evaluation.get('predicted_age_group', 'N/A')} (Target: {result['age']})")
                    print(f"    Safe: {'Yes' if not evaluation.get('is_toxic', True) else 'No'}")
                else:
                    result['evaluation'] = None
                    result['evaluated'] = False

        # Calculate averages and add to summary
        if quality_scores['grammar']:
            results['summary']['average_scores'] = {
                'grammar': sum(quality_scores['grammar']) / len(quality_scores['grammar']),
                'coherence': sum(quality_scores['coherence']) / len(quality_scores['coherence']),
                'flesch_kincaid': sum(quality_scores['flesch_kincaid']) / len(quality_scores['flesch_kincaid'])
            }

        results['summary']['quality_scores'] = quality_scores
        results['summary']['safety_stats'] = safety_stats
        results['evaluated'] = True

        log_operation_status("Sample evaluation", "completed")
        return results

    def get_timestamp(self) -> str:
        """Get current timestamp for results."""
        from datetime import datetime
        return datetime.now().isoformat()

    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to JSON file."""
        timestamp = self.get_timestamp().replace(':', '-').split('.')[0]
        results_file = Path(self.config['paths']['samples']) / f"mistral_results_{timestamp}.json"

        save_json(results, results_file)
        return str(results_file)

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted text report from results."""
        from helpers import get_age_group  # Add the import here too

        summary = results['summary']
        model_info = results.get('model_info', {})

        report = []
        report.append("SAMPLE STORY GENERATION REPORT - MISTRAL 7B")
        report.append("=" * 60)
        report.append(f"Generated: {results['timestamp']}")
        report.append(f"Model: {model_info.get('name', 'mistral-7b-instruct-v0.3')}")
        report.append(f"Base Model: {model_info.get('base_model', 'mistralai/Mistral-7B-Instruct-v0.3')}")
        report.append("")

        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Users: {summary['total_users']}")
        report.append(f"Total Prompts: {summary['total_prompts']}")
        report.append(f"Successful Generations: {summary['successful_generations']}")
        report.append(f"Failed Generations: {summary['failed_generations']}")
        report.append(f"Success Rate: {(summary['successful_generations'] / summary['total_prompts'] * 100):.1f}%")
        report.append("")

        # Quality scores
        if 'average_scores' in summary:
            avg_scores = summary['average_scores']
            report.append("QUALITY METRICS")
            report.append("-" * 30)
            report.append(f"Average Grammar Score: {avg_scores['grammar']:.1f}/100")
            report.append(f"Average Coherence Score: {avg_scores['coherence']:.1f}/100")
            report.append(f"Average Reading Level: {avg_scores['flesch_kincaid']:.1f}")
            report.append("")

        # Age group breakdown
        report.append("AGE GROUP BREAKDOWN")
        report.append("-" * 30)
        for age_group, count in summary['age_group_breakdown'].items():
            report.append(f"{age_group.capitalize()}: {count} stories")
        report.append("")

        # Safety statistics
        safety = summary['safety_stats']
        report.append("SAFETY STATISTICS")
        report.append("-" * 30)
        report.append(f"Safe Stories: {safety['safe_stories']}")
        report.append(f"Toxic Stories: {safety['toxic_stories']}")
        total_safety = safety['safe_stories'] + safety['toxic_stories']
        if total_safety > 0:
            report.append(f"Safety Rate: {(safety['safe_stories'] / total_safety * 100):.1f}%")
        report.append("")

        # Continuation statistics
        continuation = summary['continuation_stats']
        report.append("STORY CONTINUATION")
        report.append("-" * 30)
        report.append(f"Total Continuations: {continuation['total_continuations']}")
        report.append(f"Successful Continuations: {continuation['successful_continuations']}")
        if continuation['total_continuations'] > 0:
            success_rate = (continuation['successful_continuations'] / continuation['total_continuations'] * 100)
            report.append(f"Continuation Success Rate: {success_rate:.1f}%")
        report.append("")

        # Detailed user results
        report.append("DETAILED USER RESULTS")
        report.append("-" * 30)

        for username, user_results in results['detailed_results'].items():
            user_age = user_results[0]['age']
            age_group = get_age_group(user_age)  # Use imported function

            report.append(f"\n{username.upper()} (Age {user_age}, {age_group.capitalize()}):")

            for result in user_results:
                prompt_num = result['prompt_number']
                is_continuation = result['is_continuation']
                success = result['generation_result']['success']

                status = "✅" if success else "❌"
                continuation_mark = " [CONTINUATION]" if is_continuation else ""

                report.append(f"  {status} Prompt {prompt_num}{continuation_mark}")
                report.append(f"    {result['prompt']}")

                if success:
                    eval_data = result['evaluation']
                    report.append(f"    Length: {result['word_count']} words")
                    report.append(
                        f"    Quality: Grammar {eval_data.get('grammar_score', 0):.1f}, Coherence {eval_data.get('coherence_score', 0):.1f}")
                    report.append(f"    Age Prediction: {eval_data.get('predicted_age_group', 'N/A')}")
                    report.append(f"    Safe: {'Yes' if not eval_data.get('is_toxic', True) else 'No'}")
                else:
                    report.append(f"    Error: {result.get('error', 'Unknown error')}")

        return "\n".join(report)


def main():
    """Main function to run sample processing with Mistral 7B."""
    log_operation_status("Sample story processing using Mistral 7B")

    # Process samples (generation only)
    processor = SampleProcessor()
    results = processor.process_all_samples()

    if not results:
        print("Sample processing failed")
        return

    print(f"\nSample generation completed!")
    print(f"Generated {results['summary']['successful_generations']} stories")

    # Ask user if they want to evaluate
    evaluate_response = input("\nEvaluate generated samples? (y/N): ").strip().lower()

    if evaluate_response == 'y':
        print("Starting evaluation...")
        results = processor.evaluate_samples(results)
        print("Evaluation completed!")
    else:
        print("Skipping evaluation.")

    # Save results
    results_file = processor.save_results(results)
    print(f"Results saved to: {results_file}")

    # Generate and save report
    report = processor.generate_report(results)

    report_file = results_file.replace('mistral_results_', 'mistral_report_').replace('.json', '.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to: {report_file}")

    # Display summary
    print(f"\n{report}")

    print(f"\nResults available in: {processor.config['paths']['samples']}")


if __name__ == "__main__":
    main()