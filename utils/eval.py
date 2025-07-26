import pandas as pd
import numpy as np
import torch
import re
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
from detoxify import Detoxify
import textstat
from typing import Dict, List, Any, Tuple, Optional
import os
from datetime import datetime
from utils.helpers import (
    load_config, ensure_dir_exists, calculate_text_stats, get_age_group,
    check_cache_overwrite, log_operation_status, create_progress_bar, batch_process
)
from helpers import set_cwd

# Get current working directory for path operations
cwd = set_cwd()


class OptimizedMistralEvaluator:
    """Optimized evaluator using Mistral model with batching and caching for faster evaluation."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.eval_config = config['evaluation']
        self.model = None
        self.tokenizer = None
        self.cache = {}
        self.batch_size = 4  # Process multiple texts at once
        self.max_input_length = 512  # Reduced from 800 for faster processing
        self._load_model()

    def _load_model(self):
        """Load Mistral model for evaluation with optimizations."""
        print("Loading optimized Mistral model for evaluation...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            use_cache=True  # Enable KV cache for faster generation
        )
        self.model.eval()

        # Pre-compile common prompt templates
        self._precompile_prompts()
        print("Optimized Mistral evaluator loaded")

    def _precompile_prompts(self):
        """Pre-tokenize common prompt templates to save time."""
        self.grammar_prompt_template = "Evaluate the following text on two criteria and respond with ONLY two numbers separated by a comma:\n1. Grammar: Rate spelling, punctuation, and syntax from 0-100\n2. Coherence: Rate logical flow, consistency, and clarity from 0-100\n\nText: {}\n\nResponse format: grammar_score,coherence_score"

        self.age_prompt_template = "Is this text appropriate for a {}-year-old ({})? Respond with ONLY: appropriate/inappropriate,predicted_age_group,score\n\nWhere score is 0-100 (100 = perfectly appropriate)\n\nText: {}\n\nResponse format: appropriate/inappropriate,predicted_age_group,score"

    def _get_cache_key(self, text: str, prompt_type: str) -> str:
        """Generate cache key for text and prompt type."""
        return f"{prompt_type}_{hash(text[:200])}"

    def _truncate_text(self, text: str) -> str:
        """Truncate text to reduce processing time while maintaining quality."""
        # Use first and last portions to capture beginning and conclusion
        if len(text) <= self.max_input_length:
            return text

        half_length = self.max_input_length // 2
        return text[:half_length] + "..." + text[-half_length:]

    def _generate_batch_responses(self, prompts: List[str], max_tokens: int = 25) -> List[str]:
        """Generate responses for multiple prompts in a batch."""
        if not prompts:
            return []

        # Format prompts for Mistral
        formatted_prompts = [f"<s>[INST] {prompt} [/INST]" for prompt in prompts]

        # Tokenize all prompts
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent results
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[inputs['input_ids'][i].shape[0]:],
                skip_special_tokens=True
            ).strip()
            responses.append(response)

        return responses

    def evaluate_grammar_coherence_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Evaluate grammar and coherence for multiple texts in batch."""
        results = []

        # Check cache first
        prompts = []
        cache_indices = []

        for i, text in enumerate(texts):
            truncated_text = self._truncate_text(text)
            cache_key = self._get_cache_key(truncated_text, "grammar")

            if cache_key in self.cache:
                results.append(self.cache[cache_key])
            else:
                prompt = self.grammar_prompt_template.format(truncated_text)
                prompts.append(prompt)
                cache_indices.append((i, cache_key))
                results.append(None)  # Placeholder

        # Process uncached prompts
        if prompts:
            try:
                batch_responses = self._generate_batch_responses(prompts, max_tokens=20)

                for (result_idx, cache_key), response in zip(cache_indices, batch_responses):
                    numbers = re.findall(r'\d+', response)

                    if len(numbers) >= 2:
                        grammar_score = max(0, min(100, int(numbers[0])))
                        coherence_score = max(0, min(100, int(numbers[1])))
                        result = {
                            'grammar_score': float(grammar_score),
                            'coherence_score': float(coherence_score)
                        }
                    else:
                        result = {'grammar_score': 75.0, 'coherence_score': 75.0}

                    # Cache result
                    self.cache[cache_key] = result
                    results[result_idx] = result

            except Exception as e:
                print(f"Batch grammar/coherence evaluation failed: {e}")
                # Fill remaining results with defaults
                for result_idx, _ in cache_indices:
                    if results[result_idx] is None:
                        results[result_idx] = {'grammar_score': 75.0, 'coherence_score': 75.0}

        return results

    def evaluate_age_appropriateness_batch(self, texts: List[str], target_ages: List[int]) -> List[Dict[str, Any]]:
        """Evaluate age appropriateness for multiple texts in batch."""
        results = []

        # Check cache first
        prompts = []
        cache_indices = []

        for i, (text, target_age) in enumerate(zip(texts, target_ages)):
            truncated_text = self._truncate_text(text)
            age_group = get_age_group(target_age)
            cache_key = self._get_cache_key(f"{truncated_text}_{target_age}", "age")

            if cache_key in self.cache:
                results.append(self.cache[cache_key])
            else:
                prompt = self.age_prompt_template.format(target_age, age_group, truncated_text)
                prompts.append(prompt)
                cache_indices.append((i, cache_key, age_group))
                results.append(None)  # Placeholder

        # Process uncached prompts
        if prompts:
            try:
                batch_responses = self._generate_batch_responses(prompts, max_tokens=30)

                for (result_idx, cache_key, age_group), response in zip(cache_indices, batch_responses):
                    parts = [p.strip() for p in response.split(',')]

                    if len(parts) >= 3:
                        is_appropriate = parts[0].lower() == 'appropriate'
                        predicted_age_group = parts[1].lower()
                        score_match = re.search(r'\d+', parts[2])
                        score = max(0, min(100, int(score_match.group()))) if score_match else 75

                        result = {
                            'is_appropriate': is_appropriate,
                            'predicted_age_group': predicted_age_group,
                            'appropriateness_score': float(score)
                        }
                    else:
                        result = {
                            'is_appropriate': True,
                            'predicted_age_group': age_group,
                            'appropriateness_score': 75.0
                        }

                    # Cache result
                    self.cache[cache_key] = result
                    results[result_idx] = result

            except Exception as e:
                print(f"Batch age appropriateness evaluation failed: {e}")
                # Fill remaining results with defaults
                for result_idx, _, age_group in cache_indices:
                    if results[result_idx] is None:
                        results[result_idx] = {
                            'is_appropriate': True,
                            'predicted_age_group': age_group,
                            'appropriateness_score': 75.0
                        }

        return results

    def evaluate_grammar_coherence(self, text: str) -> Dict[str, float]:
        """Single text evaluation - uses batch method for consistency."""
        return self.evaluate_grammar_coherence_batch([text])[0]

    def evaluate_age_appropriateness(self, text: str, target_age: int) -> Dict[str, Any]:
        """Single text evaluation - uses batch method for consistency."""
        return self.evaluate_age_appropriateness_batch([text], [target_age])[0]


def load_evaluation_models():
    """Load models required for evaluation."""
    print("Loading evaluation models...")
    config = load_config()

    # Determine model path (try fine-tuned first, then base)
    models_path = config['paths']['models']
    tuned_model_path = os.path.join(models_path, "tuned_story_llm")
    base_model_path = os.path.join(models_path, "mistral-7b-base")

    if os.path.exists(tuned_model_path):
        model_path = tuned_model_path
        print("Using fine-tuned Mistral model for evaluation")
    elif os.path.exists(base_model_path):
        model_path = base_model_path
        print("Using base Mistral model for evaluation")
    else:
        model_path = config['model']['base_model']
        print("Using Hugging Face model for evaluation")

    # Initialize optimized Mistral evaluator
    mistral_evaluator = OptimizedMistralEvaluator(model_path, config)

    # Load toxicity detector
    detoxify_model = Detoxify('original')

    # Load perplexity model (GPT-2)
    perplexity_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    perplexity_model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if perplexity_tokenizer.pad_token is None:
        perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token

    return mistral_evaluator, detoxify_model, perplexity_tokenizer, perplexity_model


def calculate_perplexity(text: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> float:
    """Calculate perplexity score for text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return min(perplexity, 1000.0)


def categorize_perplexity(perplexity: float, buckets: List[int]) -> str:
    """Categorize perplexity into buckets."""
    if perplexity <= buckets[0]:
        return "low"
    elif perplexity <= buckets[1]:
        return "medium"
    else:
        return "high"


def evaluate_toxicity(text: str, detoxify_model) -> Dict[str, float]:
    """Evaluate text toxicity using Detoxify."""
    results = detoxify_model.predict(text)

    return {
        'toxicity_score': float(results['toxicity']),
        'severe_toxicity': float(results['severe_toxicity']),
        'obscene': float(results['obscene']),
        'threat': float(results['threat']),
        'insult': float(results['insult']),
        'identity_attack': float(results['identity_attack'])
    }


def evaluate_toxicity_batch(texts: List[str], detoxify_model) -> List[Dict[str, float]]:
    """Evaluate toxicity for multiple texts in batch."""
    results = []

    # Process in smaller batches to avoid memory issues
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_results = detoxify_model.predict(batch_texts)

        for j in range(len(batch_texts)):
            result = {
                'toxicity_score': float(batch_results['toxicity'][j]),
                'severe_toxicity': float(batch_results['severe_toxicity'][j]),
                'obscene': float(batch_results['obscene'][j]),
                'threat': float(batch_results['threat'][j]),
                'insult': float(batch_results['insult'][j]),
                'identity_attack': float(batch_results['identity_attack'][j])
            }
            results.append(result)

    return results


def evaluate_readability(text: str, target_age: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate text readability."""
    try:
        flesch_score = textstat.flesch_reading_ease(text)
        flesch_kincaid = textstat.flesch_kincaid_grade(text)

        age_group = get_age_group(target_age)
        target_range = config['evaluation']['flesch_kincaid_ranges'][age_group]

        is_appropriate_level = target_range[0] <= flesch_kincaid <= target_range[1]

        target_middle = (target_range[0] + target_range[1]) / 2
        distance_from_target = abs(flesch_kincaid - target_middle)
        readability_score = max(0, min(100, 100 - (distance_from_target * 10)))

        return {
            'flesch_reading_ease': flesch_score,
            'flesch_kincaid_grade': flesch_kincaid,
            'target_grade_range': target_range,
            'is_appropriate_level': is_appropriate_level,
            'readability_score': readability_score
        }
    except Exception:
        return {
            'flesch_reading_ease': 50.0,
            'flesch_kincaid_grade': 10.0,
            'target_grade_range': [0, 20],
            'is_appropriate_level': True,
            'readability_score': 50.0
        }


def evaluate_single_text(text: str, target_age: int, evaluation_models: Tuple) -> Dict[str, Any]:
    """Evaluate a single text across all metrics."""
    mistral_evaluator, detoxify_model, perplexity_tokenizer, perplexity_model = evaluation_models
    config = load_config()

    # Basic text statistics
    text_stats = calculate_text_stats(text)

    # Perplexity evaluation
    perplexity = calculate_perplexity(text, perplexity_tokenizer, perplexity_model)
    perplexity_category = categorize_perplexity(perplexity, config['evaluation']['perplexity_buckets'])

    # Grammar and coherence using optimized Mistral
    grammar_coherence = mistral_evaluator.evaluate_grammar_coherence(text)

    # Age appropriateness using optimized Mistral
    age_eval = mistral_evaluator.evaluate_age_appropriateness(text, target_age)

    # Toxicity
    toxicity = evaluate_toxicity(text, detoxify_model)
    is_toxic = toxicity['toxicity_score'] > config['evaluation']['toxicity_threshold']

    # Readability
    readability = evaluate_readability(text, target_age, config)

    return {
        'text_stats': text_stats,
        'perplexity': perplexity,
        'perplexity_category': perplexity_category,
        'grammar_score': grammar_coherence['grammar_score'],
        'coherence_score': grammar_coherence['coherence_score'],
        'predicted_age_group': age_eval['predicted_age_group'],
        'appropriateness_score': age_eval['appropriateness_score'],
        'is_age_appropriate': age_eval['is_appropriate'],
        'is_toxic': is_toxic,
        'toxicity_scores': toxicity,
        'readability': readability,
        'flesch_kincaid_score': readability['flesch_kincaid_grade'],
        'overall_quality_score': (
                                         grammar_coherence['grammar_score'] +
                                         grammar_coherence['coherence_score'] +
                                         age_eval['appropriateness_score'] +
                                         readability['readability_score']
                                 ) / 4
    }


def evaluate_stories_list_optimized(stories: List[str], models: Tuple, config: Dict[str, Any],
                                    description: str = "stories") -> List[Dict[str, Any]]:
    """Optimized evaluation for a list of stories using batch processing."""
    print(f"Evaluating {len(stories)} {description} with optimized batch processing...")

    mistral_evaluator, detoxify_model, perplexity_tokenizer, perplexity_model = models
    evaluation_results = []

    default_age = 12  # Kid age group for general evaluation

    # Process in batches for efficiency
    batch_size = mistral_evaluator.batch_size

    from tqdm import tqdm

    for i in tqdm(range(0, len(stories), batch_size), desc=f"Processing {description} batches", unit="batch"):
        batch_stories = stories[i:i + batch_size]
        batch_ages = [default_age] * len(batch_stories)

        try:
            # Batch evaluations
            grammar_coherence_results = mistral_evaluator.evaluate_grammar_coherence_batch(batch_stories)
            age_results = mistral_evaluator.evaluate_age_appropriateness_batch(batch_stories, batch_ages)
            toxicity_results = evaluate_toxicity_batch(batch_stories, detoxify_model)

            for j, story in enumerate(batch_stories):
                result = {
                    'grammar_score': grammar_coherence_results[j]['grammar_score'],
                    'coherence_score': grammar_coherence_results[j]['coherence_score'],
                    'flesch_kincaid_score': 8.0,  # Default, calculated separately if needed
                    'perplexity': 50.0,  # Default, calculated separately if needed
                    'is_toxic': toxicity_results[j]['toxicity_score'] > config['evaluation']['toxicity_threshold']
                }

                # Basic text stats
                text_stats = calculate_text_stats(story)
                result.update(text_stats)

                evaluation_results.append(result)

        except Exception as e:
            print(f"Batch evaluation failed: {e}")
            # Fallback to individual processing for this batch
            for story in batch_stories:
                result = {
                    'grammar_score': 75.0,
                    'coherence_score': 75.0,
                    'flesch_kincaid_score': 8.0,
                    'perplexity': 50.0,
                    'is_toxic': True,  # Mark as toxic to be safe
                    'length': len(story),
                    'word_count': len(story.split()),
                    'sentence_count': len(story.split('.')),
                    'avg_word_length': 5.0,
                    'avg_sentence_length': 15.0
                }
                evaluation_results.append(result)

    return evaluation_results


def evaluate_stories_list(stories: List[str], models: Tuple, config: Dict[str, Any], description: str = "stories") -> \
List[Dict[str, Any]]:
    """Evaluate a list of stories with optional optimization."""
    # Use optimized version if Mistral evaluator supports batch processing
    mistral_evaluator = models[0]
    if hasattr(mistral_evaluator, 'evaluate_grammar_coherence_batch'):
        return evaluate_stories_list_optimized(stories, models, config, description)

    # Fallback to original implementation
    print(f"Evaluating {len(stories)} {description}...")

    mistral_evaluator, detoxify_model, perplexity_tokenizer, perplexity_model = models
    evaluation_results = []

    default_age = 12

    from tqdm import tqdm

    for story in tqdm(stories, desc=f"Evaluating {description}", unit="story"):
        try:
            result = {
                'grammar_score': 75.0,
                'coherence_score': 75.0,
                'flesch_kincaid_score': 8.0,
                'perplexity': 50.0,
                'is_toxic': False
            }

            # Toxicity evaluation
            toxicity = evaluate_toxicity(story, detoxify_model)
            result['is_toxic'] = toxicity['toxicity_score'] > config['evaluation']['toxicity_threshold']

            # Grammar and coherence using Mistral
            grammar_coherence = mistral_evaluator.evaluate_grammar_coherence(story)
            result['grammar_score'] = grammar_coherence['grammar_score']
            result['coherence_score'] = grammar_coherence['coherence_score']

            # Readability
            readability = evaluate_readability(story, default_age, config)
            result['flesch_kincaid_score'] = readability['flesch_kincaid_grade']

            # Basic text stats
            text_stats = calculate_text_stats(story)
            result.update(text_stats)

            evaluation_results.append(result)

        except Exception as e:
            print(f"Evaluation failed for story: {e}")
            result = {
                'grammar_score': 50.0,
                'coherence_score': 50.0,
                'flesch_kincaid_score': 10.0,
                'perplexity': 75.0,
                'is_toxic': True,
                'length': len(story),
                'word_count': len(story.split()),
                'sentence_count': len(story.split('.')),
                'avg_word_length': 5.0,
                'avg_sentence_length': 15.0
            }
            evaluation_results.append(result)

    return evaluation_results


def filter_safe_stories(stories: List[str], evaluations: List[Dict[str, Any]]) -> List[str]:
    """Filter out toxic stories based on evaluation results."""
    print("Filtering safe stories...")

    safe_stories = []
    for story, eval_result in zip(stories, evaluations):
        if not eval_result.get('is_toxic', True):
            safe_stories.append(story)

    removed_count = len(stories) - len(safe_stories)
    print(f"Filtered out {removed_count} toxic stories from {len(stories)} total")

    return safe_stories


def evaluate_story_dataset(stories_df: pd.DataFrame, age_group: str) -> Dict[str, Any]:
    """Evaluate an entire dataset of stories."""
    print(f"Evaluating {age_group} story dataset...")

    evaluation_models = load_evaluation_models()
    config = load_config()

    # Sample age for age group
    age_ranges = config['age_groups'][age_group]
    sample_age = (age_ranges[0] + age_ranges[1]) // 2

    results = []
    total_stories = len(stories_df)

    from tqdm import tqdm

    for idx, row in tqdm(stories_df.iterrows(), total=total_stories, desc=f"Evaluating {age_group} stories",
                         unit="story"):
        text = row['text']

        try:
            eval_result = evaluate_single_text(text, sample_age, evaluation_models)
            eval_result['story_id'] = idx
            eval_result['age_group'] = age_group
            results.append(eval_result)
        except Exception as e:
            print(f"Evaluation failed for story {idx}: {e}")
            continue

    # Calculate summary statistics
    if results:
        summary_stats = {
            'total_stories': len(results),
            'avg_quality_score': np.mean([r['overall_quality_score'] for r in results]),
            'avg_grammar_score': np.mean([r['grammar_score'] for r in results]),
            'avg_coherence_score': np.mean([r['coherence_score'] for r in results]),
            'avg_appropriateness_score': np.mean([r['appropriateness_score'] for r in results]),
            'toxic_stories': sum([r['is_toxic'] for r in results]),
            'safe_stories': sum([not r['is_toxic'] for r in results]),
            'age_appropriate_stories': sum([r['is_age_appropriate'] for r in results]),
            'perplexity_distribution': {
                'low': sum([r['perplexity_category'] == 'low' for r in results]),
                'medium': sum([r['perplexity_category'] == 'medium' for r in results]),
                'high': sum([r['perplexity_category'] == 'high' for r in results])
            }
        }
    else:
        summary_stats = {
            'total_stories': 0,
            'avg_quality_score': 0,
            'avg_grammar_score': 0,
            'avg_coherence_score': 0,
            'avg_appropriateness_score': 0,
            'toxic_stories': 0,
            'safe_stories': 0,
            'age_appropriate_stories': 0,
            'perplexity_distribution': {'low': 0, 'medium': 0, 'high': 0}
        }

    return {
        'age_group': age_group,
        'summary_stats': summary_stats,
        'detailed_results': results
    }


def process_all_datasets() -> bool:
    """Process and evaluate all age group datasets."""
    print("Processing all datasets for evaluation...")

    config = load_config()
    processed_path = config['paths']['data_processed']
    evaluated_path = config['paths']['data_evaluated']

    ensure_dir_exists(evaluated_path)

    age_groups = ['child', 'kid', 'teen', 'adult']
    all_results = {}

    for age_group in age_groups:
        file_path = os.path.join(processed_path, f"{age_group}_stories.csv")

        if not os.path.exists(file_path):
            print(f"Missing {age_group} dataset")
            continue

        stories_df = pd.read_csv(file_path)
        print(f"Evaluating {len(stories_df)} {age_group} stories...")

        results = evaluate_story_dataset(stories_df, age_group)
        all_results[age_group] = results

        # Save individual results
        results_file = os.path.join(evaluated_path, f"{age_group}_evaluation.json")
        from utils.helpers import save_json
        save_json(results, results_file)
        print(f"Saved {age_group} evaluation results")

    # Save combined results
    combined_file = os.path.join(evaluated_path, "all_evaluations.json")
    from utils.helpers import save_json
    save_json(all_results, combined_file)

    # Generate summary report
    generate_evaluation_report(all_results, evaluated_path)

    print("Dataset evaluation completed")
    return True


def generate_evaluation_report(all_results: Dict[str, Any], output_path: str):
    """Generate a comprehensive evaluation report."""
    print("Generating evaluation report...")

    report_lines = [
        "STORY DATASET EVALUATION REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SUMMARY BY AGE GROUP",
        "-" * 30
    ]

    total_stories = 0
    total_safe = 0
    total_appropriate = 0

    for age_group, results in all_results.items():
        stats = results['summary_stats']
        total_stories += stats['total_stories']
        total_safe += stats['safe_stories']
        total_appropriate += stats['age_appropriate_stories']

        report_lines.extend([
            f"",
            f"{age_group.upper()} STORIES:",
            f"  Total: {stats['total_stories']}",
            f"  Quality Score: {stats['avg_quality_score']:.1f}/100",
            f"  Grammar: {stats['avg_grammar_score']:.1f}/100",
            f"  Coherence: {stats['avg_coherence_score']:.1f}/100",
            f"  Age Appropriateness: {stats['avg_appropriateness_score']:.1f}/100",
            f"  Safe Stories: {stats['safe_stories']} ({(stats['safe_stories'] / max(stats['total_stories'], 1) * 100):.1f}%)",
            f"  Age Appropriate: {stats['age_appropriate_stories']} ({(stats['age_appropriate_stories'] / max(stats['total_stories'], 1) * 100):.1f}%)"
        ])

    report_lines.extend([
        "",
        "OVERALL STATISTICS",
        "-" * 30,
        f"Total Stories Evaluated: {total_stories}",
        f"Overall Safety Rate: {(total_safe / max(total_stories, 1) * 100):.1f}%",
        f"Overall Appropriateness Rate: {(total_appropriate / max(total_stories, 1) * 100):.1f}%"
    ])

    report_file = os.path.join(output_path, "evaluation_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Report saved to {report_file}")


def main():
    """Main function for evaluation."""
    log_operation_status("Story dataset evaluation using optimized Mistral")

    process_all_datasets()

    log_operation_status("Story dataset evaluation", "completed")


if __name__ == "__main__":
    main()