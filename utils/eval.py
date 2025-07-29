import pandas as pd
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from detoxify import Detoxify
import textstat
from typing import Dict, List, Any, Tuple, Optional, Union
import os
from datetime import datetime
from utils.helpers import (
    load_config, ensure_dir_exists, calculate_text_stats, get_age_group, log_operation_status
)
from helpers import set_cwd

# Get current working directory for path operations
cwd = set_cwd()


class GenericMistralEvaluator:
    """Generic evaluator that can use either shared or standalone Mistral models."""

    def __init__(self, model=None, tokenizer=None, model_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize with either shared model/tokenizer or model path for standalone loading.

        Args:
            model: Shared Mistral model (optional)
            tokenizer: Shared Mistral tokenizer (optional)
            model_path: Path to load standalone model (optional)
            config: Configuration dict (optional)
        """
        self.shared_model = model
        self.shared_tokenizer = tokenizer
        self.model_path = model_path
        self.config = config or load_config()
        self.eval_config = self.config['evaluation']

        # For standalone loading
        self.standalone_model = None
        self.standalone_tokenizer = None
        self.cache = {}

        self.batch_size = 4
        self.max_input_length = 512

        if model and tokenizer:
            print("Generic Mistral evaluator initialized with shared model")
        elif model_path:
            print("Generic Mistral evaluator initialized for standalone loading")
            self._load_standalone_model()
        else:
            print("Generic Mistral evaluator initialized without model - will load on demand")

    def _load_standalone_model(self):
        """Load standalone Mistral model for evaluation."""
        if self.standalone_model is not None:
            return  # Already loaded

        print("Loading standalone Mistral model for evaluation...")

        self.standalone_tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.standalone_tokenizer.pad_token is None:
            self.standalone_tokenizer.pad_token = self.standalone_tokenizer.eos_token

        self.standalone_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            use_cache=True
        )
        self.standalone_model.eval()
        print("Standalone Mistral evaluator loaded")

    def _get_model_and_tokenizer(self):
        """Get the appropriate model and tokenizer (shared or standalone)."""
        if self.shared_model and self.shared_tokenizer:
            return self.shared_model, self.shared_tokenizer
        elif self.standalone_model and self.standalone_tokenizer:
            return self.standalone_model, self.standalone_tokenizer
        elif self.model_path:
            self._load_standalone_model()
            return self.standalone_model, self.standalone_tokenizer
        else:
            raise RuntimeError("No model available - need either shared model or model_path")

    def _generate_response(self, prompt: str, max_tokens: int = 25) -> str:
        """Generate response using available model."""
        model, tokenizer = self._get_model_and_tokenizer()

        formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        # Move inputs to same device as model
        if next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response

    def evaluate_grammar_coherence(self, text: str) -> Dict[str, float]:
        """Evaluate grammar and coherence using available Mistral model."""
        # Truncate text for efficiency
        truncated_text = text[:self.max_input_length] if len(text) > self.max_input_length else text

        prompt = f"""Evaluate the following text on two criteria and respond with ONLY two numbers separated by a comma:
1. Grammar: Rate spelling, punctuation, and syntax from 0-100
2. Coherence: Rate logical flow, consistency, and clarity from 0-100

Text: {truncated_text}

Response format: grammar_score,coherence_score"""

        try:
            response = self._generate_response(prompt, max_tokens=20)
            numbers = re.findall(r'\d+', response)

            if len(numbers) >= 2:
                grammar_score = max(0, min(100, int(numbers[0])))
                coherence_score = max(0, min(100, int(numbers[1])))
                return {
                    'grammar_score': float(grammar_score),
                    'coherence_score': float(coherence_score)
                }
            else:
                return {'grammar_score': 75.0, 'coherence_score': 75.0}

        except Exception as e:
            print(f"Grammar/coherence evaluation failed: {e}")
            return {'grammar_score': 75.0, 'coherence_score': 75.0}

    def evaluate_age_appropriateness(self, text: str, target_age: int) -> Dict[str, Any]:
        """Evaluate age appropriateness using available Mistral model."""
        truncated_text = text[:self.max_input_length] if len(text) > self.max_input_length else text
        age_group = get_age_group(target_age)

        prompt = f"""Evaluate if this text is appropriate for a {target_age}-year-old. Respond with only three items separated by commas:
    1. "appropriate" or "inappropriate" 
    2. The best age group: "child", "kid", "teen", or "adult"
    3. A score from 0-100

    Text: {truncated_text}

    Answer:"""

        try:
            response = self._generate_response(prompt, max_tokens=15)

            # Clean and split response
            response = response.strip().lower()
            parts = [p.strip() for p in response.split(',')]

            if len(parts) >= 2:
                # Parse appropriateness
                is_appropriate = 'appropriate' in parts[0] and 'inappropriate' not in parts[0]

                # Parse predicted age group - be more robust
                predicted_age_group = age_group  # fallback to target age group
                for part in parts:
                    if 'child' in part:
                        predicted_age_group = 'child'
                        break
                    elif 'kid' in part:
                        predicted_age_group = 'kid'
                        break
                    elif 'teen' in part:
                        predicted_age_group = 'teen'
                        break
                    elif 'adult' in part:
                        predicted_age_group = 'adult'
                        break

                # Parse score
                score = 75.0  # default
                if len(parts) >= 3:
                    score_match = re.search(r'\d+', parts[2])
                    if score_match:
                        score = max(0, min(100, int(score_match.group())))

                return {
                    'is_appropriate': is_appropriate,
                    'predicted_age_group': predicted_age_group,
                    'appropriateness_score': float(score)
                }
            else:
                return {
                    'is_appropriate': True,
                    'predicted_age_group': age_group,
                    'appropriateness_score': 75.0
                }

        except Exception as e:
            print(f"Age appropriateness evaluation failed: {e}")
            return {
                'is_appropriate': True,
                'predicted_age_group': age_group,
                'appropriateness_score': 75.0
            }

    def evaluate_grammar_coherence_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Evaluate grammar and coherence for multiple texts in batch."""
        results = []
        for text in texts:
            results.append(self.evaluate_grammar_coherence(text))
        return results

    def evaluate_age_appropriateness_batch(self, texts: List[str], target_ages: List[int]) -> List[Dict[str, Any]]:
        """Evaluate age appropriateness for multiple texts in batch."""
        results = []
        for text, target_age in zip(texts, target_ages):
            results.append(self.evaluate_age_appropriateness(text, target_age))
        return results


def load_evaluation_models(shared_model=None, shared_tokenizer=None):
    """
    Load models required for evaluation.

    Args:
        shared_model: Optional shared Mistral model to reuse
        shared_tokenizer: Optional shared Mistral tokenizer to reuse
    """
    print("Loading evaluation models...")
    config = load_config()

    # Determine model path
    models_path = config['paths']['models']
    tuned_model_path = os.path.join(models_path, "tuned_story_llm")
    base_model_path = os.path.join(models_path, "mistral-7b-base")

    if shared_model and shared_tokenizer:
        print("Using shared Mistral model for evaluation")
        mistral_evaluator = GenericMistralEvaluator(
            model=shared_model,
            tokenizer=shared_tokenizer,
            config=config
        )
    else:
        if os.path.exists(tuned_model_path):
            model_path = tuned_model_path
            print("Using fine-tuned Mistral model for evaluation")
        elif os.path.exists(base_model_path):
            model_path = base_model_path
            print("Using base Mistral model for evaluation")
        else:
            model_path = config['model']['base_model']
            print("Using Hugging Face model for evaluation")

        mistral_evaluator = GenericMistralEvaluator(
            model_path=model_path,
            config=config
        )

    # Load toxicity detector
    detoxify_model = Detoxify('original')

    # Load perplexity model (GPT-2) - keep on CPU to save VRAM
    perplexity_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    perplexity_model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu"  # Keep on CPU to save VRAM
    )

    if perplexity_tokenizer.pad_token is None:
        perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token

    return mistral_evaluator, detoxify_model, perplexity_tokenizer, perplexity_model


def calculate_perplexity(text: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> float:
    """Calculate perplexity score for text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

    # GPT-2 is on CPU, so move inputs to CPU
    inputs = {k: v.cpu() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return min(perplexity, 1000.0)


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


def evaluate_single_text(text: str, target_age: int, evaluation_models: Tuple,
                         include_perplexity: bool = True) -> Dict[str, Any]:
    """
    Generic function to evaluate a single text across all metrics.
    Works with both shared and standalone evaluation models.
    """
    mistral_evaluator, detoxify_model, perplexity_tokenizer, perplexity_model = evaluation_models
    config = load_config()

    # Basic text statistics
    text_stats = calculate_text_stats(text)

    # Perplexity evaluation (optional for performance)
    if include_perplexity:
        perplexity = calculate_perplexity(text, perplexity_tokenizer, perplexity_model)
        perplexity_category = categorize_perplexity(perplexity, config['evaluation']['perplexity_buckets'])
    else:
        perplexity = 50.0
        perplexity_category = "medium"

    # Grammar and coherence using Mistral
    grammar_coherence = mistral_evaluator.evaluate_grammar_coherence(text)

    # Age appropriateness using Mistral
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


def evaluate_stories_list(stories: List[str], models: Tuple, config: Dict[str, Any],
                          description: str = "stories", include_perplexity: bool = False) -> List[Dict[str, Any]]:
    """
    Generic function to evaluate a list of stories.
    Works with both shared and standalone evaluation models.
    """
    print(f"Evaluating {len(stories)} {description}...")

    mistral_evaluator, detoxify_model, perplexity_tokenizer, perplexity_model = models
    evaluation_results = []

    default_age = 12  # Kid age group for general evaluation

    from tqdm import tqdm

    for story in tqdm(stories, desc=f"Evaluating {description}", unit="story"):
        try:
            result = evaluate_single_text(
                story, default_age, models, include_perplexity=include_perplexity
            )
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


def categorize_perplexity(perplexity: float, buckets: List[int]) -> str:
    """Categorize perplexity into buckets."""
    if perplexity <= buckets[0]:
        return "low"
    elif perplexity <= buckets[1]:
        return "medium"
    else:
        return "high"


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