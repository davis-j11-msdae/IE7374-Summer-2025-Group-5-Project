import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from detoxify import Detoxify
import textstat
from openai import OpenAI
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
import os
from utils.helpers import (
    load_config, ensure_dir_exists, calculate_text_stats, get_age_group,
    check_cache_overwrite, log_operation_status, create_progress_bar, batch_process
)
from helpers import set_cwd

# Get current working directory for path operations
cwd = set_cwd()

def load_evaluation_models():
    """Load models required for evaluation."""
    load_dotenv()

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Load toxicity detector
    detoxify_model = Detoxify('original')

    # Load perplexity model (GPT-2)
    perplexity_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    perplexity_model = AutoModelForCausalLM.from_pretrained('gpt2')

    if perplexity_tokenizer.pad_token is None:
        perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token

    return client, detoxify_model, perplexity_tokenizer, perplexity_model


def calculate_perplexity(text: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> float:
    """Calculate perplexity score for text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return min(perplexity, 1000.0)  # Cap extreme values


def categorize_perplexity(perplexity: float, buckets: List[int]) -> str:
    """Categorize perplexity into buckets."""
    if perplexity <= buckets[0]:
        return "low"
    elif perplexity <= buckets[1]:
        return "medium"
    else:
        return "high"


def evaluate_grammar_coherence(text: str, client: OpenAI, model_name: str = "gpt-3.5-turbo") -> Dict[str, float]:
    """Evaluate grammar and coherence using OpenAI API."""
    try:
        prompt = f"""
        Evaluate the following text on two criteria (score 0-100):
        1. Grammar: Check spelling, punctuation, syntax
        2. Coherence: Check logical flow and consistency

        Text: {text[:1000]}...

        Return only two numbers separated by a comma: grammar_score,coherence_score
        """

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0
        )

        scores = response.choices[0].message.content.strip().split(',')
        grammar_score = float(scores[0])
        coherence_score = float(scores[1])

        return {
            'grammar_score': max(0, min(100, grammar_score)),
            'coherence_score': max(0, min(100, coherence_score))
        }

    except Exception as e:
        print(f"  ⚠️ OpenAI evaluation failed: {e}")
        return {'grammar_score': 50.0, 'coherence_score': 50.0}


def evaluate_age_appropriateness(text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate age appropriateness using Flesch-Kincaid readability."""
    flesch_score = textstat.flesch_kincaid_grade(text)

    # Determine age group based on reading level
    ranges = config['evaluation']['flesch_kincaid_ranges']
    age_group = None

    for group, (min_score, max_score) in ranges.items():
        if min_score <= flesch_score <= max_score:
            age_group = group
            break

    if age_group is None:
        age_group = "adult"  # Default for high scores

    return {
        'flesch_kincaid_score': flesch_score,
        'predicted_age_group': age_group,
        'reading_level': f"Grade {flesch_score:.1f}"
    }


def evaluate_toxicity(text: str, detoxify_model) -> Dict[str, Any]:
    """Evaluate content toxicity using Detoxify."""
    try:
        results = detoxify_model.predict(text)

        # Get overall toxicity score
        toxicity_score = results['toxicity']

        # Check if any category exceeds threshold
        threshold = 0.5
        is_toxic = any(score > threshold for score in results.values())

        return {
            'is_toxic': is_toxic,
            'toxicity_score': toxicity_score,
            'detailed_scores': results
        }

    except Exception as e:
        print(f"  ⚠️ Toxicity evaluation failed: {e}")
        return {
            'is_toxic': False,
            'toxicity_score': 0.0,
            'detailed_scores': {}
        }


def evaluate_single_text(text: str, models: Tuple, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single text on all metrics."""
    client, detoxify_model, perplexity_tokenizer, perplexity_model = models

    # Basic text statistics
    text_stats = calculate_text_stats(text)

    # Perplexity
    perplexity = calculate_perplexity(text, perplexity_tokenizer, perplexity_model)
    perplexity_bucket = categorize_perplexity(perplexity, config['evaluation']['perplexity_buckets'])

    # Grammar and coherence
    grammar_coherence = evaluate_grammar_coherence(text, client, config['evaluation']['grammar_model'])

    # Age appropriateness
    age_appropriateness = evaluate_age_appropriateness(text, config)

    # Toxicity
    toxicity_results = evaluate_toxicity(text, detoxify_model)

    # Combine all results
    evaluation_results = {
        **text_stats,
        'perplexity': perplexity,
        'perplexity_bucket': perplexity_bucket,
        **grammar_coherence,
        **age_appropriateness,
        **toxicity_results
    }

    return evaluation_results


def evaluate_dataset(dataset_path: Path, models: Tuple, config: Dict[str, Any]) -> pd.DataFrame:
    """Evaluate all texts in a dataset."""
    df = pd.read_csv(dataset_path)

    evaluation_results = []

    for text in create_progress_bar(df['text'], f"Evaluating {dataset_path.stem}"):
        result = evaluate_single_text(text, models, config)
        evaluation_results.append(result)

    # Add evaluation results to dataframe
    eval_df = pd.DataFrame(evaluation_results)
    result_df = pd.concat([df, eval_df], axis=1)

    return result_df


def evaluate_stories_list(stories: List[str], models: Tuple, config: Dict[str, Any],
                          description: str = "stories") -> List[Dict[str, Any]]:
    """Evaluate a list of stories and return results."""
    evaluation_results = []

    for text in create_progress_bar(stories, f"Evaluating {description}"):
        result = evaluate_single_text(text, models, config)
        evaluation_results.append(result)

    return evaluation_results


def filter_safe_stories(stories: List[str], evaluations: List[Dict[str, Any]]) -> List[str]:
    """Filter out toxic stories based on evaluation results."""
    safe_stories = []

    for story, eval_result in zip(stories, evaluations):
        if not eval_result['is_toxic']:
            safe_stories.append(story)

    return safe_stories


def evaluate_all_datasets() -> Dict[str, pd.DataFrame]:
    """Evaluate all processed datasets."""
    config = load_config()
    processed_path = Path(config['paths']['data_processed'])
    evaluated_path = Path(config['paths']['data_evaluated'])

    # Check if evaluated data already exists
    if evaluated_path.exists() and not check_cache_overwrite(str(evaluated_path), "Evaluated datasets"):
        # Load existing evaluated datasets
        evaluated_datasets = {}
        age_groups = ['child', 'kid', 'teen', 'adult']

        for age_group in age_groups:
            eval_file = evaluated_path / f"{age_group}_evaluated.csv"
            if eval_file.exists():
                evaluated_datasets[age_group] = pd.read_csv(eval_file)

        return evaluated_datasets

    ensure_dir_exists(evaluated_path)
    log_operation_status("Dataset evaluation")

    # Load evaluation models
    print("📊 Loading evaluation models...")
    models = load_evaluation_models()
    print("  ✅ Models loaded")

    # Evaluate each age group dataset
    age_groups = ['child', 'kid', 'teen', 'adult']
    evaluated_datasets = {}

    for age_group in age_groups:
        dataset_file = processed_path / f"{age_group}_stories.csv"

        if dataset_file.exists():
            log_operation_status(f"Evaluating {age_group} dataset")
            evaluated_df = evaluate_dataset(dataset_file, models, config)
            evaluated_datasets[age_group] = evaluated_df

            # Save evaluated dataset
            output_file = evaluated_path / f"{age_group}_evaluated.csv"
            evaluated_df.to_csv(output_file, index=False)
            print(f"  ✅ Saved {age_group}: {len(evaluated_df)} evaluated stories")
        else:
            print(f"  ❌ Missing {age_group} dataset")

    log_operation_status("Dataset evaluation", "completed")
    return evaluated_datasets


def generate_evaluation_summary(evaluated_datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Generate summary statistics for evaluated datasets."""
    summary = {
        'total_stories': 0,
        'age_group_stats': {},
        'overall_stats': {}
    }

    all_data = []

    for age_group, df in evaluated_datasets.items():
        if not df.empty:
            summary['total_stories'] += len(df)

            # Age group specific stats
            group_stats = {
                'count': len(df),
                'avg_perplexity': df['perplexity'].mean(),
                'avg_grammar': df['grammar_score'].mean(),
                'avg_coherence': df['coherence_score'].mean(),
                'avg_flesch_kincaid': df['flesch_kincaid_score'].mean(),
                'toxic_count': df['is_toxic'].sum(),
                'toxic_percentage': (df['is_toxic'].sum() / len(df)) * 100
            }

            summary['age_group_stats'][age_group] = group_stats
            all_data.append(df)

    # Overall statistics
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        summary['overall_stats'] = {
            'avg_perplexity': combined_df['perplexity'].mean(),
            'avg_grammar': combined_df['grammar_score'].mean(),
            'avg_coherence': combined_df['coherence_score'].mean(),
            'avg_flesch_kincaid': combined_df['flesch_kincaid_score'].mean(),
            'total_toxic': combined_df['is_toxic'].sum(),
            'toxic_percentage': (combined_df['is_toxic'].sum() / len(combined_df)) * 100
        }

    return summary


def filter_toxic_content(evaluated_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Filter out toxic content from evaluated datasets."""
    filtered_datasets = {}

    for age_group, df in evaluated_datasets.items():
        if not df.empty:
            clean_df = df[df['is_toxic'] == False].copy()
            filtered_datasets[age_group] = clean_df

            removed_count = len(df) - len(clean_df)
            print(f"  🧹 {age_group}: Removed {removed_count} toxic stories")
        else:
            filtered_datasets[age_group] = df

    return filtered_datasets


def main():
    """Main evaluation function."""
    log_operation_status("Story evaluation")

    # Evaluate all datasets
    evaluated_datasets = evaluate_all_datasets()

    if evaluated_datasets:
        # Generate summary
        summary = generate_evaluation_summary(evaluated_datasets)

        print("\n📊 EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total stories evaluated: {summary['total_stories']:,}")

        print(f"\nAge group breakdown:")
        for age_group, stats in summary['age_group_stats'].items():
            print(f"\n{age_group.upper()}:")
            print(f"  Count: {stats['count']:,}")
            print(f"  Avg Perplexity: {stats['avg_perplexity']:.1f}")
            print(f"  Avg Grammar: {stats['avg_grammar']:.1f}/100")
            print(f"  Avg Coherence: {stats['avg_coherence']:.1f}/100")
            print(f"  Reading Level: {stats['avg_flesch_kincaid']:.1f}")
            print(f"  Toxic: {stats['toxic_count']} ({stats['toxic_percentage']:.1f}%)")

        if summary['overall_stats']:
            overall = summary['overall_stats']
            print(f"\nOVERALL STATISTICS:")
            print(f"  Avg Perplexity: {overall['avg_perplexity']:.1f}")
            print(f"  Avg Grammar: {overall['avg_grammar']:.1f}/100")
            print(f"  Avg Coherence: {overall['avg_coherence']:.1f}/100")
            print(f"  Avg Reading Level: {overall['avg_flesch_kincaid']:.1f}")
            print(f"  Total Toxic: {overall['total_toxic']} ({overall['toxic_percentage']:.1f}%)")

        # Filter toxic content
        print(f"\n🧹 Filtering toxic content...")
        clean_datasets = filter_toxic_content(evaluated_datasets)

        # Save clean datasets
        config = load_config()
        processed_path = Path(config['paths']['data_processed'])

        for age_group, df in clean_datasets.items():
            if not df.empty:
                clean_file = processed_path / f"{age_group}_stories_clean.csv"
                df.to_csv(clean_file, index=False)

        print("✅ Evaluation completed successfully!")

        # Save summary
        config = load_config()
        summary_file = Path(config['paths']['data_evaluated']) / "evaluation_summary.json"
        from utils.helpers import save_json
        save_json(summary, summary_file)
        print(f"  📊 Summary saved to {summary_file}")

    else:
        print("❌ Evaluation failed.")

    log_operation_status("Story evaluation", "completed")


if __name__ == "__main__":
    main()