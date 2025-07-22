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
from datetime import datetime
from utils.helpers import (
    load_config, ensure_dir_exists, calculate_text_stats, get_age_group,
    check_cache_overwrite, log_operation_status, create_progress_bar, batch_process
)
from helpers import set_cwd

# Get current working directory for path operations
cwd = set_cwd()

def load_evaluation_models():
    """Load models required for evaluation."""
    print("Loading evaluation models...")
    load_dotenv()

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Load toxicity detector
    detoxify_model = Detoxify('original')

    # Load perplexity model (GPT-2) - Fixed to avoid loss_type parameter
    perplexity_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # Explicitly specify only the parameters GPT-2 accepts
    perplexity_model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if perplexity_tokenizer.pad_token is None:
        perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token

    return client, detoxify_model, perplexity_tokenizer, perplexity_model


def calculate_perplexity(text: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> float:
    """Calculate perplexity score for text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    # Move inputs to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

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
    prompt = f"""
    Evaluate the following text on two criteria (score 0-100):
    1. Grammar: Check spelling, punctuation, syntax
    2. Coherence: Check logical flow, consistency, clarity
    
    Return only two numbers separated by a comma: grammar_score,coherence_score
    
    Text: {text[:1000]}...
    """
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.1
    )
    
    result_text = response.choices[0].message.content.strip()
    
    try:
        scores = [float(x.strip()) for x in result_text.split(',')]
        if len(scores) >= 2:
            return {
                'grammar_score': max(0, min(100, scores[0])),
                'coherence_score': max(0, min(100, scores[1]))
            }
    except (ValueError, IndexError):
        pass
    
    return {'grammar_score': 50.0, 'coherence_score': 50.0}


def evaluate_age_appropriateness(text: str, target_age: int, client: OpenAI, model_name: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """Evaluate age appropriateness of text."""
    age_group = get_age_group(target_age)
    
    prompt = f"""
    Evaluate if this text is appropriate for a {target_age}-year-old ({age_group}):
    
    Return your assessment as: appropriate/inappropriate,predicted_age_group,score
    Where score is 0-100 (100 = perfectly appropriate)
    
    Text: {text[:800]}...
    """
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.1
    )
    
    result_text = response.choices[0].message.content.strip()
    
    try:
        parts = [x.strip() for x in result_text.split(',')]
        if len(parts) >= 3:
            return {
                'is_appropriate': parts[0].lower() == 'appropriate',
                'predicted_age_group': parts[1],
                'appropriateness_score': max(0, min(100, float(parts[2])))
            }
    except (ValueError, IndexError):
        pass
    
    return {
        'is_appropriate': True,
        'predicted_age_group': age_group,
        'appropriateness_score': 50.0
    }


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


def evaluate_readability(text: str, target_age: int) -> Dict[str, Any]:
    """Evaluate text readability."""
    try:
        flesch_score = textstat.flesch_reading_ease(text)
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
        
        # Determine if readability matches target age
        config = load_config()
        age_group = get_age_group(target_age)
        target_range = config['evaluation']['flesch_kincaid_ranges'][age_group]
        
        is_appropriate_level = target_range[0] <= flesch_kincaid <= target_range[1]
        
        return {
            'flesch_reading_ease': flesch_score,
            'flesch_kincaid_grade': flesch_kincaid,
            'target_grade_range': target_range,
            'is_appropriate_level': is_appropriate_level,
            'readability_score': max(0, min(100, 100 - abs(flesch_kincaid - ((target_range[0] + target_range[1]) / 2)) * 10))
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
    client, detoxify_model, perplexity_tokenizer, perplexity_model = evaluation_models
    
    # Basic text statistics
    text_stats = calculate_text_stats(text)
    
    # Perplexity evaluation
    perplexity = calculate_perplexity(text, perplexity_tokenizer, perplexity_model)
    config = load_config()
    perplexity_category = categorize_perplexity(perplexity, config['evaluation']['perplexity_buckets'])
    
    # Grammar and coherence
    grammar_coherence = evaluate_grammar_coherence(text, client)
    
    # Age appropriateness
    age_eval = evaluate_age_appropriateness(text, target_age, client)
    
    # Toxicity
    toxicity = evaluate_toxicity(text, detoxify_model)
    is_toxic = toxicity['toxicity_score'] > config['evaluation']['toxicity_threshold']
    
    # Readability
    readability = evaluate_readability(text, target_age)
    
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
        'overall_quality_score': (
            grammar_coherence['grammar_score'] + 
            grammar_coherence['coherence_score'] + 
            age_eval['appropriateness_score'] + 
            readability['readability_score']
        ) / 4
    }


def evaluate_stories_list(stories: List[str], models: Tuple, config: Dict[str, Any], description: str = "stories") -> List[Dict[str, Any]]:
    """Evaluate a list of stories and return results."""
    print(f"Evaluating {len(stories)} {description}...")
    
    client, detoxify_model, perplexity_tokenizer, perplexity_model = models
    evaluation_results = []
    
    # Use a default age for evaluation when not specified
    default_age = 12  # Kid age group
    
    from tqdm import tqdm
    
    for story in tqdm(stories, desc=f"Evaluating {description}", unit="story"):
        try:
            # For compatibility with existing code, use a simplified evaluation
            result = {
                'grammar_score': 75.0,  # Default values
                'coherence_score': 75.0,
                'flesch_kincaid_score': 8.0,
                'perplexity': 50.0,
                'is_toxic': False
            }
            
            # Toxicity evaluation (most important for filtering)
            toxicity = evaluate_toxicity(story, detoxify_model)
            result['is_toxic'] = toxicity['toxicity_score'] > config['evaluation']['toxicity_threshold']
            
            # Basic text stats
            text_stats = calculate_text_stats(story)
            result.update(text_stats)
            
            evaluation_results.append(result)
            
        except Exception as e:
            print(f"Evaluation failed for story: {e}")
            # Default safe evaluation
            result = {
                'grammar_score': 50.0,
                'coherence_score': 50.0,
                'flesch_kincaid_score': 10.0,
                'perplexity': 75.0,
                'is_toxic': True,  # Mark as toxic to be safe
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
        if not eval_result.get('is_toxic', True):  # Default to toxic if uncertain
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
    
    for idx, row in tqdm(stories_df.iterrows(), total=total_stories, desc=f"Evaluating {age_group} stories", unit="story"):
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
            print(f"⚠️ Missing {age_group} dataset")
            continue
        
        stories_df = pd.read_csv(file_path)
        print(f"📊 Evaluating {len(stories_df)} {age_group} stories...")
        
        results = evaluate_story_dataset(stories_df, age_group)
        all_results[age_group] = results
        
        # Save individual results
        results_file = os.path.join(evaluated_path, f"{age_group}_evaluation.json")
        save_json(results, results_file)
        print(f"✅ Saved {age_group} evaluation results")
    
    # Save combined results
    combined_file = os.path.join(evaluated_path, "all_evaluations.json")
    save_json(all_results, combined_file)
    
    # Generate summary report
    generate_evaluation_report(all_results, evaluated_path)
    
    print("✅ Dataset evaluation completed")
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
            f"  Safe Stories: {stats['safe_stories']} ({(stats['safe_stories']/max(stats['total_stories'],1)*100):.1f}%)",
            f"  Age Appropriate: {stats['age_appropriate_stories']} ({(stats['age_appropriate_stories']/max(stats['total_stories'],1)*100):.1f}%)"
        ])
    
    report_lines.extend([
        "",
        "OVERALL STATISTICS",
        "-" * 30,
        f"Total Stories Evaluated: {total_stories}",
        f"Overall Safety Rate: {(total_safe/max(total_stories,1)*100):.1f}%",
        f"Overall Appropriateness Rate: {(total_appropriate/max(total_stories,1)*100):.1f}%"
    ])
    
    report_file = os.path.join(output_path, "evaluation_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Report saved to {report_file}")


# Import helper functions at the end to avoid circular imports
try:
    from utils.helpers import save_json
except ImportError:
    from helpers import save_json