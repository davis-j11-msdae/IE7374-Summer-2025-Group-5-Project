# IE7374-Summer-2025-Group-5-Project

# Personalized Storytelling System

A comprehensive AI-powered storytelling system that fine-tunes Mistral 8x7B to generate age-appropriate stories with user history integration and comprehensive evaluation.

## Project Structure

```
project-root/
├── configs/
│   └── model_config.yaml      # Model and training configuration
├── utils/
│   └── helpers.py             # Common utility functions
├── src/
│   ├── eval.py               # Text evaluation and classification
│   ├── data_loader.py        # Dataset loading and preprocessing
│   ├── train.py              # Model fine-tuning
│   ├── model_runner.py       # Story generation with history
│   ├── history.py            # User history management
│   └── samples.py            # Sample evaluation pipeline
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Processed datasets
├── outputs/
│   ├── user_history/         # User story histories
│   └── *.pkl                 # Trained models
├── requirements.txt          # Python dependencies
├── main.py                   # Main pipeline script
└── README.md                 # This file
```

## Features

### 🎯 Core Functionality
- **Age-Appropriate Story Generation**: Automatically adjusts content and complexity for children, teens, and adults
- **User History Integration**: Learns from previous stories to maintain consistency and continuity
- **Story Continuation**: Can continue existing stories seamlessly
- **Multi-Model Support**: Supports both unified and age-specific fine-tuned models

### 📊 Comprehensive Evaluation
- **Perplexity Analysis**: 3-bucket classification (low, medium, high)
- **Grammar & Coherence Scoring**: LLM-based evaluation with percentage scores
- **Age Appropriateness**: Flesch-Kincaid readability assessment
- **Content Safety**: Toxicity detection using Detoxify
- **Text Statistics**: Length, word count, sentence analysis

### 🔄 Smart History Management
- **Automatic Summarization**: Uses BART-large-CNN for story summarization
- **Relevance Matching**: Finds related historical stories for context
- **Storage Management**: Maintains configurable history length limits
- **Export Capabilities**: JSON and text format exports

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd personalized-storytelling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Data Setup

Download the required datasets from Kaggle:
- [Children Stories Text Corpus](https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus)
- [Sci-Fi Stories Text Corpus](https://www.kaggle.com/datasets/jannesklaas/scifi-stories-text-corpus)

Place the CSV files in `data/raw/` directory.

### Running the Complete Pipeline

```bash
# Run the complete pipeline
python main.py

# Run specific steps
python main.py --step data      # Data processing only
python main.py --step train     # Training only  
python main.py --step evaluate  # Evaluation only

# Skip training (use existing model)
python main.py --skip-training
```

### Individual Module Usage

#### Data Processing
```bash
python src/data_loader.py
```

#### Model Training
```bash
python src/train.py
```

#### Story Generation
```python
from src.model_runner import StoryModelRunner

runner = StoryModelRunner()
runner.load_trained_model()

result = runner.generate_story(
    user_id="user_123",
    age=10,
    approximate_length="medium",
    prompt="A magical adventure in space"
)

print(result['story'])
```

#### Sample Evaluation
```bash
python src/samples.py
```

## Configuration

The system is configured via `configs/model_config.yaml`:

### Model Settings
- **base_model**: Mistral 8x7B model identifier
- **temperature**: Creativity level (0.7 default)
- **max_length**: Maximum generation length

### Training Parameters
- **batch_size**: Training batch size (4 default)
- **learning_rate**: Learning rate (2e-5 default)
- **epochs**: Training epochs (3 default)

### Evaluation Thresholds
- **perplexity_buckets**: [20, 50, 100] for low/medium/high classification
- **flesch_kincaid_ranges**: Age group reading level ranges

## Sample Data

The system includes predefined sample data for testing:

### Users and Prompts
- **Child User 1** (Age 7): "A friendly dragon who helps children"
- **Teen User 1** (Age 15): "A high school student discovers they have superpowers"
- **Adult User 1** (Age 28): "A detective solving a mysterious case in a small town"
- **Adult User 2** (Age 35): "A space explorer finds an abandoned alien city"
- **Child User 2** (Age 9): Multiple prompts including history continuation
- **Adult User 3** (Age 42): Multiple prompts including history continuation

## Output Examples

### Generated Story Report
```
==============================================================
STORY GENERATION AND EVALUATION REPORT
==============================================================
Total Users: 6
Total Prompts: 10
Successful Generations: 10
Failed Generations: 0

EVALUATION STATISTICS
------------------------------
Average Grammar Score: 87.50/100
Average Coherence Score: 89.20/100
Average Flesch-Kincaid Score: 8.45
Toxic Stories Detected: 0

AGE APPROPRIATENESS BREAKDOWN
-----------------------------------
Child: 4 stories
Teen: 1 stories
Adult: 5 stories
```

### Story Generation Result
```python
{
    'success': True,
    'story': 'Once upon a time, in a magical forest...',
    'user_id': 'child_user_1',
    'age': 7,
    'age_group': 'child',
    'length_requested': 'short',
    'actual_length': 245,
    'used_history': False,
    'continue_story': False
}
```

## Advanced Features

### History-Based Generation
The system automatically finds and incorporates relevant historical stories:

```python
# Generate story with history context
result = runner.generate_story(
    user_id="user_123",
    age=10,
    prompt="Another adventure with the dragon",
    use_history=True
)
```

### Story Continuation
Continue existing stories seamlessly:

```python
# Continue the last story
continuation = runner.continue_story(
    user_id="user_123",
    age=10,
    continuation_prompt="The dragon discovered a hidden treasure"
)
```

### User Statistics
Track user engagement and preferences:

```python
stats = runner.get_user_story_statistics("user_123")
# Returns: total_stories, avg_length, last_story_date, etc.
```

## Evaluation Metrics

### Text Quality Assessment
- **Grammar Score**: 0-100 scale via GPT-3.5-turbo
- **Coherence Score**: 0-100 scale via GPT-3.5-turbo
- **Perplexity**: Calculated using GPT-2, categorized into buckets

### Age Appropriateness
- **Flesch-Kincaid Grade Level**: Automated readability assessment
- **Age Group Classification**: Child (0-5.9), Teen (6.0-12.9), Adult (13.0+)

### Content Safety
- **Toxicity Detection**: Binary classification using Detoxify
- **Detailed Scores**: Toxicity, severe_toxicity, obscene, threat, insult, identity_attack

## Performance Considerations

### Memory Requirements
- **Training**: Requires GPU with 24GB+ VRAM for Mistral 8x7B
- **Inference**: 16GB+ VRAM recommended
- **Batch Processing**: Adjust batch sizes based on available memory

### Optimization Tips
- Use gradient checkpointing during training
- Enable mixed precision (FP16) for efficiency
- Process evaluations in batches for large datasets

## Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Ensure model files exist
ls outputs/trained_model.pkl

# Check GPU memory
nvidia-smi
```

#### API Rate Limits
```python
# For OpenAI API calls, implement retry logic
# Adjust batch sizes for evaluation
```

#### Memory Issues
```yaml
# In model_config.yaml, reduce batch sizes
training:
  batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 16  # Increase to maintain effective batch size
```

### Debug Mode
Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Mistral AI** for the base language model
- **Hugging Face** for the transformers library
- **Kaggle** for the story datasets
- **OpenAI** for evaluation APIs

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation
