# Personalized Storytelling System

An AI-powered storytelling system that fine-tunes Mistral 7B Instruct v0.3 to generate age-appropriate stories with user history integration and comprehensive evaluation.

## 🎯 Project Overview

This system creates personalized stories for users of different ages by:
- Fine-tuning Mistral 7B Instruct v0.3 on children's and sci-fi story datasets from Project Gutenberg
- Generating age-appropriate content for 4 age groups: child (0-5), kid (6-12), teen (13-17), adult (18+)
- Maintaining user story history with automatic summarization using Mistral
- Evaluating story quality, safety, and age appropriateness with integrated Mistral evaluation
- Supporting story continuation and interactive sessions
- Optimized hyperparameter tuning for efficient training

## 📁 Project Structure

```
IE7374-Summer-2025-Group-5-Project/
├── configs/
│   ├── model_config.yaml          # Model and training configuration
│   └── deepspeed_config.json      # DeepSpeed optimization settings
├── utils/
│   ├── helpers.py                 # Common utility functions
│   ├── environment_check.py       # System validation
│   ├── generate_users.py          # User account generator
│   ├── download_data.py           # Data and model downloading
│   ├── data_loader.py             # Raw data processing with evaluation
│   ├── data_tokenizer.py          # Dataset tokenization with stratification
│   ├── eval.py                    # Story evaluation using Mistral
│   ├── train.py                   # Model fine-tuning with quantization
│   ├── hyperparameter_tuning.py   # Automated hyperparameter optimization
│   ├── history.py                 # User history management with Mistral summarization
│   ├── model_runner.py            # Story generation with authentication
│   └── samples.py                 # Sample evaluation pipeline
├── data/
│   ├── raw/                       # Downloaded Project Gutenberg datasets
│   ├── processed/                 # Cleaned and age-grouped stories
│   ├── tokenized/                 # Tokenized datasets for training
│   ├── evaluated/                 # Quality-assessed stories
│   └── users/                     # User authentication data
├── models/
│   ├── mistral-7b-base/           # Base Mistral 7B Instruct v0.3 model
│   └── tuned_story_llm/           # Fine-tuned storytelling model
├── outputs/
│   ├── user_history/              # Individual user story histories
│   └── samples/                   # Sample evaluation results
├── logs/                          # System logs
├── hyperparameter_results/        # Hyperparameter tuning results
├── src/
│   └── full.py                    # Main control interface with authentication
├── colab_training.ipynb           # Google Colab training notebook
├── colab_env_setup.py             # Colab environment setup script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd IE7374-Summer-2025-Group-5-Project

# Install dependencies
pip install -r requirements.txt

# Create environment file (optional - for Hugging Face token)
cat > .env << EOF
HF_TOKEN=your_huggingface_token
EOF
```

### 2. Run the System

```bash
# Navigate to src directory and run main control script
cd src
python full.py
```

### 3. System Workflow

**Admin Workflow** (requires admin login: username=admin, password=admin):
1. Check Environment
2. Download Data and Models  
3. Generate Users File
4. Process and Evaluate Data
5. Tokenize Data
6. Hyperparameter Tuning
7. Train Model
8. Process Sample Stories
9. Interactive Story Creation

**User Workflow** (regular user login: username format like child_1, password=test):
- Direct access to Interactive Story Creation

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)

Key settings:
- **Model**: Mistral 7B Instruct v0.3 from Hugging Face
- **Training**: 1-5 epochs (tuning vs production), configurable batch sizes, learning rates
- **Quantization**: 4-bit quantization with NF4 for memory efficiency
- **LoRA**: Rank 16-64, alpha = 2x rank, targeting key transformer modules
- **Age Groups**: Child (0-5), Kid (6-12), Teen (13-17), Adult (18+)
- **Data Sources**: Project Gutenberg categories with 2000+ download threshold
- **Evaluation**: Mistral-based quality assessment with toxicity filtering

### Hardware Optimization

The system automatically adapts to available hardware:
- **A100 (40GB+)**: Full batch sizes, no gradient accumulation
- **10GB GPUs**: Reduced batch sizes with gradient accumulation
- **Quantization**: 4-bit loading for memory efficiency
- **DeepSpeed**: Optional for large-scale training

## 📊 Data Pipeline

### 1. Data Download (`utils/download_data.py`)
- **Project Gutenberg Stories**: Automated download from multiple categories
  - Science Fiction & Fantasy (bookshelf 638)
  - Children's & Young Adult Literature (bookshelf 636)
  - Adventure Stories (bookshelf 644)
  - Mythology & Folklore (bookshelf 646)
  - Humor (bookshelf 641)
  - Short Stories (bookshelf 634)
  - Fairy Tales (bookshelf 216)
- **Base Model**: Mistral 7B Instruct v0.3 from Hugging Face Hub
- **Copyright Validation**: Automatic public domain verification

### 2. Data Processing (`utils/data_loader.py`)
- Extract individual stories from downloaded collections
- Clean and filter content for quality using regex patterns
- Assign age groups based on Flesch-Kincaid reading level analysis
- **Integrated Evaluation**: Quality, safety, and appropriateness checking using Mistral
- Generate comprehensive statistics and summaries

### 3. Tokenization (`utils/data_tokenizer.py`)
- Format stories with age-appropriate instructions using Mistral chat template
- Tokenize using Mistral tokenizer with padding to max_length
- Create stratified train/validation/test splits (80/10/10) preserving source distribution
- Select hyperparameter tuning samples (5% by default)
- Generate tokenization statistics and validation reports

### 4. Hyperparameter Tuning (`utils/hyperparameter_tuning.py`)
- **Sequential Optimization**: Learning rate → LoRA rank → Effective batch size → Dropout → Weight decay → Warmup ratio
- **Hardware-Aware**: Automatically configures batch sizes based on GPU memory
- **Resumable**: Can continue from interruptions using persistent state
- **Config-Driven**: Search spaces defined in model_config.yaml

### 5. Model Training (`utils/train.py`)
- Fine-tune Mistral 7B Instruct v0.3 on processed story datasets
- 4-bit quantization with LoRA adapters for memory efficiency
- Early stopping based on validation loss improvements
- Automatic model saving and evaluation metrics
- Supports resuming from checkpoints

## 🎭 Story Generation Features

### User Authentication
- **Admin Users**: Full system access including training and configuration
- **Regular Users**: Story creation and history management
- **Pre-generated Users**: 20 users across age groups (username format: `child_1`, `kid_1`, etc.)
- **Default Credentials**: Password `test` for regular users, `admin` for admin user

### Story Generation
- **Age-Appropriate Content**: Automatic adjustment based on user age using reading level analysis
- **History Integration**: Stories can reference previous narratives through Mistral summarization
- **Story Continuation**: Seamlessly extend existing stories with context awareness
- **Quality Assurance**: Automatic filtering using Mistral evaluation and Detoxify
- **Real-time Evaluation**: Grammar, coherence, and safety scoring

### Interactive Features
- User login with age verification
- Create new stories or continue existing ones
- View and manage personal story history with Mistral-generated summaries
- Save stories with AI-suggested or custom titles
- Delete unwanted stories from history

## 📈 Evaluation and Quality Control

### Content Filtering
- **Toxicity Detection**: Automatic filtering using Detoxify (threshold: 0.5)
- **Age Verification**: Stories matched to appropriate age groups using Flesch-Kincaid scoring
- **Title Validation**: User-provided titles checked for appropriateness using Detoxify
- **Retry Logic**: Automatic regeneration if content fails quality checks (max 3 attempts)

### Quality Metrics (Using Mistral Evaluation)
- **Grammar/Coherence**: Scored via fine-tuned Mistral model with optimized prompts
- **Perplexity**: Measured using GPT-2 for text naturalness (bucketed: low/medium/high)
- **Readability**: Flesch-Kincaid grade level analysis with age-appropriate ranges
- **Safety**: Comprehensive toxicity detection across multiple categories
- **Length Validation**: Stories within appropriate word count limits (100-10,000 words)

## 💾 User History Management

### Features (`utils/history.py`)
- **Automatic Summarization**: Mistral-based story summarization (2-3 sentences, max 150 chars)
- **Title Management**: AI-suggested titles with custom override option using Mistral
- **Story Continuation**: Two modes - update original or save as new story
- **Statistics Tracking**: Word counts, creation dates, reading analytics
- **Storage Optimization**: Maximum 5 stories per user with automatic cleanup

### Storage Format
- Individual JSON files per user in `outputs/user_history/`
- Rich metadata including prompts, evaluations, timestamps
- Cross-session persistence with data validation

## 🖥️ Hardware Requirements

### Training Requirements
- **GPU**: 10GB+ VRAM (RTX 3080, RTX 4090, A100, etc.)
- **RAM**: 16GB+ system memory
- **Storage**: 60GB+ free space for models and data
- **CUDA**: Compatible GPU drivers and toolkit

### Inference Requirements  
- **GPU**: 8GB+ VRAM (RTX 3070, RTX 4070, or equivalent)
- **RAM**: 12GB+ system memory
- **Storage**: 40GB+ for base and fine-tuned models

### Google Colab Support
- **Colab Pro/Pro+**: Recommended for A100 access
- **Environment Setup**: Automated via `colab_env_setup.py`
- **Notebook**: `colab_training.ipynb` for cloud training

## 🔍 System Monitoring

### Status Tracking
- Real-time operation status with timestamps
- Progress bars for long-running operations (>1 minute) using tqdm
- Comprehensive error reporting with context
- Performance metrics and resource usage monitoring

### Cache Management
- Automatic detection of existing processed data
- User prompts for cache overwrite decisions
- Efficient reuse of previous work and computations
- Storage optimization and cleanup utilities

## 🧪 Sample Stories and Testing

### Automated Testing (`utils/samples.py`)
- 10 comprehensive test prompts across all age groups
- Story continuation testing with context preservation
- Quality and safety validation for all generated content using Mistral evaluation
- Detailed evaluation reports with metrics and statistics

### Sample Categories
- **2 prompts each** for child, kid, teen, and adult age groups
- **2 continuation prompts** demonstrating history integration
- **Comprehensive evaluation** including grammar, coherence, and safety
- **Results archived** in `outputs/samples/` with timestamps

## 📚 Usage Examples

### Complete System Setup
```bash
cd src
python full.py
# Login as admin (username: admin, password: admin)
# Follow workflow: 1 → 2 → 9 → 3 → 4 → 5 → 6 → 7 → 8
```

### User Story Creation
```bash
cd src  
python full.py
# Login as regular user (e.g., username: child_1, password: test)
# Create and manage personal stories
```

### Direct Component Access
```bash
# Check system environment
cd utils && python environment_check.py

# Generate user accounts  
cd utils && python generate_users.py

# Download data only
cd utils && python download_data.py
```

### Google Colab Usage
```python
# Upload colab_env_setup.py to Colab, then run:
exec(open('colab_env_setup.py').read())

# Or if files are in Google Drive:
import os
if os.path.exists("colab_env_setup.py"):
    exec(open("colab_env_setup.py").read())
else:
    for root, dirs, files in os.walk("/content/drive"):
        if "colab_env_setup.py" in files:
            exec(open(os.path.join(root, "colab_env_setup.py")).read())
            break

# Then run the main system:
exec(open('src/full.py').read())
```

## 🔧 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: System automatically uses 4-bit quantization and gradient accumulation
2. **Download Failures**: Verify internet connection and optional HF_TOKEN for gated models
3. **Model Loading Errors**: Ensure complete model downloads and sufficient disk space
4. **Tokenization Issues**: Check for duplicate columns in processed data (automatically cleaned)

### Performance Optimization
- **Quantization**: 4-bit loading enabled by default for memory efficiency
- **Batch Size Adaptation**: Automatic adjustment based on GPU memory
- **LoRA Configuration**: Efficient fine-tuning with minimal memory footprint
- **Gradient Accumulation**: Hardware-aware configuration for consistent effective batch sizes

## 🤝 Contributing

1. Fork the repository
2. Create feature branch with descriptive name
3. Add comprehensive tests for new functionality
4. Ensure all existing tests and evaluations pass
5. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the Mistral 7B Instruct v0.3 foundation model
- **Hugging Face** for transformers library and model hosting
- **Project Gutenberg** for public domain literature datasets
- **Meta** for PEFT (LoRA) implementation
- **Microsoft** for DeepSpeed optimization framework
- **Google** for Colab platform support

---

*Built for educational research in AI-powered creative storytelling with age-appropriate content generation*