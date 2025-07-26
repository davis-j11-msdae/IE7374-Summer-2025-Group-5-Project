# Personalized Storytelling System

An AI-powered storytelling system that fine-tunes Mixtral 8x7B to generate age-appropriate stories with user history integration and comprehensive evaluation.

## 🎯 Project Overview

This system creates personalized stories for users of different ages by:
- Fine-tuning Mixtral 8x7B on children's and sci-fi story datasets from Project Gutenberg
- Generating age-appropriate content for 4 age groups: child (0-5), kid (6-12), teen (13-17), adult (18+)
- Maintaining user story history with automatic summarization
- Evaluating story quality, safety, and age appropriateness
- Supporting story continuation and interactive sessions

## 📁 Project Structure

```
IE7374-Summer-2025-Group-5-Project/
├── configs/
│   └── model_config.yaml          # Model and training configuration
├── utils/
│   ├── helpers.py                 # Common utility functions
│   ├── environment_check.py       # System validation
│   ├── generate_users.py          # User account generator
│   ├── download_data.py           # Data and model downloading
│   ├── data_loader.py             # Raw data processing
│   ├── data_tokenizer.py          # Dataset tokenization
│   ├── eval.py                    # Story evaluation and classification
│   ├── train.py                   # Model fine-tuning
│   ├── hyperparameter_tuning.py   # Optimization of Model Hyperparameters
│   ├── history.py                 # User history management
│   ├── model_runner.py            # Story generation with authentication
│   └── samples.py                 # Sample evaluation pipeline
├── data/
│   ├── raw/                       # Downloaded datasets
│   ├── processed/                 # Cleaned and age-grouped stories
│   ├── tokenized/                 # Tokenized datasets for training
│   ├── evaluated/                 # Quality-assessed stories
│   └── users/                     # User authentication data
├── models/
│   ├── mixtral-7b-base/           # Base Mixtral model
│   └── tuned_story_llm/           # Fine-tuned storytelling model
├── outputs/
│   ├── user_history/              # Individual user story histories
│   └── samples/                   # Sample evaluation results
├── logs/                          # System logs
├── src/
│   └── Full.py                    # Main control interface
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd IE7574-Summer-2025-Group-5-Project

# Install dependencies
pip install -r requirements.txt

# Create environment file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
EOF
```

### 2. Run the System

```bash
# Navigate to src directory and run main control script
cd src
python Full.py
```

### 3. System Workflow

**Admin Workflow** (requires admin login):
1. Check Environment
2. Download Data and Models  
3. Generate Users File
4. Process and Evaluate Data
5. Tokenize Data
6. Train Model
7. Process Sample Stories
8. Interactive Story Creation

**User Workflow** (regular user login):
- Direct access to Interactive Story Creation

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)

Key settings:
- **Model**: Mixtral 8x7B base model from Hugging Face
- **Training**: 3 epochs, batch size 4, learning rate 2e-5, DeepSpeed optimization
- **Age Groups**: Child (0-5), Kid (6-12), Teen (13-17), Adult (18+)
- **Data Sources**: Project Gutenberg categories with configurable minimum downloads
- **Evaluation**: Perplexity, grammar, coherence, toxicity, and readability metrics

### DeepSpeed Configuration (`configs/deepspeed_config.json`)

Optimized for A100 GPU training:
- ZeRO Stage 2 optimization
- CPU optimizer offloading
- FP16 mixed precision
- Gradient accumulation and clipping

## 📊 Data Pipeline

### 1. Data Download (`utils/download_data.py`)
- **Project Gutenberg Stories**: Automated download from multiple categories
  - Science Fiction & Fantasy
  - Children's & Young Adult Literature  
  - Adventure Stories
  - Fairy Tales & Folk Tales
  - Mythology & Folklore
  - Humor and Short Stories
- **Base Model**: Mixtral 8x7B from Hugging Face Hub
- **Copyright Validation**: Automatic public domain verification

### 2. Data Processing (`utils/data_loader.py`)
- Extract individual stories from downloaded collections
- Clean and filter content for quality
- Assign age groups based on source categories
- **Integrated Evaluation**: Quality, safety, and appropriateness checking
- Generate comprehensive statistics and summaries

### 3. Tokenization (`utils/data_tokenizer.py`)
- Format stories with age-appropriate instructions
- Tokenize using Mixtral tokenizer with padding
- Create train/validation/test splits (80/10/10)
- Generate tokenization statistics and validation

### 4. Model Training (`utils/train.py`)
- Fine-tune Mixtral 8x7B on processed story datasets
- DeepSpeed optimization for memory efficiency
- Automatic model saving and evaluation metrics
- Comprehensive training statistics and validation

## 🎭 Story Generation Features

### User Authentication
- **Admin Users**: Full system access including training and configuration
- **Regular Users**: Story creation and history management
- **Pre-generated Users**: 20 users across age groups (username format: `child_1`, `kid_1`, etc.)
- **Default Credentials**: Password `test` for regular users, `admin` for admin user

### Story Generation
- **Age-Appropriate Content**: Automatic adjustment based on user age
- **History Integration**: Stories can reference previous narratives
- **Story Continuation**: Seamlessly extend existing stories
- **Quality Assurance**: Automatic filtering of inappropriate content
- **Real-time Evaluation**: Grammar, coherence, and safety scoring

### Interactive Features
- User login with age verification
- Create new stories or continue existing ones
- View and manage personal story history with summaries
- Save stories with suggested or custom titles
- Delete unwanted stories from history

## 📈 Evaluation and Quality Control

### Content Filtering
- **Toxicity Detection**: Automatic filtering using Detoxify
- **Age Verification**: Stories matched to appropriate age groups using Flesch-Kincaid scoring
- **Title Validation**: User-provided titles checked for appropriateness
- **Retry Logic**: Automatic regeneration if content fails quality checks

### Quality Metrics
- **Perplexity**: Measured using GPT-2 for text naturalness
- **Grammar/Coherence**: Scored via OpenAI GPT-3.5-turbo API
- **Readability**: Flesch-Kincaid grade level analysis
- **Safety**: Comprehensive toxicity detection across multiple categories
- **Length Validation**: Stories within appropriate word count limits

## 💾 User History Management

### Features (`utils/history.py`)
- **Automatic Summarization**: BART-large-CNN for concise story summaries
- **Title Management**: AI-suggested titles with custom override option
- **Story Continuation**: Two modes - update original or save as new story
- **Statistics Tracking**: Word counts, creation dates, reading analytics
- **Storage Optimization**: Maximum 5 stories per user with automatic cleanup

### Storage Format
- Individual JSON files per user in `outputs/user_history/`
- Rich metadata including prompts, evaluations, timestamps
- Cross-session persistence with data validation

## 🖥️ Hardware Requirements

### Training Requirements
- **GPU**: A100 with 40GB+ VRAM (required for Mixtral 8x7B)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free space for models and data
- **CUDA**: Compatible GPU drivers and toolkit

### Inference Requirements  
- **GPU**: 16GB+ VRAM (RTX 4090, A6000, or equivalent)
- **RAM**: 16GB+ system memory
- **Storage**: 60GB+ for base and fine-tuned models

## 🔍 System Monitoring

### Status Tracking
- Real-time operation status with timestamps
- Progress bars for long-running operations (>1 minute)
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
- Quality and safety validation for all generated content
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
python Full.py
# Login as admin (username: admin, password: admin)
# Follow workflow: 1 → 2 → 8 → 3 → 4 → 5 → 6 → 7
```

### User Story Creation
```bash
cd src  
python Full.py
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

## 🔧 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Use DeepSpeed config, reduce batch sizes
2. **Download Failures**: Verify internet connection and API credentials
3. **Model Loading Errors**: Ensure complete model downloads and sufficient disk space
4. **OpenAI API Issues**: Verify API key validity and rate limits

### Performance Optimization
- Enable DeepSpeed for memory-efficient training
- Use gradient checkpointing to reduce memory usage
- Optimize batch sizes for available hardware
- Leverage FP16 precision when supported

## 🤝 Contributing

1. Fork the repository
2. Create feature branch with descriptive name
3. Add comprehensive tests for new functionality
4. Ensure all existing tests and evaluations pass
5. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the Mixtral 8x7B foundation model
- **Hugging Face** for transformers library and model hosting
- **Project Gutenberg** for public domain literature datasets
- **OpenAI** for evaluation and quality assessment APIs
- **Microsoft** for DeepSpeed optimization framework
- **Meta** for BART summarization model

---

*Built for educational research in AI-powered creative storytelling*
