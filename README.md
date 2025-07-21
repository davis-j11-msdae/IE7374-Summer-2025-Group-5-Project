# IE7374-Summer-2025-Group-5-Project
# Personalized Storytelling System

An AI-powered storytelling system that fine-tunes Mixtral 8x7B to generate age-appropriate stories with user history integration and comprehensive evaluation.

## 🎯 Project Overview

This system creates personalized stories for users of different ages by:
- Fine-tuning Mixtral 8x7B on children's and sci-fi story datasets
- Generating age-appropriate content for 4 age groups: child (0-5), kid (6-12), teen (13-17), adult (18+)
- Maintaining user story history with summarization
- Evaluating story quality, safety, and age appropriateness
- Supporting story continuation and interactive sessions

## 📁 Project Structure

```
personalized-storytelling/
├── configs/
│   ├── model_config.yaml          # Model and training configuration
│   └── deepspeed_config.json      # DeepSpeed optimization settings
├── utils/
│   ├── helpers.py                 # Common utility functions
│   ├── environment_check.py       # System validation
│   ├── colab_env_setup.py         # For use in properly preparing new colab runtimes
│   ├── environment_check.py       # System validation
│   ├── environment_check.py       # System validation
│   ├── generate_users.py          # User account generator
│   ├── download_data.py           # Data and model downloading
│   ├── data_loader.py             # Raw data processing
│   ├── data_tokenizer.py          # Dataset tokenization
│   ├── eval.py                    # Story evaluation and classification
│   ├── train.py                   # Model fine-tuning
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
│   ├── mixtral-8x7b		   # Base Model
│   ├── story_llm		   # Fine-Tuned Model
├── outputs/
│   ├── user_history/              # Individual user story histories
│   └── samples/                   # Sample evaluation results
├── logs/                          # System logs
├── src/
│   └── Full.py                    # Full control interface
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
## Google Colab
# Option 1: Upload colab_env_setup.py to Colab
		run python exec(open("colab_env_setup.py").read())
# Option 2: Upload colab_env_setup.py to Google Drive, run 
    		for root, dirs, files in os.walk("/content/drive"):
        		if "colab_env_setup.py" in files:
            			exec(open(os.path.join(root, "colab_env_setup.py")).read())
#Register Secret Keys
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

## Local/Other
# Clone the repository
git clone https://github.com/davis-j11-msdae/IE7374-Summer-2025-Group-5-Project.git

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

### 2. Run Main System

exec(open('src/full.py').read())

Follow the menu options in order:
1. Check Environment
2. Download Data and Models
3. Process Raw Data
4. Tokenize Data
5. Evaluate Data (optional)
6. Train Model
7. Process Sample Stories
8. Interactive Story Creation

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)

Key settings:
- **Model**: Mixtral 8x7B base model
- **Training**: 3 epochs, batch size 4, learning rate 2e-5
- **Age Groups**: Child (0-5), Kid (6-12), Teen (13-17), Adult (18+)
- **Evaluation**: Perplexity, grammar, coherence, toxicity checking

### DeepSpeed Configuration (`configs/deepspeed_config.json`)

Optimized for A100 GPU training with:
- ZeRO Stage 2 optimization
- CPU offloading for optimizer
- FP16 mixed precision
- Gradient accumulation

## 📊 Data Pipeline

### 1. Download Data
- **Children's Stories**: Kaggle fairy tales corpus
- **Sci-Fi Stories**: Kaggle science fiction corpus
- **Base Model**: Mixtral 8x7B from Hugging Face

### 2. Data Processing
- Extract individual stories from text files
- Clean and filter content
- Assign age groups based on content type
- Generate statistics and summaries

### 3. Tokenization
- Format stories with age-appropriate instructions
- Tokenize using Mixtral tokenizer
- Create train/validation/test splits (80/10/10)

### 4. Evaluation (Optional)
- **Quality Metrics**: Grammar and coherence scoring via OpenAI API
- **Readability**: Flesch-Kincaid grade level analysis
- **Safety**: Toxicity detection using Detoxify
- **Age Appropriateness**: Automated classification

## 🎭 Story Generation

### User Authentication
- 20 pre-generated users (5 per age group)
- Username format: `child_1`, `kid_1`, `teen_1`, `adult_1`
- Default password: `test`

### Story Features
- **Age-Appropriate Content**: Automatic adjustment based on user age
- **History Integration**: Stories can reference previous narratives
- **Story Continuation**: Extend existing stories seamlessly
- **Quality Assurance**: Automatic filtering of inappropriate content

### Interactive Sessions
- User login with age verification
- Create new stories or continue existing ones
- View and manage story history
- Save stories with custom titles

## 📈 Sample Stories

The system includes 10 predefined sample prompts:
- **2 prompts each** for child, kid, teen, and adult users
- **2 continuation prompts** demonstrating history integration
- **Comprehensive evaluation** of all generated content
- **Results saved** in `outputs/samples/`

## 🛡️ Safety and Quality

### Content Filtering
- **Toxicity Detection**: Automatic filtering using Detoxify
- **Age Verification**: Stories matched to appropriate age groups
- **Title Validation**: User-provided titles checked for appropriateness
- **Retry Logic**: Automatic regeneration if content fails checks

### Quality Metrics
- **Perplexity**: Measured using GPT-2
- **Grammar/Coherence**: Scored via OpenAI API
- **Readability**: Flesch-Kincaid grade level
- **Length Validation**: Stories within appropriate word limits

## 💾 User History

### Features
- **Automatic Summarization**: BART-large-CNN for story summaries
- **Title Management**: Suggested and custom titles
- **Story Continuation**: Two modes - update original or save as new
- **Statistics Tracking**: Word counts, creation dates, user preferences

### Storage Format
- Individual JSON files per user in `outputs/user_history/`
- Maximum 5 stories per user (configurable)
- Rich metadata including prompts, evaluations, timestamps

## 🖥️ Hardware Requirements

### Minimum (Training)
- **GPU**: 40GB+ VRAM (A100 recommended)
- **RAM**: 32GB system memory
- **Storage**: 100GB free space
- **CUDA**: Compatible GPU drivers

### Inference
- **GPU**: 16GB+ VRAM (RTX 4090, A6000)
- **RAM**: 16GB system memory
- **Storage**: 60GB for models

## 🔍 Monitoring and Logs

### Status Tracking
- Real-time operation status updates
- Progress bars for long-running operations
- Comprehensive error reporting
- Performance metrics collection

### Cache Management
- Automatic detection of existing processed data
- User prompts for cache overwrite decisions
- Efficient reuse of previous work
- Storage optimization

## 🧪 Testing and Validation

### Sample Evaluation
- 10 comprehensive test prompts
- Coverage of all age groups
- History continuation testing
- Quality and safety validation

### Environment Checks
- Dependency verification
- GPU availability confirmation
- API key validation
- Directory structure verification

## 📚 Usage Examples

### Generate a Story
```bash
python main.py
# Select option 8: Interactive Story Creation
# Login with username: child_1, password: test
# Create new story with prompt: "A magical adventure"
```

### Process Sample Stories
```bash
python main.py
# Select option 7: Process Sample Stories
# Review generated stories and evaluation metrics
```

### Continue a Story
```bash
# After creating initial stories
# Select "Continue existing story"
# Choose story from history
# Provide continuation prompt
```

## 🔧 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size in config
2. **Download Failures**: Check internet connection and API keys
3. **Model Loading Errors**: Verify model files are complete
4. **Evaluation Failures**: Ensure OpenAI API key is valid

### Performance Optimization
- Use DeepSpeed for memory efficiency
- Enable gradient checkpointing
- Optimize batch sizes for your hardware
- Use FP16 precision when possible

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Ensure all evaluations pass
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the Mixtral 8x7B model
- **Hugging Face** for transformers and datasets
- **Kaggle** for story datasets
- **OpenAI** for evaluation APIs
- **Microsoft** for DeepSpeed optimization

---

*Built for creative storytelling and AI research*