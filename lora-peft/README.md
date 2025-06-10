# LoRA/PEFT Implementation

This project implements Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) based on the checklist in `LoRA - PEFT.md`.

## 📋 Complete Implementation Checklist

Based on the original checklist, this implementation covers:

### ✅ 1. Environment Setup
- Python >= 3.8 support
- Virtual environment configuration
- PyTorch with GPU support
- All required libraries (transformers, datasets, peft, accelerate, bitsandbytes, etc.)

### ✅ 2. Data Preparation
- Hugging Face Hub dataset loading
- Custom dataset support (JSON, CSV, JSONL)
- Comprehensive preprocessing for different tasks
- Train/validation splitting

### ✅ 3. Base Model Selection
- Support for popular models (BERT, LLaMA, Mistral, Falcon, T5, GPT)
- Automatic model and tokenizer loading
- Quantization support (4-bit, 8-bit)

### ✅ 4. PEFT Configuration
- LoRA, Prefix Tuning, Prompt Tuning, IA3 support
- Automatic target module detection
- Configurable parameters (rank, alpha, dropout, etc.)

### ✅ 5. Model Training
- Comprehensive training pipeline
- Multiple optimization strategies
- Progress tracking and logging
- Early stopping and checkpointing

### ✅ 6. Model Saving & Loading
- PEFT model serialization
- Checkpoint management
- Model merging capabilities

### ✅ 7. Evaluation & Deployment
- Comprehensive metrics (accuracy, F1, BLEU, ROUGE, etc.)
- Model comparison tools
- Production-ready inference pipeline

### ✅ 8. Advanced Features
- QLoRA support with quantization
- Multiple PEFT method experimentation
- Batch processing and optimization

## 📁 Project Structure

```
lora-peft/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── setup.py                 # Package setup
├── cli.py                   # Command-line interface
├── config/                  # Configuration files
│   ├── __init__.py
│   ├── model_config.py      # Model configurations
│   └── training_config.py   # Training configurations
├── data/                    # Data processing
│   ├── __init__.py
│   ├── data_loader.py       # Data loading utilities
│   └── preprocessing.py     # Data preprocessing
├── models/                  # Model implementations
│   ├── __init__.py
│   ├── base_model.py        # Base model wrapper
│   └── peft_model.py        # PEFT model implementation
├── training/                # Training scripts
│   ├── __init__.py
│   ├── trainer.py           # Training logic
│   └── utils.py             # Training utilities
├── evaluation/              # Evaluation scripts
│   ├── __init__.py
│   ├── evaluator.py         # Model evaluation
│   └── metrics.py           # Evaluation metrics
├── inference/               # Inference scripts
│   ├── __init__.py
│   └── pipeline.py          # Inference pipeline
├── examples/                # Example scripts
│   ├── text_classification.py
│   ├── text_generation.py
│   └── question_answering.py
└── notebooks/               # Jupyter notebooks
    ├── 01_setup_environment.ipynb
    ├── 02_data_preparation.ipynb
    ├── 03_model_training.ipynb
    └── 04_evaluation_inference.ipynb
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd lora-peft

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Basic Usage

#### Text Classification Example
```bash
python examples/text_classification.py
```

#### Text Generation Example
```bash
python examples/text_generation.py
```

#### Question Answering Example
```bash
python examples/question_answering.py
```

### 3. Command Line Interface

```bash
# Train a model
python cli.py train --task classification --model bert-base-uncased --dataset imdb

# Evaluate a model
python cli.py evaluate --model-path ./results/model --dataset imdb

# Run inference
python cli.py infer --model-path ./results/model --text "This movie is great!"
```

### 4. Jupyter Notebooks

Start with the interactive tutorials:

```bash
jupyter notebook notebooks/01_setup_environment.ipynb
```

## 📚 Features

### Core PEFT Methods
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with low-rank matrices
- **Prefix Tuning**: Prepend trainable prefix tokens
- **Prompt Tuning**: Learn soft prompts for tasks
- **IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)**

### Supported Models
- **BERT family**: bert-base-uncased, bert-large, roberta, distilbert
- **LLaMA family**: llama-7b, llama-13b, llama-30b, llama-65b
- **Mistral models**: mistral-7b, mixtral-8x7b
- **Falcon models**: falcon-7b, falcon-40b
- **T5 models**: t5-small, t5-base, t5-large
- **GPT models**: gpt2, gpt-neo, gpt-j

### Advanced Features
- **Quantization**: 4-bit and 8-bit quantization with bitsandbytes
- **QLoRA**: Quantized LoRA for memory-efficient training
- **Multi-GPU**: Distributed training support
- **Mixed Precision**: FP16/BF16 training
- **Gradient Checkpointing**: Memory optimization
- **Early Stopping**: Prevent overfitting
- **Comprehensive Metrics**: Task-specific evaluation metrics

## 🧠 Supported Tasks

### Text Classification
- Sentiment analysis
- Topic classification
- Intent detection
- Spam detection

### Text Generation
- Creative writing
- Code generation
- Instruction following
- Dialogue generation

### Question Answering
- Reading comprehension
- Factual QA
- Conversational QA

### Sequence-to-Sequence
- Summarization
- Translation
- Paraphrasing

## 📖 Documentation

### Tutorials
1. **Environment Setup**: `notebooks/01_setup_environment.ipynb`
2. **Data Preparation**: `notebooks/02_data_preparation.ipynb`
3. **Model Training**: `notebooks/03_model_training.ipynb`
4. **Evaluation & Inference**: `notebooks/04_evaluation_inference.ipynb`

### API Reference
- **Configuration**: `config/` - Model and training configurations
- **Data Processing**: `data/` - Dataset loading and preprocessing
- **Models**: `models/` - PEFT model implementations
- **Training**: `training/` - Training pipeline and utilities
- **Evaluation**: `evaluation/` - Metrics and evaluation tools
- **Inference**: `inference/` - Production inference pipeline

## 🔧 Configuration

### Model Configuration
```python
from config import ModelConfig

model_config = ModelConfig(
    model_name_or_path="bert-base-uncased",
    num_labels=2,
    max_length=512,
    use_quantization=True,
    quantization_bits=4
)
```

### PEFT Configuration
```python
from config import PEFTConfig
from peft import TaskType

peft_config = PEFTConfig(
    peft_type="LORA",
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)
```

### Training Configuration
```python
from config import TrainingConfig

training_config = TrainingConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-4,
    evaluation_strategy="steps",
    save_steps=500
)
```

## 🚀 Performance Tips

### Memory Optimization
- Use quantization for large models (4-bit recommended)
- Enable gradient checkpointing for memory-constrained environments
- Use smaller batch sizes with gradient accumulation
- Consider using DeepSpeed for very large models

### Training Optimization
- Start with smaller LoRA ranks (r=8-16) and increase if needed
- Use learning rates between 1e-4 and 5e-4 for LoRA
- Enable mixed precision training (FP16/BF16)
- Use early stopping to prevent overfitting

### Inference Optimization
- Merge LoRA weights with base model for faster inference
- Use batch processing for multiple inputs
- Consider model quantization for deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the transformers and PEFT libraries
- Microsoft for the LoRA paper and implementation
- The open-source community for various model implementations

## 📞 Support

- Check the `notebooks/` for detailed tutorials
- Review `examples/` for working code samples
- Open an issue for bugs or feature requests

## 🔗 Related Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Original Checklist](../fine-tuning/LoRA%20-%20PEFT.md)
