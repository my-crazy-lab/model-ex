# 🎯 Prefix Tuning & Prompt Tuning Implementation

This project implements comprehensive Prefix Tuning and Prompt Tuning techniques based on the checklist in `fine-tuning/Prefix Tuning.md`.

## 📋 What is Prefix Tuning?

Prefix Tuning is an extremely parameter-efficient fine-tuning method that:
- **Only trains prefix embeddings** while freezing all model parameters
- **Prepends learnable vectors** to the input sequence
- **Achieves competitive performance** with minimal trainable parameters
- **Enables task-specific adaptation** through prefix conditioning

## 🏗️ Architecture

```
Input: "Classify: This movie is great!"
    ↓
Prefix: [P1] [P2] [P3] [P4] [P5] + Input tokens
    ↓
Frozen LLM: Process prefixed input
    ↓
Output: Task-specific predictions
```

### Prefix Tuning vs Other Methods

| Method | Trainable Parameters | Storage | Performance | Flexibility |
|--------|---------------------|---------|-------------|-------------|
| Full Fine-tuning | 100% | High | 100% | Low |
| LoRA | ~0.5% | Medium | 96-99% | Medium |
| Adapter | ~1% | Medium | 95-98% | Medium |
| **Prefix Tuning** | **~0.1%** | **Very Low** | **90-95%** | **High** |

## 📁 Project Structure

```
prefix-tuning/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── setup.py                    # Package setup
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── prefix_config.py        # Prefix tuning configurations
│   ├── prompt_config.py        # Prompt tuning configurations
│   └── model_config.py         # Model configurations
├── prefix_tuning/              # Core implementation
│   ├── __init__.py
│   ├── prefix_model.py         # Prefix tuning model
│   ├── prompt_model.py         # Prompt tuning model
│   ├── prefix_embeddings.py    # Prefix embedding layers
│   └── prompt_embeddings.py    # Prompt embedding layers
├── training/                   # Training systems
│   ├── __init__.py
│   ├── prefix_trainer.py       # Prefix tuning trainer
│   ├── prompt_trainer.py       # Prompt tuning trainer
│   └── evaluation.py           # Evaluation utilities
├── data/                       # Data processing
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   └── preprocessing.py        # Data preprocessing
├── experiments/                # Experiment management
│   ├── __init__.py
│   ├── experiment_runner.py    # Experiment orchestration
│   └── comparison.py           # Method comparison
├── examples/                   # Example scripts
│   ├── text_classification_prefix.py
│   ├── text_generation_prefix.py
│   └── multi_task_prefix.py
└── notebooks/                  # Jupyter notebooks
    ├── 01_prefix_tuning_basics.ipynb
    ├── 02_prompt_tuning_comparison.ipynb
    └── 03_parameter_efficiency_analysis.ipynb
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Basic Prefix Tuning

```python
from prefix_tuning import PrefixTuningModel, PrefixTrainer
from config import PrefixConfig, ModelConfig

# Setup configurations
model_config = ModelConfig(
    model_name_or_path="gpt2-medium",
    task_type="text_generation"
)

prefix_config = PrefixConfig(
    prefix_length=10,
    prefix_hidden_size=512,
    prefix_dropout=0.1,
    reparameterization=True
)

# Create prefix tuning model
model = PrefixTuningModel(model_config, prefix_config)

# Train model
trainer = PrefixTrainer(model, training_config)
trainer.train(train_dataset, eval_dataset)
```

### 3. Prompt Tuning

```python
from prefix_tuning import PromptTuningModel, PromptTrainer
from config import PromptConfig

# Setup prompt tuning
prompt_config = PromptConfig(
    num_virtual_tokens=20,
    prompt_tuning_init="random",  # random, text, vocab_sample
    prompt_tuning_init_text="Classify the sentiment:"
)

# Create prompt tuning model
model = PromptTuningModel(model_config, prompt_config)

# Train model
trainer = PromptTrainer(model, training_config)
trainer.train(train_dataset, eval_dataset)
```

## 🔧 Key Features

### ✅ Extreme Parameter Efficiency
- **Only prefix/prompt embeddings are trainable** (0.01-0.1% of total)
- Automatic parameter freezing for base model
- Memory-efficient training and inference

### ✅ Multiple Prefix Strategies
- **Prefix Tuning**: Learnable prefixes for each layer
- **Prompt Tuning**: Learnable soft prompts at input
- **P-Tuning v2**: Deep prompt tuning across layers

### ✅ Flexible Initialization
- **Random initialization**: Start from scratch
- **Text initialization**: Initialize from text prompts
- **Vocabulary sampling**: Sample from model vocabulary

### ✅ Task Adaptation
- **Classification tasks**: Sentiment, topic, NLI
- **Generation tasks**: Summarization, translation
- **Multi-task learning**: Shared prefixes across tasks

## 📊 Supported Tasks

### Text Classification
- Sentiment Analysis (SST-2, IMDB)
- Topic Classification (AG News)
- Natural Language Inference (MNLI, SNLI)

### Text Generation
- Summarization (CNN/DailyMail, XSum)
- Translation (WMT datasets)
- Dialogue Generation

### Question Answering
- Reading Comprehension (SQuAD)
- Commonsense QA (CommonsenseQA)

## 🧠 Prefix Tuning Principles

### 1. Prefix Conditioning
```python
# Standard input: [x1, x2, x3, ...]
# Prefixed input: [P1, P2, ..., Pk, x1, x2, x3, ...]
# Where P1, P2, ..., Pk are learnable prefix embeddings
```

### 2. Layer-wise Prefixes
```python
# Prefix tuning adds prefixes to each transformer layer
for layer in transformer_layers:
    # Add prefix to key and value projections
    prefixed_keys = concat([prefix_keys[layer], original_keys])
    prefixed_values = concat([prefix_values[layer], original_values])
```

### 3. Reparameterization
```python
# Use MLP to generate actual prefix embeddings
prefix_embeddings = MLP(trainable_prefix_params)
# This improves optimization stability
```

## 📈 Performance Benefits

### Parameter Efficiency
```
GPT-2 Medium: 345M parameters
Prefix Tuning: 0.1M trainable parameters (0.03% of total)
Storage: 0.4MB vs 1.4GB (3500x reduction)
```

### Training Speed
```
Full Fine-tuning: 100% computation
Prefix Tuning: ~5% computation (only prefix gradients)
Training time: 20x faster
```

### Memory Usage
```
Full Fine-tuning: High memory for all gradients
Prefix Tuning: Low memory for prefix gradients only
Memory reduction: 95%
```

## 🔬 Advanced Features

### Multi-Task Prefix Sharing
```python
# Share base model, use task-specific prefixes
shared_model = "gpt2-medium"
task_a_prefix = "sentiment_prefix.bin"  # 0.4MB
task_b_prefix = "topic_prefix.bin"      # 0.4MB
task_c_prefix = "qa_prefix.bin"         # 0.4MB

# Total: 1.4GB + 1.2MB vs 4.2GB for separate models
```

### Dynamic Prefix Length
```python
# Adjust prefix length based on task complexity
simple_task_config = PrefixConfig(prefix_length=5)
complex_task_config = PrefixConfig(prefix_length=20)
```

### Prefix Interpolation
```python
# Interpolate between task-specific prefixes
mixed_prefix = alpha * task_a_prefix + (1-alpha) * task_b_prefix
```

## 📖 Documentation

See the `notebooks/` directory for detailed tutorials:
1. **Prefix Tuning Basics**: Understanding prefix conditioning
2. **Prompt Tuning Comparison**: Comparing different approaches
3. **Parameter Efficiency Analysis**: Analyzing efficiency gains

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Prefix Tuning Paper](https://arxiv.org/abs/2101.00190)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
- [P-Tuning v2 Paper](https://arxiv.org/abs/2110.07602)
- [PEFT Library](https://github.com/huggingface/peft)
