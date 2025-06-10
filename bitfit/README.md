# ⚡ BitFit: Bias-only Fine-tuning Implementation

This project implements BitFit (Bias-only Fine-tuning) based on the checklist in `fine-tuning/BitFit.md`.

## 📋 What is BitFit?

BitFit is an extremely parameter-efficient fine-tuning method that:
- **Only trains bias parameters** while freezing all other weights
- Achieves competitive performance with minimal parameter updates
- Requires only 0.08-0.1% of total parameters to be trainable
- Provides fast training and inference with minimal memory overhead

## 🏗️ Architecture

```
Pre-trained Model (110M parameters)
    ↓
Freeze All Weights (109.9M parameters frozen)
    ↓
Train Only Bias Terms (0.1M parameters trainable)
    ↓
Fine-tuned Model (99.9% parameter reduction!)
```

### BitFit vs Other Methods

| Method | Trainable Parameters | Performance | Memory | Speed |
|--------|---------------------|-------------|---------|-------|
| Full Fine-tuning | 100% | 100% | High | Slow |
| Adapter | ~1% | 95-98% | Medium | Medium |
| LoRA | ~0.5% | 96-99% | Medium | Medium |
| **BitFit** | **~0.1%** | **90-95%** | **Low** | **Fast** |

## 📁 Project Structure

```
bitfit/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── setup.py                    # Package setup
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── model_config.py         # Model configurations
│   ├── bitfit_config.py        # BitFit-specific configs
│   └── training_config.py      # Training configurations
├── bitfit/                     # Core BitFit implementation
│   ├── __init__.py
│   ├── bitfit_model.py         # BitFit model wrapper
│   ├── parameter_utils.py      # Parameter freezing utilities
│   └── bias_optimizer.py       # Bias-only optimizer
├── data/                       # Data processing
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   └── preprocessing.py        # Data preprocessing
├── training/                   # Training scripts
│   ├── __init__.py
│   ├── bitfit_trainer.py       # BitFit trainer
│   ├── evaluation.py           # Evaluation utilities
│   └── callbacks.py            # Training callbacks
├── inference/                  # Inference scripts
│   ├── __init__.py
│   └── bitfit_pipeline.py      # Inference pipeline
├── experiments/                # Experiment tracking
│   ├── __init__.py
│   ├── experiment_manager.py   # Experiment management
│   └── comparison.py           # Method comparison
├── examples/                   # Example scripts
│   ├── text_classification.py
│   ├── sentiment_analysis.py
│   └── glue_benchmark.py
└── notebooks/                  # Jupyter notebooks
    ├── 01_bitfit_basics.ipynb
    ├── 02_parameter_analysis.ipynb
    └── 03_performance_comparison.ipynb
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Basic BitFit Fine-tuning

```python
from bitfit import BitFitModel, BitFitTrainer
from config import ModelConfig, BitFitConfig, TrainingConfig

# Setup configurations
model_config = ModelConfig(
    model_name_or_path="bert-base-uncased",
    num_labels=2,
    task_type="classification"
)

bitfit_config = BitFitConfig(
    freeze_all_weights=True,
    train_bias_only=True,
    bias_learning_rate=1e-3
)

training_config = TrainingConfig(
    output_dir="./bitfit_results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=1e-3
)

# Create BitFit model
model = BitFitModel(model_config, bitfit_config)

# Train model
trainer = BitFitTrainer(model, training_config)
trainer.train(train_dataset, eval_dataset)
```

### 3. Inference

```python
from inference import BitFitPipeline

# Load trained model
pipeline = BitFitPipeline.from_pretrained("./bitfit_results")

# Make predictions
result = pipeline.predict("This movie is great!")
print(f"Prediction: {result}")
```

## 🔧 Key Features

### ✅ Extreme Parameter Efficiency
- **Only bias parameters are trainable** (0.08-0.1% of total)
- Automatic parameter freezing and bias identification
- Memory-efficient training and inference

### ✅ Fast Training
- **Minimal gradient computation** for bias terms only
- Reduced memory footprint during training
- Quick convergence due to focused parameter updates

### ✅ Easy Integration
- **Drop-in replacement** for standard fine-tuning
- Compatible with any transformer architecture
- Seamless integration with Hugging Face models

### ✅ Comprehensive Analysis
- **Parameter counting** and efficiency metrics
- Performance comparison with other methods
- Detailed experiment tracking and logging

## 📊 Supported Tasks

### Text Classification
- Sentiment Analysis (SST-2, IMDB)
- Topic Classification (AG News)
- Intent Detection

### Natural Language Inference
- MNLI, SNLI, RTE
- Textual Entailment

### Token Classification
- Named Entity Recognition (CoNLL-2003)
- Part-of-Speech Tagging

### Question Answering
- SQuAD 1.1/2.0
- Reading Comprehension

## 🧠 BitFit Principles

### 1. Bias-only Training
```python
# Only bias parameters are trainable
for name, param in model.named_parameters():
    if 'bias' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

### 2. Minimal Parameter Updates
```python
# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
efficiency = trainable_params / total_params * 100
print(f"Parameter efficiency: {efficiency:.2f}%")
```

### 3. Task-specific Adaptation
```python
# Bias terms adapt to task-specific patterns
# while preserving pre-trained representations
```

## 📈 Performance Benefits

### Parameter Efficiency
```
BERT-base: 110M parameters
BitFit trainable: 0.1M parameters (0.09% of total)
Storage: 0.4MB vs 440MB (1100x reduction)
```

### Training Speed
```
Full Fine-tuning: 100% computation
BitFit: ~10% computation (bias gradients only)
Training time: 5-10x faster
```

### Memory Usage
```
Full Fine-tuning: High memory for all gradients
BitFit: Low memory for bias gradients only
Memory reduction: 80-90%
```

## 🔬 Advanced Features

### Selective Bias Training
```python
# Train only specific bias types
bitfit_config = BitFitConfig(
    train_attention_bias=True,
    train_feedforward_bias=True,
    train_layer_norm_bias=False,
    train_classifier_bias=True
)
```

### Learning Rate Scheduling
```python
# Different learning rates for different bias types
bias_optimizer = BiasOptimizer(
    attention_lr=1e-3,
    feedforward_lr=5e-4,
    classifier_lr=2e-3
)
```

### Gradient Clipping
```python
# Specialized gradient clipping for bias terms
training_config = TrainingConfig(
    bias_gradient_clipping=True,
    max_bias_grad_norm=1.0
)
```

## 📖 Documentation

See the `notebooks/` directory for detailed tutorials:
1. **BitFit Basics**: Understanding bias-only fine-tuning
2. **Parameter Analysis**: Analyzing parameter efficiency
3. **Performance Comparison**: Comparing with other methods

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [BitFit Paper](https://arxiv.org/abs/2106.10199) for the original method
- [Hugging Face](https://huggingface.co/) for transformer implementations
- [Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2106.04647) survey
