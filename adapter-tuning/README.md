# 🔧 Adapter Tuning Implementation

This project implements Adapter Tuning based on the checklist in `fine-tuning/Adapter Tuning.md`.

## 📋 What is Adapter Tuning?

Adapter Tuning is a parameter-efficient fine-tuning method that:
- Inserts small neural network modules (adapters) into pre-trained models
- Freezes the original model parameters
- Only trains the adapter parameters
- Achieves comparable performance to full fine-tuning with much fewer parameters

## 🏗️ Architecture

```
Original Transformer Layer:
Input → Self-Attention → Add&Norm → Feed-Forward → Add&Norm → Output

With Adapters:
Input → Self-Attention → Add&Norm → Adapter → Feed-Forward → Add&Norm → Adapter → Output
```

Each Adapter is a simple 2-layer MLP:
```
Input → Linear(down) → ReLU → Linear(up) → Residual Connection → Output
```

## 📁 Project Structure

```
adapter-tuning/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── setup.py                    # Package setup
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── model_config.py         # Model and adapter configurations
│   └── training_config.py      # Training configurations
├── adapters/                   # Adapter implementations
│   ├── __init__.py
│   ├── adapter_layer.py        # Core adapter layer
│   ├── adapter_model.py        # Model with adapters
│   └── adapter_config.py       # Adapter-specific configs
├── data/                       # Data processing
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   └── preprocessing.py        # Data preprocessing
├── training/                   # Training scripts
│   ├── __init__.py
│   ├── trainer.py              # Training logic
│   └── utils.py                # Training utilities
├── evaluation/                 # Evaluation scripts
│   ├── __init__.py
│   ├── evaluator.py            # Model evaluation
│   └── metrics.py              # Evaluation metrics
├── inference/                  # Inference scripts
│   ├── __init__.py
│   └── pipeline.py             # Inference pipeline
├── examples/                   # Example scripts
│   ├── text_classification.py
│   ├── sentiment_analysis.py
│   └── multi_task_learning.py
└── notebooks/                  # Jupyter notebooks
    ├── 01_adapter_basics.ipynb
    ├── 02_training_adapters.ipynb
    └── 03_multi_adapter_management.ipynb
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Basic Usage

```python
from config import ModelConfig, AdapterConfig, TrainingConfig
from adapters import AdapterModel
from training import AdapterTrainer

# Configure model with adapters
model_config = ModelConfig(
    model_name_or_path="bert-base-uncased",
    num_labels=2
)

adapter_config = AdapterConfig(
    adapter_size=64,
    adapter_dropout=0.1,
    adapter_activation="relu"
)

# Create model with adapters
model = AdapterModel(model_config, adapter_config)

# Train adapters
trainer = AdapterTrainer(model_config, adapter_config, training_config)
trainer.train(train_dataset, eval_dataset)
```

### 3. Command Line Usage

```bash
# Train adapters for text classification
python examples/text_classification.py --dataset imdb --adapter-size 64

# Multi-task learning with multiple adapters
python examples/multi_task_learning.py --tasks sentiment,classification
```

## 🔧 Key Features

### ✅ Core Adapter Implementation
- **Bottleneck Architecture**: Down-projection → Activation → Up-projection
- **Residual Connections**: Skip connections for stable training
- **Flexible Placement**: Insert adapters after any transformer layer
- **Parameter Efficiency**: Only 0.5-3% additional parameters

### ✅ Multiple Adapter Types
- **Standard Adapters**: Basic bottleneck adapters
- **Parallel Adapters**: Multiple adapters in parallel
- **Sequential Adapters**: Stacked adapters for complex tasks
- **Task-Specific Adapters**: Different adapters for different tasks

### ✅ Advanced Features
- **Multi-Task Learning**: Train multiple adapters simultaneously
- **Adapter Fusion**: Combine multiple adapters intelligently
- **Dynamic Adapter Selection**: Choose adapters based on input
- **Adapter Composition**: Stack and combine adapters

### ✅ Production Ready
- **Efficient Inference**: Fast adapter switching
- **Model Serialization**: Save/load adapters independently
- **Batch Processing**: Handle multiple tasks in one batch
- **Memory Optimization**: Minimal memory overhead

## 📊 Supported Tasks

- **Text Classification**: Sentiment analysis, topic classification
- **Named Entity Recognition**: Token-level classification
- **Question Answering**: Reading comprehension tasks
- **Text Generation**: Conditional text generation
- **Multi-Task Learning**: Multiple tasks with shared backbone

## 🧠 Supported Models

- **BERT family**: bert-base-uncased, bert-large, roberta, distilbert
- **GPT family**: gpt2, gpt-neo, gpt-j
- **T5 family**: t5-small, t5-base, t5-large
- **Custom models**: Any transformer-based model

## 📈 Performance Benefits

### Memory Efficiency
```
Full Fine-tuning: 110M parameters (BERT-base)
Adapter Tuning: 110M + 0.5M = 110.5M parameters (0.45% increase)
```

### Training Speed
```
Full Fine-tuning: Train all 110M parameters
Adapter Tuning: Train only 0.5M parameters (220x fewer)
```

### Model Sharing
```
Base Model: 110M parameters (shared)
Task A Adapter: 0.5M parameters
Task B Adapter: 0.5M parameters
Total Storage: 111M parameters (vs 220M for two full models)
```

## 🔬 Experimental Features

- **Adapter Pruning**: Remove unnecessary adapter parameters
- **Knowledge Distillation**: Transfer knowledge between adapters
- **Meta-Learning**: Learn to adapt quickly to new tasks
- **Continual Learning**: Add new tasks without forgetting old ones

## 📖 Documentation

See the `notebooks/` directory for detailed tutorials:
1. **Adapter Basics**: Understanding adapter architecture
2. **Training Adapters**: Step-by-step training guide
3. **Multi-Adapter Management**: Managing multiple adapters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Adapter-Hub](https://adapterhub.ml/) for the adapter ecosystem
- [Houlsby et al. (2019)](https://arxiv.org/abs/1902.00751) for the original adapter paper
- Hugging Face for the transformers library
