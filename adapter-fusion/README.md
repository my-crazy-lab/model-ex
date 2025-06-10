# 🔗 Adapter Fusion Implementation

This project implements Adapter Fusion based on the checklist in `fine-tuning/Adapter Fusion.md`.

## 📋 What is Adapter Fusion?

Adapter Fusion is an advanced parameter-efficient fine-tuning method that:
- Trains multiple task-specific adapters independently
- Combines knowledge from multiple adapters using fusion mechanisms
- Enables multi-task learning without catastrophic forgetting
- Achieves superior performance compared to individual adapters

## 🏗️ Architecture

```
Base Model (Frozen)
    ↓
Task A Adapter → \
Task B Adapter → → Fusion Layer → Combined Output
Task C Adapter → /
```

### Fusion Mechanisms

1. **Attention-based Fusion**: Uses attention to weight adapter outputs
2. **Weighted Average**: Learnable weights for combining adapters
3. **Gating**: Dynamic selection of adapters based on input

## 📁 Project Structure

```
adapter-fusion/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── setup.py                    # Package setup
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── model_config.py         # Model configurations
│   ├── adapter_config.py       # Adapter configurations
│   ├── fusion_config.py        # Fusion-specific configs
│   └── training_config.py      # Training configurations
├── fusion/                     # Fusion implementations
│   ├── __init__.py
│   ├── fusion_layer.py         # Core fusion mechanisms
│   ├── adapter_manager.py      # Multi-adapter management
│   └── fusion_model.py         # Model with fusion
├── adapters/                   # Individual adapter implementations
│   ├── __init__.py
│   ├── adapter_layer.py        # Basic adapter layer
│   └── task_adapters.py        # Task-specific adapters
├── data/                       # Data processing
│   ├── __init__.py
│   ├── multi_task_loader.py    # Multi-task data loading
│   └── task_datasets.py        # Task-specific datasets
├── training/                   # Training scripts
│   ├── __init__.py
│   ├── adapter_trainer.py      # Individual adapter training
│   ├── fusion_trainer.py       # Fusion training
│   └── multi_task_trainer.py   # Multi-task training
├── evaluation/                 # Evaluation scripts
│   ├── __init__.py
│   ├── fusion_evaluator.py     # Fusion evaluation
│   └── task_evaluator.py       # Task-specific evaluation
├── inference/                  # Inference scripts
│   ├── __init__.py
│   └── fusion_pipeline.py      # Fusion inference pipeline
├── examples/                   # Example scripts
│   ├── train_individual_adapters.py
│   ├── fusion_training.py
│   └── multi_task_fusion.py
└── notebooks/                  # Jupyter notebooks
    ├── 01_adapter_fusion_basics.ipynb
    ├── 02_training_workflow.ipynb
    └── 03_fusion_analysis.ipynb
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Basic Workflow

#### Step 1: Train Individual Adapters
```python
from training import AdapterTrainer
from config import ModelConfig, AdapterConfig

# Train adapter for each task
for task_name in ["sentiment", "nli", "qa"]:
    trainer = AdapterTrainer(task_name)
    trainer.train(task_dataset)
    trainer.save_adapter(f"./adapters/{task_name}")
```

#### Step 2: Fusion Training
```python
from training import FusionTrainer
from config import FusionConfig

# Load trained adapters and perform fusion
fusion_config = FusionConfig(
    adapter_paths=["./adapters/sentiment", "./adapters/nli", "./adapters/qa"],
    fusion_method="attention"
)

fusion_trainer = FusionTrainer(fusion_config)
fusion_trainer.train_fusion(multi_task_dataset)
```

#### Step 3: Inference
```python
from inference import FusionPipeline

# Use fused model for inference
pipeline = FusionPipeline.from_pretrained("./fusion_model")
result = pipeline.predict("This movie is great!", task="sentiment")
```

## 🔧 Key Features

### ✅ Multi-Task Adapter Training
- **Independent Training**: Train adapters for different tasks separately
- **Task-Specific Optimization**: Optimized for each task's characteristics
- **Modular Design**: Easy to add/remove tasks

### ✅ Advanced Fusion Mechanisms
- **Attention Fusion**: Learn to attend to relevant adapters
- **Weighted Fusion**: Learnable combination weights
- **Dynamic Fusion**: Input-dependent adapter selection
- **Hierarchical Fusion**: Multi-level fusion strategies

### ✅ Flexible Architecture
- **Multiple Fusion Points**: Fusion at different transformer layers
- **Adapter Composition**: Stack and combine different adapter types
- **Task Routing**: Automatic task detection and routing

### ✅ Training Strategies
- **Sequential Training**: Train adapters then fusion
- **Joint Training**: Train adapters and fusion together
- **Continual Learning**: Add new tasks without forgetting

## 📊 Supported Tasks

### Text Classification
- Sentiment Analysis (SST-2, IMDB)
- Natural Language Inference (MNLI, SNLI)
- Topic Classification (AG News)

### Token Classification
- Named Entity Recognition (CoNLL-2003)
- Part-of-Speech Tagging

### Question Answering
- Reading Comprehension (SQuAD)
- Multiple Choice QA

### Text Generation
- Summarization
- Translation

## 🧠 Fusion Methods

### 1. Attention-based Fusion
```python
fusion_config = FusionConfig(
    fusion_method="attention",
    num_attention_heads=8,
    attention_dropout=0.1
)
```

### 2. Weighted Average Fusion
```python
fusion_config = FusionConfig(
    fusion_method="weighted",
    learnable_weights=True,
    weight_initialization="uniform"
)
```

### 3. Gating Fusion
```python
fusion_config = FusionConfig(
    fusion_method="gating",
    gate_activation="sigmoid",
    gate_bias=True
)
```

## 📈 Performance Benefits

### Parameter Efficiency
```
Base Model: 110M parameters (shared)
Task A Adapter: 0.5M parameters
Task B Adapter: 0.5M parameters  
Task C Adapter: 0.5M parameters
Fusion Layer: 0.1M parameters
Total: 111.6M parameters (vs 330M for separate models)
```

### Knowledge Transfer
- Cross-task knowledge sharing
- Improved performance on low-resource tasks
- Better generalization

### Scalability
- Easy to add new tasks
- Modular architecture
- Efficient inference

## 🔬 Advanced Features

### Multi-Level Fusion
- Fusion at multiple transformer layers
- Layer-specific fusion strategies
- Hierarchical knowledge combination

### Dynamic Adapter Selection
- Input-dependent adapter activation
- Sparse adapter usage
- Efficient computation

### Continual Learning
- Add new adapters without retraining
- Prevent catastrophic forgetting
- Incremental knowledge accumulation

## 📖 Documentation

See the `notebooks/` directory for detailed tutorials:
1. **Adapter Fusion Basics**: Understanding fusion mechanisms
2. **Training Workflow**: Step-by-step training guide
3. **Fusion Analysis**: Analyzing fusion behavior

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [AdapterFusion Paper](https://arxiv.org/abs/2005.00247) for the original fusion concept
- [Adapter-Hub](https://adapterhub.ml/) for the adapter ecosystem
- Hugging Face for the transformers library
