# 🎯 Multitask Fine-tuning Implementation

This project implements comprehensive Multitask Fine-tuning techniques based on the checklist in `fine-tuning/Multitask Fine-tuning.md`.

## 📋 What is Multitask Fine-tuning?

Multitask Fine-tuning is a training paradigm that:
- **Trains a single model** on multiple tasks simultaneously
- **Shares knowledge** across related tasks for better generalization
- **Improves efficiency** by leveraging common representations
- **Enables zero-shot transfer** to new tasks

## 🏗️ Architecture

```
Shared Backbone (BERT/T5/GPT)
         ↓
    Shared Encoder
         ↓
   Task-Specific Heads
    ↙    ↓    ↘
Task A  Task B  Task C
(NLI)   (QA)   (Summ)
```

### Multitask vs Single-task Training

| Aspect | Single-task | Multitask |
|--------|-------------|-----------|
| Model Count | N models | 1 model |
| Training Time | N × T | 1.2 × T |
| Memory Usage | N × M | 1 × M |
| Knowledge Sharing | None | High |
| Zero-shot Transfer | Poor | Good |
| Task Interference | None | Possible |

## 📁 Project Structure

```
multitask-fine-tuning/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── setup.py                    # Package setup
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── multitask_config.py     # Multitask configurations
│   ├── task_config.py          # Individual task configurations
│   └── training_config.py      # Training configurations
├── multitask/                  # Core implementation
│   ├── __init__.py
│   ├── multitask_model.py      # Main multitask model
│   ├── task_heads.py           # Task-specific heads
│   ├── data_manager.py         # Data management
│   └── loss_manager.py         # Loss balancing
├── tasks/                      # Task implementations
│   ├── __init__.py
│   ├── classification.py       # Text classification
│   ├── question_answering.py   # Question answering
│   ├── summarization.py        # Text summarization
│   ├── named_entity_recognition.py # NER
│   └── natural_language_inference.py # NLI
├── training/                   # Training systems
│   ├── __init__.py
│   ├── multitask_trainer.py    # Main trainer
│   ├── sampling_strategies.py  # Data sampling
│   ├── evaluation.py           # Evaluation utilities
│   └── callbacks.py            # Training callbacks
├── data/                       # Data processing
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocessing.py        # Data preprocessing
│   └── task_datasets.py        # Task-specific datasets
├── experiments/                # Experiment management
│   ├── __init__.py
│   ├── experiment_runner.py    # Experiment orchestration
│   ├── hyperparameter_search.py # HPO
│   └── analysis.py             # Result analysis
├── examples/                   # Example scripts
│   ├── multitask_classification.py
│   ├── multitask_qa_summarization.py
│   └── zero_shot_transfer.py
└── notebooks/                  # Jupyter notebooks
    ├── 01_multitask_basics.ipynb
    ├── 02_task_interference_analysis.ipynb
    └── 03_zero_shot_evaluation.ipynb
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Basic Multitask Training

```python
from multitask_fine_tuning import MultitaskModel, MultitaskTrainer
from config import MultitaskConfig, TaskConfig

# Define tasks
tasks = {
    "sentiment": TaskConfig(
        task_type="classification",
        num_labels=2,
        dataset_name="imdb"
    ),
    "nli": TaskConfig(
        task_type="classification", 
        num_labels=3,
        dataset_name="snli"
    ),
    "qa": TaskConfig(
        task_type="question_answering",
        dataset_name="squad"
    )
}

# Setup multitask configuration
config = MultitaskConfig(
    model_name_or_path="bert-base-uncased",
    tasks=tasks,
    task_sampling_strategy="proportional",
    loss_weighting_strategy="equal"
)

# Create multitask model
model = MultitaskModel(config)

# Train model
trainer = MultitaskTrainer(model, config)
trainer.train(train_datasets, eval_datasets)
```

### 3. Zero-shot Transfer

```python
# Load trained multitask model
model = MultitaskModel.from_pretrained("./multitask_model")

# Evaluate on new task without training
results = model.evaluate_zero_shot(
    task_name="new_task",
    test_dataset=new_task_dataset,
    task_type="classification"
)
```

## 🔧 Key Features

### ✅ Multiple Task Types
- **Text Classification**: Sentiment, topic, intent classification
- **Question Answering**: Extractive and abstractive QA
- **Text Summarization**: Document and dialogue summarization
- **Named Entity Recognition**: Token-level classification
- **Natural Language Inference**: Textual entailment

### ✅ Flexible Architecture
- **Shared Backbone**: BERT, RoBERTa, T5, GPT models
- **Task-specific Heads**: Customizable output layers
- **Prompt-based Training**: Instruction-following paradigm
- **Adapter Integration**: Parameter-efficient multitask learning

### ✅ Advanced Training Strategies
- **Task Sampling**: Proportional, temperature-based, curriculum
- **Loss Balancing**: Equal weighting, uncertainty weighting, gradient-based
- **Data Mixing**: Batch-level and example-level mixing
- **Progressive Training**: Gradual task introduction

### ✅ Evaluation & Analysis
- **Per-task Metrics**: Task-specific evaluation
- **Transfer Analysis**: Positive/negative transfer measurement
- **Interference Detection**: Task conflict identification
- **Zero-shot Evaluation**: Generalization assessment

## 📊 Supported Tasks

### Text Classification
- Sentiment Analysis (IMDB, SST-2)
- Topic Classification (AG News, 20 Newsgroups)
- Intent Classification (ATIS, SNIPS)

### Question Answering
- Reading Comprehension (SQuAD, MS MARCO)
- Commonsense QA (CommonsenseQA, PIQA)
- Open-domain QA (Natural Questions)

### Text Generation
- Summarization (CNN/DailyMail, XSum)
- Dialogue Generation (PersonaChat)
- Data-to-text (WebNLG, E2E)

### Sequence Labeling
- Named Entity Recognition (CoNLL-2003)
- Part-of-speech Tagging (Penn Treebank)
- Chunking (CoNLL-2000)

## 🧠 Multitask Learning Principles

### 1. Shared Representations
```python
# Shared backbone processes all tasks
shared_features = backbone_model(input_text)

# Task-specific heads process shared features
task_a_output = task_a_head(shared_features)
task_b_output = task_b_head(shared_features)
task_c_output = task_c_head(shared_features)
```

### 2. Task Sampling Strategies
```python
# Proportional sampling (based on dataset size)
def proportional_sampling(datasets):
    total_size = sum(len(d) for d in datasets.values())
    probabilities = {task: len(d)/total_size for task, d in datasets.items()}
    return probabilities

# Temperature-based sampling (control task balance)
def temperature_sampling(datasets, temperature=1.0):
    sizes = [len(d) for d in datasets.values()]
    probs = softmax([s**temperature for s in sizes])
    return dict(zip(datasets.keys(), probs))
```

### 3. Loss Balancing
```python
# Uncertainty-based weighting
def uncertainty_weighting(task_losses, task_uncertainties):
    weights = {}
    for task in task_losses:
        # Higher uncertainty = lower weight
        weights[task] = 1.0 / (2 * task_uncertainties[task]**2)
    return weights

# Gradient-based balancing
def gradient_balancing(task_gradients):
    # Balance gradient magnitudes across tasks
    grad_norms = {task: torch.norm(grad) for task, grad in task_gradients.items()}
    weights = {task: 1.0 / norm for task, norm in grad_norms.items()}
    return weights
```

## 📈 Performance Benefits

### Knowledge Sharing
```
Single-task Training:
Task A: 85% accuracy (trained separately)
Task B: 82% accuracy (trained separately)  
Task C: 78% accuracy (trained separately)

Multitask Training:
Task A: 87% accuracy (+2% from shared knowledge)
Task B: 85% accuracy (+3% from shared knowledge)
Task C: 81% accuracy (+3% from shared knowledge)
```

### Resource Efficiency
```
Single-task Approach:
- Models: 3 separate models
- Training time: 3 × T hours
- Memory: 3 × M GB
- Storage: 3 × S GB

Multitask Approach:
- Models: 1 shared model
- Training time: 1.2 × T hours (20% overhead)
- Memory: 1 × M GB
- Storage: 1 × S GB
```

### Zero-shot Transfer
```
Multitask Model Performance on Unseen Tasks:
- Similar tasks: 70-85% of supervised performance
- Related tasks: 50-70% of supervised performance
- Distant tasks: 30-50% of supervised performance
```

## 🔬 Advanced Features

### Task Interference Mitigation
```python
# Gradient surgery to reduce negative transfer
def gradient_surgery(task_gradients, task_weights):
    # Project conflicting gradients
    for task_a, grad_a in task_gradients.items():
        for task_b, grad_b in task_gradients.items():
            if task_a != task_b:
                # Check for gradient conflict
                similarity = torch.cosine_similarity(grad_a, grad_b, dim=0)
                if similarity < 0:  # Conflicting gradients
                    # Project grad_a onto grad_b's orthogonal space
                    projection = torch.dot(grad_a, grad_b) / torch.dot(grad_b, grad_b) * grad_b
                    task_gradients[task_a] = grad_a - projection
    return task_gradients
```

### Curriculum Learning
```python
# Progressive task introduction
curriculum_schedule = {
    "phase_1": ["sentiment"],  # Start with simple task
    "phase_2": ["sentiment", "nli"],  # Add related task
    "phase_3": ["sentiment", "nli", "qa"]  # Add complex task
}
```

### Meta-learning Integration
```python
# MAML-style meta-learning for quick adaptation
def meta_learning_update(model, support_tasks, query_tasks):
    # Inner loop: adapt to support tasks
    adapted_params = model.adapt(support_tasks, num_steps=5)
    
    # Outer loop: optimize for query tasks
    meta_loss = model.compute_loss(query_tasks, adapted_params)
    return meta_loss
```

## 📖 Documentation

See the `notebooks/` directory for detailed tutorials:
1. **Multitask Basics**: Understanding multitask learning
2. **Task Interference Analysis**: Analyzing positive/negative transfer
3. **Zero-shot Evaluation**: Evaluating generalization capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [MT-DNN Paper](https://arxiv.org/abs/1901.11504)
- [ExT5 Paper](https://arxiv.org/abs/2111.10952)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
