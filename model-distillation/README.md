# 🎓 Model Distillation & Knowledge Transfer Implementation

This project implements comprehensive Model Distillation techniques based on the checklist in `fine-tuning/Model Distillation.md`.

## 📋 What is Model Distillation?

Model Distillation is a knowledge transfer technique that:
- **Transfers knowledge** from a large teacher model to a smaller student model
- **Maintains performance** while significantly reducing model size
- **Enables deployment** on resource-constrained devices
- **Accelerates inference** with minimal accuracy loss

## 🏗️ Architecture

```
Teacher Model (Large)    Student Model (Small)
     ↓                        ↓
   Logits                   Logits
     ↓                        ↓
 Soft Targets  ──────→  Knowledge Transfer
     ↓                        ↓
Temperature Scaling      Distillation Loss
     ↓                        ↓
Knowledge Distillation ──→ Compressed Model
```

### Distillation vs Other Compression Methods

| Method | Size Reduction | Performance Retention | Training Time | Complexity |
|--------|----------------|----------------------|---------------|------------|
| Pruning | 50-90% | 85-95% | Low | Medium |
| Quantization | 75% | 90-98% | Very Low | Low |
| **Distillation** | **60-95%** | **85-98%** | **Medium** | **Medium** |
| Combined | 95-99% | 80-95% | High | High |

## 📁 Project Structure

```
model-distillation/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── setup.py                    # Package setup
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── distillation_config.py  # Distillation configurations
│   ├── teacher_config.py       # Teacher model configurations
│   └── student_config.py       # Student model configurations
├── distillation/               # Core implementation
│   ├── __init__.py
│   ├── distiller.py            # Main distillation class
│   ├── losses.py               # Distillation loss functions
│   ├── teacher_model.py        # Teacher model wrapper
│   └── student_model.py        # Student model wrapper
├── techniques/                 # Distillation techniques
│   ├── __init__.py
│   ├── logit_distillation.py   # Logit-based distillation
│   ├── feature_distillation.py # Feature-based distillation
│   ├── attention_distillation.py # Attention transfer
│   └── progressive_distillation.py # Progressive knowledge transfer
├── training/                   # Training systems
│   ├── __init__.py
│   ├── distillation_trainer.py # Main trainer
│   ├── evaluation.py           # Evaluation utilities
│   └── callbacks.py            # Training callbacks
├── compression/                # Model compression
│   ├── __init__.py
│   ├── quantization.py         # Post-training quantization
│   ├── pruning.py              # Structured/unstructured pruning
│   └── optimization.py         # Model optimization
├── experiments/                # Experiment management
│   ├── __init__.py
│   ├── experiment_runner.py    # Experiment orchestration
│   ├── benchmark.py            # Benchmarking utilities
│   └── analysis.py             # Result analysis
├── examples/                   # Example scripts
│   ├── text_classification_distillation.py
│   ├── language_model_distillation.py
│   └── multi_task_distillation.py
└── notebooks/                  # Jupyter notebooks
    ├── 01_distillation_basics.ipynb
    ├── 02_teacher_student_comparison.ipynb
    └── 03_compression_analysis.ipynb
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Basic Knowledge Distillation

```python
from model_distillation import Distiller, TeacherModel, StudentModel
from config import DistillationConfig, TeacherConfig, StudentConfig

# Setup teacher model
teacher_config = TeacherConfig(
    model_name_or_path="bert-large-uncased",
    task_type="classification",
    num_labels=2
)
teacher = TeacherModel(teacher_config)

# Setup student model
student_config = StudentConfig(
    model_name_or_path="distilbert-base-uncased",
    task_type="classification",
    num_labels=2
)
student = StudentModel(student_config)

# Setup distillation
distillation_config = DistillationConfig(
    distillation_type="logit",
    temperature=4.0,
    alpha=0.7,
    beta=0.3
)

# Create distiller
distiller = Distiller(teacher, student, distillation_config)

# Train student model
distiller.distill(train_dataset, eval_dataset)
```

### 3. Feature-based Distillation

```python
from techniques import FeatureDistillation

# Setup feature distillation
feature_distillation = FeatureDistillation(
    teacher_layers=[6, 12, 18, 24],  # BERT-large layers
    student_layers=[3, 6, 9, 12],    # DistilBERT layers
    feature_loss_weight=0.5
)

distillation_config = DistillationConfig(
    distillation_type="feature",
    feature_distillation=feature_distillation
)
```

## 🔧 Key Features

### ✅ Multiple Distillation Techniques
- **Logit Distillation**: Transfer output probability distributions
- **Feature Distillation**: Transfer intermediate layer representations
- **Attention Distillation**: Transfer attention patterns
- **Progressive Distillation**: Gradual knowledge transfer

### ✅ Flexible Teacher-Student Pairs
- **BERT-large → DistilBERT**: 6x smaller, 97% performance
- **GPT-2-xl → GPT-2-small**: 16x smaller, 90% performance
- **T5-large → T5-small**: 4x smaller, 95% performance

### ✅ Advanced Loss Functions
- **KL Divergence**: For probability distribution matching
- **MSE Loss**: For feature representation matching
- **Cosine Similarity**: For attention pattern matching
- **Combined Loss**: Weighted combination of multiple losses

### ✅ Compression Integration
- **Post-training Quantization**: INT8/FP16 quantization
- **Structured Pruning**: Remove entire neurons/layers
- **Knowledge Distillation + Compression**: Combined approach

## 📊 Supported Tasks

### Text Classification
- Sentiment Analysis (IMDB, SST-2)
- Topic Classification (AG News)
- Natural Language Inference (MNLI, SNLI)

### Language Modeling
- Causal Language Modeling (GPT-style)
- Masked Language Modeling (BERT-style)
- Sequence-to-Sequence (T5-style)

### Question Answering
- Reading Comprehension (SQuAD)
- Commonsense QA (CommonsenseQA)

## 🧠 Distillation Principles

### 1. Temperature Scaling
```python
# Soften probability distributions for better knowledge transfer
def temperature_scaled_softmax(logits, temperature):
    return F.softmax(logits / temperature, dim=-1)

# Teacher soft targets
teacher_probs = temperature_scaled_softmax(teacher_logits, temperature=4.0)

# Student predictions
student_probs = temperature_scaled_softmax(student_logits, temperature=4.0)
```

### 2. Knowledge Transfer Loss
```python
# Combined loss function
def distillation_loss(student_logits, teacher_logits, true_labels, alpha, temperature):
    # Distillation loss (soft targets)
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Task loss (hard targets)
    ce_loss = F.cross_entropy(student_logits, true_labels)
    
    # Combined loss
    return alpha * kl_loss + (1 - alpha) * ce_loss
```

### 3. Feature Matching
```python
# Match intermediate representations
def feature_distillation_loss(student_features, teacher_features):
    # Project student features to teacher dimension if needed
    if student_features.size(-1) != teacher_features.size(-1):
        student_features = projection_layer(student_features)
    
    # MSE loss between features
    return F.mse_loss(student_features, teacher_features)
```

## 📈 Performance Benefits

### Model Size Reduction
```
BERT-large → DistilBERT:
- Parameters: 340M → 66M (6x reduction)
- Model size: 1.3GB → 255MB (5x reduction)
- Inference speed: 1x → 2x faster
- Performance: 100% → 97% (3% drop)
```

### Memory and Speed
```
Teacher (BERT-large):
- Memory: 4GB GPU memory
- Inference: 100ms per batch
- Accuracy: 92.5%

Student (DistilBERT):
- Memory: 1GB GPU memory (4x reduction)
- Inference: 50ms per batch (2x faster)
- Accuracy: 90.1% (2.4% drop)
```

### Deployment Benefits
```
Mobile Deployment:
- Model size: 255MB vs 1.3GB
- RAM usage: 512MB vs 2GB
- Battery life: 2x longer
- Latency: 50% reduction
```

## 🔬 Advanced Features

### Progressive Distillation
```python
# Gradually transfer knowledge layer by layer
progressive_distiller = ProgressiveDistillation(
    teacher_layers=24,
    student_layers=12,
    distillation_schedule="linear"  # linear, exponential, custom
)
```

### Multi-Task Distillation
```python
# Distill knowledge across multiple tasks simultaneously
multi_task_distiller = MultiTaskDistiller(
    tasks=["sentiment", "nli", "qa"],
    task_weights=[0.4, 0.3, 0.3],
    shared_distillation=True
)
```

### Attention Transfer
```python
# Transfer attention patterns from teacher to student
attention_distiller = AttentionDistillation(
    attention_loss_weight=0.3,
    layer_mapping={0: 0, 6: 3, 12: 6, 18: 9, 24: 12}
)
```

## 📖 Documentation

See the `notebooks/` directory for detailed tutorials:
1. **Distillation Basics**: Understanding knowledge transfer
2. **Teacher-Student Comparison**: Analyzing model differences
3. **Compression Analysis**: Evaluating compression techniques

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Knowledge Distillation Paper](https://arxiv.org/abs/1503.02531)
- [TinyBERT Paper](https://arxiv.org/abs/1909.10351)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
