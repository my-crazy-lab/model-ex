# 🎛️ Parameter Selection Guide for LoRA/PEFT Training

This guide helps you choose the right parameters based on your specific situation, resources, and goals.

## 🎯 Quick Decision Framework

### Step 1: Identify Your Scenario
- **🚀 Quick Experiment**: Testing ideas, proof of concept
- **📊 Standard Training**: Balanced performance and time
- **🎯 Production Quality**: Best possible results
- **💾 Memory Constrained**: Limited GPU memory
- **⚡ Time Constrained**: Need results quickly

### Step 2: Choose Base Configuration
```python
from config import QUICK_TRAINING_CONFIG, STANDARD_TRAINING_CONFIG, INTENSIVE_TRAINING_CONFIG

# Quick experiment (30 min - 2 hours)
config = QUICK_TRAINING_CONFIG

# Standard training (2-8 hours)  
config = STANDARD_TRAINING_CONFIG

# Production quality (8+ hours)
config = INTENSIVE_TRAINING_CONFIG
```

### Step 3: Customize Based on Your Constraints

---

## 📊 Parameter Categories & Selection Logic

### 1. 🏋️ Training Scale Parameters

#### `num_train_epochs`
**What it does**: Number of complete passes through your dataset

**How to choose**:
```python
# Small dataset (< 1K samples)
num_train_epochs = 5-10

# Medium dataset (1K - 100K samples)  
num_train_epochs = 3-5

# Large dataset (> 100K samples)
num_train_epochs = 1-3

# Rule of thumb: More data = fewer epochs needed
```

#### `per_device_train_batch_size`
**What it does**: Number of samples processed together on each GPU

**How to choose**:
```python
# Based on GPU memory:
# 4GB GPU (GTX 1050 Ti, etc.)
per_device_train_batch_size = 2-4

# 8GB GPU (RTX 3070, etc.)
per_device_train_batch_size = 8-16

# 16GB+ GPU (RTX 4090, A100, etc.)
per_device_train_batch_size = 16-32

# If you get OOM (Out of Memory), reduce this first!
```

#### `gradient_accumulation_steps`
**What it does**: Simulates larger batch sizes by accumulating gradients

**How to choose**:
```python
# Effective batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

# Target effective batch size: 32-128 for most tasks
# If per_device_train_batch_size = 8, and you want effective batch size = 32:
gradient_accumulation_steps = 32 // 8 = 4

# Memory constrained? Increase this, decrease batch_size
gradient_accumulation_steps = 4-8
per_device_train_batch_size = 2-4
```

### 2. 🎯 Learning Parameters

#### `learning_rate`
**What it does**: How big steps the model takes during learning

**How to choose**:
```python
# LoRA typically needs higher learning rates than full fine-tuning

# Conservative (safe choice)
learning_rate = 1e-4

# Standard (most common)
learning_rate = 5e-4

# Aggressive (for quick experiments)
learning_rate = 1e-3

# Large models (LLaMA 13B+)
learning_rate = 1e-4 to 3e-4

# Small models (BERT-base)
learning_rate = 5e-4 to 1e-3
```

#### `weight_decay`
**What it does**: Prevents overfitting by penalizing large weights

**How to choose**:
```python
# Small dataset (high overfitting risk)
weight_decay = 0.01 to 0.1

# Large dataset (low overfitting risk)
weight_decay = 0.001 to 0.01

# Default safe choice
weight_decay = 0.01
```

#### `warmup_ratio`
**What it does**: Gradually increases learning rate at the start

**How to choose**:
```python
# Short training (< 1000 steps)
warmup_ratio = 0.1

# Medium training (1000-10000 steps)
warmup_ratio = 0.06

# Long training (> 10000 steps)
warmup_ratio = 0.03

# Rule: Longer training = smaller warmup ratio
```

### 3. 💾 Memory Optimization Parameters

#### `fp16` vs `bf16`
**What it does**: Uses half precision to save memory

**How to choose**:
```python
# Modern GPUs (RTX 30/40 series, A100)
bf16 = True
fp16 = False

# Older GPUs (RTX 20 series, V100)
fp16 = True
bf16 = False

# CPU or very old GPUs
fp16 = False
bf16 = False
```

#### `gradient_checkpointing`
**What it does**: Trades compute for memory

**How to choose**:
```python
# Memory constrained (< 8GB GPU)
gradient_checkpointing = True

# Plenty of memory (> 16GB GPU)
gradient_checkpointing = False

# Note: Increases training time by ~20%
```

### 4. 📈 Monitoring Parameters

#### `eval_steps` & `save_steps`
**What it does**: How often to evaluate and save

**How to choose**:
```python
# Quick experiments
eval_steps = 100
save_steps = 100

# Standard training
eval_steps = 500
save_steps = 500

# Long training
eval_steps = 1000
save_steps = 1000

# Rule: 5-10 evaluations per epoch is good
```

#### `logging_steps`
**What it does**: How often to log training metrics

**How to choose**:
```python
# Quick feedback
logging_steps = 10

# Standard
logging_steps = 100

# Less noise
logging_steps = 200

# Rule: Should see logs every 1-2 minutes
```

---

## 🎛️ Scenario-Based Parameter Selection

### 🚀 Scenario 1: Quick Experiment (GPU: 8GB, Time: 1 hour)
```python
config = TrainingConfig(
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    eval_steps=100,
    save_steps=100,
    logging_steps=50,
    fp16=True,
    gradient_checkpointing=False,
)
```

### 📊 Scenario 2: Standard Training (GPU: 16GB, Time: 4 hours)
```python
config = TrainingConfig(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    bf16=True,
    gradient_checkpointing=False,
    early_stopping_patience=3,
)
```

### 🎯 Scenario 3: Production Quality (GPU: 24GB+, Time: 12+ hours)
```python
config = TrainingConfig(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    eval_steps=1000,
    save_steps=1000,
    logging_steps=200,
    bf16=True,
    gradient_checkpointing=False,
    early_stopping_patience=5,
    weight_decay=0.01,
)
```

### 💾 Scenario 4: Memory Constrained (GPU: 4GB, Any time)
```python
config = TrainingConfig(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size = 16
    learning_rate=5e-4,
    eval_steps=200,
    save_steps=200,
    logging_steps=50,
    fp16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=2,
)
```

### ⚡ Scenario 5: Time Constrained (Any GPU, Time: 30 min)
```python
config = TrainingConfig(
    num_train_epochs=1,
    per_device_train_batch_size=32,  # Max your GPU can handle
    gradient_accumulation_steps=1,
    learning_rate=1e-3,  # Higher LR for faster convergence
    eval_steps=50,
    save_steps=50,
    logging_steps=25,
    bf16=True,
    gradient_checkpointing=False,
)
```

---

## 🔧 Advanced Parameter Tuning

### Learning Rate Scheduling
```python
# For longer training
lr_scheduler_type = "cosine"  # Smooth decay
warmup_ratio = 0.1

# For shorter training  
lr_scheduler_type = "linear"  # Simple decay
warmup_ratio = 0.1

# For very short training
lr_scheduler_type = "constant"  # No decay
warmup_ratio = 0.0
```

### Early Stopping Strategy
```python
# Aggressive (stop early, save time)
early_stopping_patience = 2
early_stopping_threshold = 0.001

# Conservative (train longer, better results)
early_stopping_patience = 5
early_stopping_threshold = 0.0001

# Disable early stopping
early_stopping_patience = 0
```

### Evaluation Strategy
```python
# For small datasets
evaluation_strategy = "epoch"  # Evaluate after each epoch

# For large datasets
evaluation_strategy = "steps"  # Evaluate every N steps
eval_steps = 500
```

---

## 🎯 Parameter Selection Checklist

Before training, ask yourself:

### ✅ Resource Constraints
- [ ] How much GPU memory do I have?
- [ ] How much time can I spend?
- [ ] Do I have multiple GPUs?

### ✅ Dataset Characteristics  
- [ ] How large is my dataset?
- [ ] How complex is my task?
- [ ] Do I expect overfitting?

### ✅ Quality Requirements
- [ ] Is this a quick experiment or production model?
- [ ] How much accuracy improvement do I need?
- [ ] Can I afford longer training time?

### ✅ Monitoring Needs
- [ ] Do I need frequent updates?
- [ ] Am I using wandb/tensorboard?
- [ ] Do I need to save multiple checkpoints?

---

## 🚨 Common Mistakes to Avoid

1. **Batch size too large**: Causes OOM errors
2. **Learning rate too high**: Model doesn't converge
3. **Learning rate too low**: Training is too slow
4. **No warmup**: Unstable training start
5. **Too frequent evaluation**: Slows down training
6. **No early stopping**: Wastes time on overfitting
7. **Wrong precision type**: Compatibility issues

---

## 💡 Pro Tips

1. **Start small**: Begin with QUICK_TRAINING_CONFIG and adjust
2. **Monitor GPU usage**: Use `nvidia-smi` to check memory usage
3. **Use wandb**: Track experiments to compare parameters
4. **Save configurations**: Keep track of what works
5. **Gradual scaling**: Increase complexity step by step

This guide should help you make informed decisions about parameter selection based on your specific constraints and goals!
