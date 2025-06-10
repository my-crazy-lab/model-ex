# ⚡ Hướng Dẫn Implement BitFit Từ Số 0

Hướng dẫn này sẽ giúp bạn hiểu và xây dựng lại toàn bộ hệ thống BitFit từ đầu, từng bước một.

## 📚 Kiến Thức Cần Có Trước

### 1. Parameter-Efficient Fine-tuning
- Full fine-tuning vs parameter-efficient methods
- Overfitting và catastrophic forgetting
- Transfer learning principles

### 2. Neural Network Architecture
- Bias vs weight parameters
- Transformer architecture (attention, feedforward, layer norm)
- Parameter counting và memory usage

### 3. PyTorch Fundamentals
- `requires_grad` và gradient computation
- Parameter freezing và selective training
- Optimizer và learning rate scheduling

---

## 🎯 BitFit Là Gì?

### Vấn Đề Với Full Fine-tuning
```
BERT-base: 110M parameters
Full Fine-tuning: Train tất cả 110M parameters
→ High memory usage
→ Slow training
→ Risk of overfitting
→ Catastrophic forgetting
```

### Giải Pháp: BitFit (Bias-only Fine-tuning)
```
BERT-base: 110M parameters
BitFit: Chỉ train bias parameters (~0.1M parameters)
→ 99.9% parameter reduction!
→ 10x faster training
→ 90% memory reduction
→ Competitive performance (90-95% of full FT)
```

### Tại Sao Bias Parameters Quan Trọng?
```python
# Linear layer: y = Wx + b
# W (weight): Học representations
# b (bias): Học task-specific adjustments

# Ví dụ:
# Pre-trained: "good" → [0.8, 0.2] (positive sentiment)
# Task-specific bias: [+0.1, -0.1] 
# Final output: [0.9, 0.1] (more confident positive)
```

---

## 🏗️ Bước 1: Hiểu Kiến Trúc BitFit

### Transformer Architecture và Bias Terms
```
BERT Layer:
├── Multi-Head Attention
│   ├── Query Linear: Wq, bq ← BIAS
│   ├── Key Linear: Wk, bk ← BIAS  
│   ├── Value Linear: Wv, bv ← BIAS
│   └── Output Linear: Wo, bo ← BIAS
├── Layer Norm: γ, β ← BIAS
├── Feed Forward
│   ├── Intermediate: W1, b1 ← BIAS
│   └── Output: W2, b2 ← BIAS
└── Layer Norm: γ, β ← BIAS
```

### BitFit Strategy
```
1. Freeze ALL weight parameters (W)
2. Train ONLY bias parameters (b)
3. Keep pre-trained representations intact
4. Learn task-specific adjustments via bias
```

---

## 🔧 Bước 2: Implement Parameter Management

### 2.1 Tạo `bitfit/parameter_utils.py`

```python
"""
Parameter utilities - Trái tim của BitFit
"""
import torch
import torch.nn as nn
from typing import Dict, List
import re

class ParameterUtils:
    """Utilities for parameter management"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count different types of parameters"""
        total_params = 0
        trainable_params = 0
        bias_params = 0
        weight_params = 0
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            
            if 'bias' in name:
                bias_params += param_count
            else:
                weight_params += param_count
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "bias": bias_params,
            "weight": weight_params,
            "frozen": total_params - trainable_params
        }
    
    @staticmethod
    def print_parameter_summary(model: nn.Module):
        """Print detailed parameter summary"""
        counts = ParameterUtils.count_parameters(model)
        
        print(f"Parameter Summary:")
        print(f"  Total: {counts['total']:,}")
        print(f"  Trainable: {counts['trainable']:,}")
        print(f"  Bias: {counts['bias']:,}")
        print(f"  Weight: {counts['weight']:,}")
        
        efficiency = counts['trainable'] / counts['total'] * 100
        reduction = counts['total'] / max(counts['trainable'], 1)
        
        print(f"  Efficiency: {efficiency:.4f}%")
        print(f"  Reduction: {reduction:.1f}x")

class BiasParameterManager:
    """Manages bias parameter identification and training"""
    
    def identify_bias_parameters(self, model: nn.Module) -> Dict[str, List[str]]:
        """Identify different types of bias parameters"""
        bias_params = {
            "attention_bias": [],
            "feedforward_bias": [],
            "layer_norm_bias": [],
            "classifier_bias": [],
            "other_bias": []
        }
        
        for name, param in model.named_parameters():
            if 'bias' not in name:
                continue
            
            # Categorize bias parameter
            bias_type = self._categorize_bias_parameter(name)
            bias_params[bias_type].append(name)
        
        return bias_params
    
    def _categorize_bias_parameter(self, param_name: str) -> str:
        """Categorize bias parameter by type"""
        name_lower = param_name.lower()
        
        # Attention bias patterns
        if re.search(r'attention.*bias|attn.*bias|query.*bias|key.*bias|value.*bias', name_lower):
            return "attention_bias"
        
        # Feedforward bias patterns
        if re.search(r'intermediate.*bias|output.*bias|ffn.*bias|mlp.*bias|dense.*bias', name_lower):
            return "feedforward_bias"
        
        # Layer norm bias patterns
        if re.search(r'layernorm.*bias|layer_norm.*bias|norm.*bias', name_lower):
            return "layer_norm_bias"
        
        # Classifier bias patterns
        if re.search(r'classifier.*bias|cls.*bias|head.*bias', name_lower):
            return "classifier_bias"
        
        return "other_bias"
    
    def setup_bias_training(self, model: nn.Module, config) -> Dict[str, int]:
        """Setup bias parameters for training"""
        bias_params = self.identify_bias_parameters(model)
        trainable_counts = {}
        
        # First, freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Then, enable training for selected bias types
        for bias_type, param_names in bias_params.items():
            if not self._should_train_bias_type(bias_type, config):
                continue
            
            trainable_count = 0
            for param_name in param_names:
                param = dict(model.named_parameters())[param_name]
                param.requires_grad = True
                trainable_count += param.numel()
            
            trainable_counts[bias_type] = trainable_count
            print(f"Enabled {trainable_count:,} {bias_type} parameters")
        
        return trainable_counts
    
    def _should_train_bias_type(self, bias_type: str, config) -> bool:
        """Check if bias type should be trained"""
        type_mapping = {
            "attention_bias": getattr(config, 'train_attention_bias', True),
            "feedforward_bias": getattr(config, 'train_feedforward_bias', True),
            "layer_norm_bias": getattr(config, 'train_layer_norm_bias', True),
            "classifier_bias": getattr(config, 'train_classifier_bias', True),
            "other_bias": True
        }
        
        return type_mapping.get(bias_type, False)
```

**Giải thích chi tiết:**
- `count_parameters()`: Đếm tổng số parameters và phân loại
- `identify_bias_parameters()`: Tìm và phân loại bias parameters
- `setup_bias_training()`: Freeze weights, enable bias training
- Regular expressions để identify parameter types

---

## 🤖 Bước 3: Implement BitFit Model

### 3.1 Tạo `bitfit/bitfit_model.py`

```python
"""
BitFit model wrapper
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from .parameter_utils import ParameterUtils, BiasParameterManager

class BitFitModel(nn.Module):
    """BitFit model wrapper"""
    
    def __init__(self, model_config, bitfit_config):
        super().__init__()
        
        self.model_config = model_config
        self.bitfit_config = bitfit_config
        
        # Load base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_name_or_path,
            num_labels=model_config.num_labels
        )
        
        # Initialize parameter manager
        self.parameter_manager = BiasParameterManager()
        
        # Setup BitFit training
        self._setup_bitfit_training()
        
        print("✅ BitFit model initialized")
    
    def _setup_bitfit_training(self):
        """Setup the model for BitFit training"""
        print("🔧 Setting up BitFit training...")
        
        # Setup bias parameter training
        trainable_counts = self.parameter_manager.setup_bias_training(
            self.base_model, self.bitfit_config
        )
        
        # Print summary
        ParameterUtils.print_parameter_summary(self.base_model)
        
        total_trainable = sum(trainable_counts.values())
        print(f"Total trainable bias parameters: {total_trainable:,}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through base model"""
        return self.base_model(*args, **kwargs)
    
    def get_parameter_efficiency(self) -> Dict[str, float]:
        """Get parameter efficiency metrics"""
        counts = ParameterUtils.count_parameters(self.base_model)
        
        return {
            "efficiency": counts["trainable"] / counts["total"] * 100,
            "reduction_factor": counts["total"] / max(counts["trainable"], 1),
            "trainable_params": counts["trainable"],
            "total_params": counts["total"]
        }
    
    def compare_with_full_finetuning(self):
        """Compare with full fine-tuning"""
        efficiency = self.get_parameter_efficiency()
        
        print("\n🔍 BitFit vs Full Fine-tuning:")
        print("=" * 40)
        print(f"Full FT trainable: {efficiency['total_params']:,} (100%)")
        print(f"BitFit trainable: {efficiency['trainable_params']:,} ({efficiency['efficiency']:.4f}%)")
        print(f"Parameter reduction: {efficiency['reduction_factor']:.1f}x")
        print(f"Memory reduction: ~{efficiency['reduction_factor']:.1f}x")
        print("=" * 40)
```

**Giải thích:**
- `BitFitModel`: Wrapper around pre-trained model
- `_setup_bitfit_training()`: Configure bias-only training
- `get_parameter_efficiency()`: Calculate efficiency metrics
- `compare_with_full_finetuning()`: Show improvement

---

## ⏰ Tạm Dừng - Checkpoint 1

Đến đây bạn đã hiểu:
1. ✅ BitFit concept và tại sao hiệu quả
2. ✅ Parameter identification và categorization
3. ✅ Bias-only training setup
4. ✅ Efficiency calculation và comparison

**Tiếp theo**: Chúng ta sẽ implement training system, optimizer, và complete workflow.

---

## 🏋️ Bước 4: Implement BitFit Training System

### 4.1 Tại Sao Cần Custom Training?

```python
# Standard training: Optimize tất cả parameters
optimizer = AdamW(model.parameters(), lr=1e-5)

# BitFit training: Chỉ optimize bias parameters
bias_params = [p for n, p in model.named_parameters() if 'bias' in n and p.requires_grad]
optimizer = AdamW(bias_params, lr=1e-3)  # Higher LR for bias
```

### 4.2 Tạo `training/bitfit_trainer.py`

```python
"""
BitFit trainer implementation
"""
import torch
from transformers import Trainer, TrainingArguments
from .parameter_utils import ParameterUtils

class BitFitTrainer:
    """Trainer for BitFit models"""

    def __init__(self, model, training_config, tokenizer=None):
        self.model = model
        self.training_config = training_config
        self.tokenizer = tokenizer
        self.trainer = None

    def setup_optimizer(self):
        """Setup optimizer for bias parameters only"""
        # Get only trainable (bias) parameters
        trainable_params = [
            p for p in self.model.base_model.parameters()
            if p.requires_grad
        ]

        print(f"🔧 Setting up optimizer for {len(trainable_params)} bias parameter groups")

        # Use higher learning rate for bias parameters
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=0.0,  # No weight decay for bias
            betas=(0.9, 0.999),
            eps=1e-8
        )

        return optimizer

    def train(self, train_dataset, eval_dataset=None, compute_metrics=None):
        """Train the BitFit model"""

        print("🚀 Starting BitFit training...")

        # Print model comparison before training
        self.model.compare_with_full_finetuning()

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            learning_rate=self.training_config.learning_rate,
            weight_decay=0.0,  # No weight decay for bias
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=100,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            seed=42,
            fp16=False,  # Disable for stability with small parameters
        )

        # Setup optimizer
        optimizer = self.setup_optimizer()

        # Create trainer
        self.trainer = Trainer(
            model=self.model.base_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None)  # (optimizer, scheduler)
        )

        # Train model
        print("🏋️ Training started...")
        train_result = self.trainer.train()

        # Save model
        self.save_model()

        print("✅ Training completed!")
        print(f"Training loss: {train_result.training_loss:.4f}")

        return train_result

    def evaluate(self, eval_dataset=None):
        """Evaluate the BitFit model"""
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print("📈 Evaluating model...")
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)

        print("Evaluation results:")
        for key, value in eval_result.items():
            print(f"  {key}: {value}")

        return eval_result

    def save_model(self):
        """Save the BitFit model"""
        print(f"💾 Saving model to {self.training_config.output_dir}")

        # Save only bias parameters (more efficient)
        bias_state_dict = {}
        for name, param in self.model.base_model.named_parameters():
            if param.requires_grad and 'bias' in name:
                bias_state_dict[name] = param.cpu()

        import os
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        torch.save(bias_state_dict,
                  os.path.join(self.training_config.output_dir, "bias_parameters.bin"))

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.training_config.output_dir)

        print(f"Saved {len(bias_state_dict)} bias parameters")
```

**Giải thích:**
- `setup_optimizer()`: Chỉ optimize bias parameters
- Higher learning rate cho bias (1e-3 vs 1e-5 for full FT)
- No weight decay cho bias parameters
- Save chỉ bias parameters để tiết kiệm storage

---

## 🎯 Bước 5: Complete Example

### 5.1 Tạo `examples/text_classification.py`

```python
"""
Complete BitFit example
"""
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def main():
    """Main BitFit example"""

    print("🚀 BitFit Text Classification Example")

    # Load dataset (small sample for demo)
    dataset = load_dataset("imdb")
    train_data = dataset["train"].select(range(1000))  # Small sample
    test_data = dataset["test"].select(range(200))

    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")

    # Setup configurations
    from config import ModelConfig, BitFitConfig, TrainingConfig

    model_config = ModelConfig(
        model_name_or_path="distilbert-base-uncased",
        num_labels=2,
        task_type="classification"
    )

    bitfit_config = BitFitConfig(
        freeze_all_weights=True,
        train_bias_only=True,
        bias_learning_rate=1e-3,
        train_attention_bias=True,
        train_feedforward_bias=True,
        train_layer_norm_bias=True,
        train_classifier_bias=True
    )

    training_config = TrainingConfig(
        output_dir="./bitfit_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=1e-3
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)

    # Preprocess data
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=128
        )

    train_dataset = train_data.map(preprocess_function, batched=True)
    test_dataset = test_data.map(preprocess_function, batched=True)

    # Create BitFit model
    from bitfit import BitFitModel
    model = BitFitModel(model_config, bitfit_config)

    # Create trainer
    from training import BitFitTrainer
    trainer = BitFitTrainer(model, training_config, tokenizer)

    # Train model
    train_result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Evaluate
    eval_result = trainer.evaluate()

    # Print results
    print("\n📊 RESULTS:")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Eval accuracy: {eval_result['eval_accuracy']:.4f}")

    # Show efficiency
    efficiency = model.get_parameter_efficiency()
    print(f"\n⚡ EFFICIENCY:")
    print(f"Parameter efficiency: {efficiency['efficiency']:.4f}%")
    print(f"Reduction factor: {efficiency['reduction_factor']:.1f}x")

    # Test inference
    print("\n🧪 TESTING INFERENCE:")
    test_texts = [
        "This movie is amazing!",
        "Terrible film, waste of time."
    ]

    model.base_model.eval()
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions).item()
            confidence = predictions[0][predicted_class].item()

        sentiment = "Positive" if predicted_class == 1 else "Negative"
        print(f"'{text}' → {sentiment} ({confidence:.3f})")

    print("\n✅ BitFit example completed!")

if __name__ == "__main__":
    import torch
    main()
```

---

## 🎉 Hoàn Thành - Bạn Đã Có Hệ Thống BitFit!

### Tóm Tắt Những Gì Đã Implement:

1. ✅ **Parameter Management**: Identify và categorize bias parameters
2. ✅ **BitFit Model**: Wrapper với bias-only training setup
3. ✅ **Training System**: Custom trainer cho bias parameters
4. ✅ **Efficiency Analysis**: Parameter counting và comparison
5. ✅ **Complete Example**: End-to-end BitFit workflow

### Cách Chạy:
```bash
cd bitfit
python examples/text_classification.py
```

### Hiệu Quả Đạt Được:
```
BERT-base (110M parameters):
- Full Fine-tuning: 110M trainable (100%)
- BitFit: 0.1M trainable (0.09%)
- Reduction: 1100x fewer parameters!
- Performance: 90-95% of full fine-tuning
- Training time: 5-10x faster
- Memory usage: 80-90% reduction
```

### So Sánh Methods:
```
Method          | Params | Performance | Speed | Memory
----------------|--------|-------------|-------|--------
Full FT         | 100%   | 100%        | 1x    | High
Adapter         | 1%     | 95-98%      | 3x    | Medium
LoRA            | 0.5%   | 96-99%      | 4x    | Medium
BitFit          | 0.1%   | 90-95%      | 8x    | Low
```

### Khi Nào Dùng BitFit:
- ✅ Limited computational resources
- ✅ Fast prototyping và experimentation
- ✅ Many tasks với shared base model
- ✅ Edge deployment với memory constraints
- ❌ Tasks requiring significant architectural changes

### Bước Tiếp Theo:
1. Chạy example để thấy kết quả
2. Thử different bias type combinations
3. Compare với Adapter và LoRA
4. Test trên multiple tasks (GLUE benchmark)
5. Experiment với different learning rates

**Chúc mừng! Bạn đã hiểu và implement được BitFit từ số 0! ⚡**
