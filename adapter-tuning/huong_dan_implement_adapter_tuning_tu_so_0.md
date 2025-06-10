# 🔧 Hướng Dẫn Implement Adapter Tuning Từ Số 0

Hướng dẫn này sẽ giúp bạn hiểu và xây dựng lại toàn bộ hệ thống Adapter Tuning từ đầu, từng bước một.

## 📚 Kiến Thức Cần Có Trước

### 1. Python Cơ Bản
- Classes và Objects
- Inheritance và Abstract classes
- Type hints
- Context managers

### 2. Deep Learning Cơ Bản
- Neural Networks
- Transformer architecture
- Fine-tuning vs Transfer learning
- Parameter-efficient methods

### 3. Thư Viện Cần Biết
- `torch`: PyTorch framework
- `transformers`: Hugging Face transformers
- `datasets`: Data loading và processing
- `sklearn`: Metrics calculation

---

## 🎯 Adapter Tuning Là Gì?

### Vấn Đề Với Full Fine-tuning
```
BERT-base: 110M parameters
Fine-tune cho 1 task → Train toàn bộ 110M parameters
Fine-tune cho 10 tasks → 10 × 110M = 1.1B parameters storage!
```

### Giải Pháp: Adapter Tuning
```
Base Model: 110M parameters (freeze - không train)
Adapter: 0.5M parameters (train)
→ Chỉ cần 0.45% parameters để đạt performance tương tự!
```

### Kiến Trúc Adapter
```
Input (768 dim) 
    ↓
Down-projection (768 → 64)  # Bottleneck
    ↓
Activation (ReLU/GELU)
    ↓
Up-projection (64 → 768)
    ↓
Residual Connection (+)
    ↓
Output (768 dim)
```

---

## 🏗️ Bước 1: Hiểu Kiến Trúc Tổng Thể

### Tại Sao Cần Nhiều Files?
```
Nguyên tắc: "Single Responsibility Principle"

config/         → Quản lý cấu hình
adapters/       → Core adapter implementation
data/           → Data loading và preprocessing
training/       → Training logic
evaluation/     → Model evaluation
inference/      → Production inference
examples/       → Ví dụ sử dụng
```

### Luồng Hoạt Động
```
1. Config → Thiết lập parameters
2. Adapters → Tạo adapter modules
3. Data → Load và preprocess data
4. Training → Train adapters (freeze base model)
5. Evaluation → Đánh giá performance
6. Inference → Sử dụng model đã train
```

---

## 🔧 Bước 2: Implement Core Adapter Layer

### 2.1 Hiểu Bottleneck Architecture

**Tại sao dùng Bottleneck?**
```python
# Thay vì train full linear layer:
# 768 × 768 = 589,824 parameters

# Ta dùng bottleneck:
# Down: 768 × 64 = 49,152 parameters
# Up: 64 × 768 = 49,152 parameters
# Total: 98,304 parameters (6x ít hơn!)
```

### 2.2 Tạo `adapters/adapter_layer.py`

```python
"""
Core adapter implementation - Trái tim của hệ thống
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckAdapter(nn.Module):
    """
    Adapter với kiến trúc bottleneck
    
    Luồng: Input → Down → Activation → Up → Residual → Output
    """
    
    def __init__(
        self,
        input_size: int,      # Kích thước input (768 cho BERT)
        adapter_size: int,    # Kích thước bottleneck (64, 128, etc.)
        dropout: float = 0.1, # Dropout để tránh overfitting
        activation: str = "relu"  # Activation function
    ):
        super().__init__()
        
        # Lưu parameters
        self.input_size = input_size
        self.adapter_size = adapter_size
        
        # Down-projection: giảm dimension
        self.down_project = nn.Linear(input_size, adapter_size)
        
        # Up-projection: tăng dimension về ban đầu
        self.up_project = nn.Linear(adapter_size, input_size)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights"""
        # Initialize down projection normally
        nn.init.normal_(self.down_project.weight, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        
        # Initialize up projection to ZERO - quan trọng!
        # Điều này đảm bảo adapter ban đầu không ảnh hưởng đến model
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass qua adapter
        
        Args:
            hidden_states: [batch_size, seq_len, input_size]
        Returns:
            output: [batch_size, seq_len, input_size]
        """
        # Lưu input cho residual connection
        residual = hidden_states
        
        # Down-projection
        hidden_states = self.down_project(hidden_states)
        
        # Activation
        hidden_states = self.activation(hidden_states)
        
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # Up-projection
        hidden_states = self.up_project(hidden_states)
        
        # Residual connection - QUAN TRỌNG!
        # Đảm bảo gradient flow và stable training
        output = residual + hidden_states
        
        return output
```

**Giải thích chi tiết:**
- `down_project`: Giảm dimension từ 768 → 64 (compression)
- `activation`: Non-linearity để học complex patterns
- `up_project`: Tăng dimension từ 64 → 768 (decompression)
- `residual`: Skip connection để stable training
- `_init_weights()`: Initialize up_project = 0 để adapter ban đầu không ảnh hưởng

---

## 🤖 Bước 3: Implement Model Wrapper

### 3.1 Tại Sao Cần Model Wrapper?

```python
# Thay vì modify trực tiếp transformer:
model = AutoModel.from_pretrained("bert-base-uncased")
# Phải manually add adapters vào từng layer...

# Ta dùng wrapper để tự động:
adapter_model = AdapterModel(model_config, adapter_config)
# Tự động add adapters vào đúng vị trí!
```

### 3.2 Tạo `adapters/adapter_model.py`

```python
"""
Model wrapper để add adapters vào pre-trained models
"""
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class AdapterModel(nn.Module):
    """Wrapper để add adapters vào transformer model"""
    
    def __init__(self, model_config, adapter_config):
        super().__init__()
        
        # Load base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_name_or_path,
            num_labels=model_config.num_labels
        )
        
        # Store configs
        self.model_config = model_config
        self.adapter_config = adapter_config
        
        # Add adapters to model
        self._add_adapters()
        
        # Freeze base model parameters
        if adapter_config.freeze_base_model:
            self._freeze_base_model()
    
    def _add_adapters(self):
        """Add adapters to transformer layers"""
        
        # Get encoder layers (BERT example)
        if hasattr(self.base_model, 'bert'):
            encoder_layers = self.base_model.bert.encoder.layer
        else:
            raise ValueError("Unsupported model architecture")
        
        # Add adapter to each layer
        for layer_idx, layer in enumerate(encoder_layers):
            
            # Get hidden size from layer
            hidden_size = layer.output.dense.out_features
            
            # Create adapter
            adapter = BottleneckAdapter(
                input_size=hidden_size,
                adapter_size=self.adapter_config.adapter_size,
                dropout=self.adapter_config.adapter_dropout,
                activation=self.adapter_config.adapter_activation
            )
            
            # Add adapter to layer
            layer.adapter = adapter
            
            # Modify layer forward method
            self._modify_layer_forward(layer)
    
    def _modify_layer_forward(self, layer):
        """Modify layer forward để include adapter"""
        
        # Lưu original forward method
        original_forward = layer.forward
        
        def forward_with_adapter(hidden_states, attention_mask=None, **kwargs):
            # Gọi original forward
            outputs = original_forward(hidden_states, attention_mask, **kwargs)
            
            # Extract hidden states
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs
            
            # Apply adapter
            if hasattr(layer, 'adapter'):
                hidden_states = layer.adapter(hidden_states)
            
            # Return in original format
            if isinstance(outputs, tuple):
                return (hidden_states,) + outputs[1:]
            else:
                return hidden_states
        
        # Replace forward method
        layer.forward = forward_with_adapter
    
    def _freeze_base_model(self):
        """Freeze base model parameters"""
        for name, param in self.base_model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False
    
    def forward(self, **kwargs):
        """Forward pass through model"""
        return self.base_model(**kwargs)
    
    def get_adapter_parameters(self):
        """Get only adapter parameters for optimization"""
        return [p for n, p in self.named_parameters() if "adapter" in n]
    
    def print_adapter_info(self):
        """Print adapter information"""
        total_params = sum(p.numel() for p in self.parameters())
        adapter_params = sum(p.numel() for p in self.get_adapter_parameters())
        
        print(f"Total parameters: {total_params:,}")
        print(f"Adapter parameters: {adapter_params:,}")
        print(f"Adapter percentage: {adapter_params/total_params*100:.2f}%")
```

**Giải thích chi tiết:**
- `_add_adapters()`: Tự động detect và add adapters vào mỗi transformer layer
- `_modify_layer_forward()`: Modify forward method để call adapter sau original computation
- `_freeze_base_model()`: Freeze tất cả parameters trừ adapters
- `get_adapter_parameters()`: Lấy chỉ adapter parameters để optimize

---

## ⏰ Tạm Dừng - Checkpoint 1

Đến đây bạn đã hiểu:
1. ✅ Adapter architecture và tại sao nó hiệu quả
2. ✅ Cách implement bottleneck adapter
3. ✅ Cách integrate adapter vào pre-trained model
4. ✅ Parameter freezing và selective training

**Tiếp theo**: Chúng ta sẽ implement data processing, training system, và evaluation.

---

## 📊 Bước 4: Implement Configuration System

### 4.1 Tại Sao Cần Config System?

```python
# Thay vì hard-code:
adapter = BottleneckAdapter(768, 64, 0.1, "relu")
model = AutoModel.from_pretrained("bert-base-uncased")

# Ta dùng config để dễ thay đổi:
adapter_config = AdapterConfig(adapter_size=64, dropout=0.1)
model_config = ModelConfig(model_name="bert-base-uncased")
```

### 4.2 Tạo `config/adapter_config.py`

```python
"""
Configuration cho adapters
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AdapterConfig:
    """Configuration cho adapter modules"""

    # Basic adapter parameters
    adapter_size: int = 64              # Bottleneck dimension
    adapter_dropout: float = 0.1        # Dropout rate
    adapter_activation: str = "relu"    # Activation function

    # Adapter placement
    adapter_location: str = "both"      # "attention", "feedforward", "both"
    adapter_layers: Optional[List[int]] = None  # Which layers (None = all)

    # Training settings
    freeze_base_model: bool = True      # Freeze base model
    train_adapter_only: bool = True     # Train only adapters

    # Advanced options
    use_residual: bool = True           # Use residual connections
    adapter_init_range: float = 1e-3   # Weight initialization range

    def __post_init__(self):
        """Validate configuration"""
        valid_activations = ["relu", "gelu", "swish", "tanh"]
        if self.adapter_activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")

        valid_locations = ["attention", "feedforward", "both"]
        if self.adapter_location not in valid_locations:
            raise ValueError(f"location must be one of {valid_locations}")

    def get_adapter_size(self, hidden_size: int) -> int:
        """Calculate adapter size based on hidden size"""
        return self.adapter_size

    def should_add_adapter(self, layer_idx: int) -> bool:
        """Check if should add adapter to specific layer"""
        if self.adapter_layers is None:
            return True
        return layer_idx in self.adapter_layers
```

### 4.3 Tạo `config/model_config.py`

```python
"""
Configuration cho base models
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration cho base model"""

    # Model identification
    model_name_or_path: str = "bert-base-uncased"
    tokenizer_name_or_path: Optional[str] = None

    # Model parameters
    num_labels: int = 2                 # Number of output labels
    max_length: int = 512               # Max sequence length

    # Task configuration
    task_type: str = "classification"   # Task type

    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
```

**Giải thích:**
- `@dataclass`: Tự động tạo `__init__`, `__repr__` methods
- `__post_init__`: Validation sau khi tạo object
- Type hints: Giúp IDE hiểu và check types

---

## 📊 Bước 5: Implement Data Processing

### 5.1 Tại Sao Cần Data Processing?

```
Raw text: "This movie is great!"
↓ Tokenization
Token IDs: [101, 2023, 3185, 2003, 2307, 999, 102]
↓ Padding/Truncation
Fixed length: [101, 2023, 3185, 2003, 2307, 999, 102, 0, 0, 0]
↓ Attention mask
Mask: [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
```

### 5.2 Tạo `data/preprocessing.py`

```python
"""
Data preprocessing cho adapter tuning
"""
from abc import ABC, abstractmethod
from typing import Dict, List
from datasets import Dataset
from transformers import PreTrainedTokenizer

class DataPreprocessor(ABC):
    """Abstract base class cho preprocessing"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    @abstractmethod
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Method này phải được implement bởi subclasses"""
        pass

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Apply preprocessing lên toàn bộ dataset"""
        return dataset.map(
            self.preprocess_function,
            batched=True,
            desc="Preprocessing dataset"
        )

class TextClassificationPreprocessor(DataPreprocessor):
    """Preprocessor cho text classification"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "text",
        label_column: str = "label",
        max_length: int = 512
    ):
        super().__init__(tokenizer, max_length)
        self.text_column = text_column
        self.label_column = label_column

    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize texts và prepare labels"""

        # Tokenize texts
        result = self.tokenizer(
            examples[self.text_column],
            padding=True,           # Pad về cùng độ dài
            truncation=True,        # Cắt nếu quá dài
            max_length=self.max_length,
            return_tensors=None     # Return Python lists
        )

        # Add labels
        if self.label_column in examples:
            result["labels"] = examples[self.label_column]

        return result
```

**Giải thích:**
- `ABC`: Abstract Base Class - class cha không thể instantiate
- `@abstractmethod`: Method bắt buộc phải implement
- `batched=True`: Xử lý nhiều examples cùng lúc để nhanh hơn
- `padding=True`: Thêm padding tokens để cùng độ dài

---

## 🏋️ Bước 6: Implement Training System

### 6.1 Training Flow

```
1. Setup model + data
2. Create optimizer (chỉ cho adapter parameters)
3. Training loop với monitoring
4. Save adapters (không save base model)
```

### 6.2 Tạo `training/trainer.py`

```python
"""
Adapter trainer implementation
"""
from typing import Optional
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import torch.optim as optim

class AdapterTrainer:
    """Main trainer cho adapter tuning"""

    def __init__(
        self,
        model_config,
        adapter_config,
        training_config,
        task_type: str = "classification"
    ):
        self.model_config = model_config
        self.adapter_config = adapter_config
        self.training_config = training_config
        self.task_type = task_type

        # Initialize components
        self.adapter_model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """Setup adapter model và tokenizer"""
        if self.adapter_model is None:
            # Load tokenizer
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.tokenizer_name_or_path
            )

            # Add pad token nếu chưa có
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Create adapter model
            self.adapter_model = AdapterModel(
                self.model_config,
                self.adapter_config
            )

            print("✅ Adapter model setup completed")
            self.adapter_model.print_adapter_info()

        return self.adapter_model

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        preprocessor = None
    ):
        """Main training function"""

        print("🚀 Starting adapter training...")

        # Setup model
        self.setup_model()

        # Preprocess datasets
        if preprocessor is not None:
            print("📊 Preprocessing datasets...")
            train_dataset = preprocessor.preprocess_dataset(train_dataset)
            if eval_dataset is not None:
                eval_dataset = preprocessor.preprocess_dataset(eval_dataset)

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            learning_rate=self.training_config.learning_rate,
            evaluation_strategy="steps",
            eval_steps=500,
            logging_steps=100,
            save_steps=500,
        )

        # Create data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )

        # Create optimizer chỉ cho adapter parameters
        optimizer = optim.AdamW(
            self.adapter_model.get_adapter_parameters(),
            lr=self.training_config.learning_rate
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.adapter_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer, None)  # (optimizer, scheduler)
        )

        # Start training
        print("🏋️ Training started...")
        train_result = self.trainer.train()

        # Save adapters
        self.save_adapters()

        print("✅ Training completed!")
        return train_result

    def save_adapters(self):
        """Save chỉ adapter weights"""
        import os
        import torch

        save_dir = os.path.join(self.training_config.output_dir, "adapters")
        os.makedirs(save_dir, exist_ok=True)

        # Save adapter state dict
        adapter_state_dict = {}
        for name, param in self.adapter_model.named_parameters():
            if "adapter" in name:
                adapter_state_dict[name] = param

        torch.save(adapter_state_dict, os.path.join(save_dir, "adapter_model.bin"))

        # Save tokenizer
        self.tokenizer.save_pretrained(self.training_config.output_dir)

        print(f"💾 Adapters saved to {save_dir}")
```

**Giải thích:**
- `get_adapter_parameters()`: Chỉ lấy adapter parameters để optimize
- `DataCollatorWithPadding`: Tự động pad sequences trong batch
- `optimizers=(optimizer, None)`: Custom optimizer cho adapters
- Save chỉ adapter weights, không save base model

---

## 🎯 Bước 7: Tạo Example Đơn Giản

### 7.1 Tạo `examples/simple_adapter_example.py`

```python
"""
Ví dụ đơn giản nhất về adapter tuning
"""
from datasets import load_dataset

# Import our modules
from config import ModelConfig, AdapterConfig, TrainingConfig
from data import TextClassificationPreprocessor
from training import AdapterTrainer

def main():
    """Ví dụ train adapter cho sentiment analysis"""

    print("🚀 Starting simple adapter example...")

    # 1. Setup configs
    model_config = ModelConfig(
        model_name_or_path="distilbert-base-uncased",  # Model nhỏ để test
        num_labels=2,
        max_length=128  # Ngắn để train nhanh
    )

    adapter_config = AdapterConfig(
        adapter_size=32,  # Adapter nhỏ
        adapter_dropout=0.1,
        adapter_activation="relu",
        freeze_base_model=True
    )

    training_config = TrainingConfig(
        output_dir="./simple_results",
        num_train_epochs=1,  # 1 epoch để test nhanh
        per_device_train_batch_size=16,
        learning_rate=2e-3,  # Learning rate cao hơn cho adapters
    )

    # 2. Load dataset
    print("📊 Loading dataset...")
    dataset = load_dataset("imdb")

    # Take small subset để test nhanh
    train_dataset = dataset["train"].select(range(100))
    eval_dataset = dataset["test"].select(range(50))

    # 3. Setup trainer
    trainer = AdapterTrainer(
        model_config=model_config,
        adapter_config=adapter_config,
        training_config=training_config
    )

    # 4. Setup preprocessor
    adapter_model = trainer.setup_model()
    tokenizer = trainer.tokenizer

    preprocessor = TextClassificationPreprocessor(
        tokenizer=tokenizer,
        text_column="text",
        label_column="label",
        max_length=128
    )

    # 5. Train
    print("🏋️ Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocessor=preprocessor
    )

    # 6. Test inference
    print("🧪 Testing inference...")
    test_texts = [
        "This movie is amazing!",
        "Terrible film, waste of time."
    ]

    # Simple inference test
    adapter_model.eval()
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = adapter_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1)
            sentiment = "Positive" if prediction.item() == 1 else "Negative"
            print(f"'{text}' → {sentiment}")

    print("✅ Example completed!")

if __name__ == "__main__":
    main()
```

---

## 🎉 Hoàn Thành - Bạn Đã Có Hệ Thống Adapter Tuning!

### Tóm Tắt Những Gì Đã Implement:

1. ✅ **Core Adapter Layer**: Bottleneck architecture với residual connections
2. ✅ **Model Wrapper**: Tự động add adapters vào transformer layers
3. ✅ **Configuration System**: Quản lý parameters dễ dàng
4. ✅ **Data Processing**: Tokenization và preprocessing
5. ✅ **Training System**: Train chỉ adapters, freeze base model
6. ✅ **Example**: Ví dụ hoàn chỉnh có thể chạy được

### Cách Chạy:
```bash
cd adapter-tuning
python examples/simple_adapter_example.py
```

### Hiểu Được Gì:
- Adapter architecture và tại sao hiệu quả
- Cách integrate adapters vào pre-trained models
- Parameter freezing và selective training
- Training pipeline cho adapter tuning

### So Sánh Với Full Fine-tuning:
```
Full Fine-tuning:
- Train: 110M parameters
- Storage: 110M × số tasks
- Memory: Cao

Adapter Tuning:
- Train: 0.5M parameters (99.5% ít hơn!)
- Storage: 110M + 0.5M × số tasks
- Memory: Thấp hơn nhiều
- Performance: Tương đương!
```

### Bước Tiếp Theo:
1. Chạy example để thấy kết quả
2. Thử thay đổi adapter_size (16, 32, 64, 128)
3. Test với datasets khác
4. Implement multi-task learning
5. So sánh với LoRA

**Chúc mừng! Bạn đã hiểu và implement được Adapter Tuning từ số 0! 🎉**
