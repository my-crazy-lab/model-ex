# 🚀 Hướng Dẫn Implement LoRA/PEFT Từ Số 0

Hướng dẫn này sẽ giúp bạn hiểu và xây dựng lại toàn bộ hệ thống LoRA/PEFT từ đầu, từng bước một.

## 📚 Kiến Thức Cần Có Trước

### 1. Python Cơ Bản
- Classes và Objects
- Dataclasses
- Type hints
- Import/Export modules

### 2. Machine Learning Cơ Bản
- Tokenization (chuyển text thành số)
- Neural Networks cơ bản
- Training/Validation/Test sets

### 3. Thư Viện Cần Biết
- `transformers`: Load models từ Hugging Face
- `datasets`: Load và xử lý data
- `torch`: PyTorch framework
- `peft`: Parameter Efficient Fine-Tuning

---

## 🏗️ Bước 1: Hiểu Kiến Trúc Tổng Thể

### Tại Sao Cần Nhiều Files?
```
Nguyên tắc: "Separation of Concerns" - Mỗi file làm 1 việc cụ thể

config/     → Cấu hình (như setting trong game)
data/       → Xử lý dữ liệu (load và clean data)
models/     → Quản lý AI models
training/   → Logic huấn luyện model
evaluation/ → Đánh giá model
inference/  → Sử dụng model đã train
examples/   → Ví dụ cách sử dụng
```

### Luồng Hoạt Động
```
1. Config → Thiết lập tham số
2. Data → Load và xử lý dữ liệu
3. Model → Tạo model với PEFT
4. Training → Huấn luyện model
5. Evaluation → Đánh giá kết quả
6. Inference → Sử dụng model
```

---

## 🔧 Bước 2: Implement Config System

### Tại Sao Cần Config?
```python
# Thay vì hard-code như này:
model = AutoModel.from_pretrained("bert-base-uncased")
batch_size = 16
learning_rate = 0.001

# Ta dùng config để dễ thay đổi:
config = ModelConfig(
    model_name="bert-base-uncased",
    batch_size=16,
    learning_rate=0.001
)
```

### 2.1 Tạo `config/__init__.py`
```python
"""
File này cho Python biết config là một package
Và export những class chính để dễ import
"""
from .model_config import ModelConfig, PEFTConfig
from .training_config import TrainingConfig

__all__ = ["ModelConfig", "PEFTConfig", "TrainingConfig"]
```

### 2.2 Tạo `config/model_config.py`
```python
"""
Cấu hình cho model và PEFT
"""
from dataclasses import dataclass
from typing import List, Optional
from peft import TaskType

@dataclass
class ModelConfig:
    """Cấu hình cho base model"""
    
    # Tên model từ Hugging Face Hub
    model_name_or_path: str = "bert-base-uncased"
    
    # Số lượng labels (cho classification)
    num_labels: int = 2
    
    # Độ dài tối đa của sequence
    max_length: int = 512
    
    # Có dùng quantization không (tiết kiệm memory)
    use_quantization: bool = False
    quantization_bits: int = 4  # 4-bit hoặc 8-bit

@dataclass  
class PEFTConfig:
    """Cấu hình cho PEFT methods"""
    
    # Loại PEFT: LORA, PREFIX_TUNING, etc.
    peft_type: str = "LORA"
    
    # Loại task: classification, generation, etc.
    task_type: TaskType = TaskType.SEQ_CLS
    
    # LoRA parameters
    r: int = 16              # Rank - càng nhỏ càng ít parameters
    lora_alpha: int = 32     # Scaling factor
    lora_dropout: float = 0.1 # Dropout để tránh overfitting
    
    # Modules nào sẽ apply LoRA
    target_modules: Optional[List[str]] = None
```

**Giải thích chi tiết:**
- `@dataclass`: Tự động tạo `__init__`, `__repr__` methods
- `Optional[List[str]]`: Có thể là None hoặc list of strings
- `TaskType.SEQ_CLS`: Enum từ thư viện PEFT

### 2.3 Tạo `config/training_config.py`
```python
"""
Cấu hình cho quá trình training
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TrainingConfig:
    """Cấu hình training"""
    
    # Thư mục lưu kết quả
    output_dir: str = "./results"
    
    # Training parameters
    num_train_epochs: int = 3           # Số epochs
    per_device_train_batch_size: int = 8 # Batch size
    learning_rate: float = 5e-4         # Learning rate
    
    # Evaluation
    evaluation_strategy: str = "steps"   # Khi nào evaluate
    eval_steps: int = 500               # Evaluate mỗi 500 steps
    
    # Logging
    logging_steps: int = 100            # Log mỗi 100 steps
    
    def to_training_arguments(self) -> Dict[str, Any]:
        """Convert sang format của Hugging Face Trainer"""
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "learning_rate": self.learning_rate,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
        }
```

---

## 📊 Bước 3: Implement Data Processing

### Tại Sao Cần Data Processing?
```
Raw text: "This movie is great!"
↓ Tokenization
Token IDs: [101, 2023, 3185, 2003, 2307, 999, 102]
↓ Padding/Truncation
Fixed length: [101, 2023, 3185, 2003, 2307, 999, 102, 0, 0, 0]
```

### 3.1 Tạo `data/__init__.py`
```python
"""
Export main data classes
"""
from .data_loader import DataLoader, load_dataset_from_hub
from .preprocessing import TextClassificationPreprocessor

__all__ = [
    "DataLoader", 
    "load_dataset_from_hub",
    "TextClassificationPreprocessor"
]
```

### 3.2 Tạo `data/data_loader.py`
```python
"""
Load datasets từ nhiều nguồn khác nhau
"""
import os
import json
from typing import Union, Optional
from datasets import Dataset, DatasetDict, load_dataset

class DataLoader:
    """Class để load datasets"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
    
    def load_from_hub(self, dataset_name: str) -> DatasetDict:
        """Load dataset từ Hugging Face Hub"""
        try:
            dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)
            print(f"✅ Loaded dataset: {dataset_name}")
            return dataset
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            raise
    
    def load_from_json(self, file_path: str) -> Dataset:
        """Load dataset từ JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Dataset.from_list(data)

def load_dataset_from_hub(dataset_name: str) -> DatasetDict:
    """Convenience function"""
    loader = DataLoader()
    return loader.load_from_hub(dataset_name)
```

**Giải thích:**
- `DatasetDict`: Dictionary chứa train/validation/test splits
- `Dataset`: Một split cụ thể
- `cache_dir`: Thư mục cache để không download lại

### 3.3 Tạo `data/preprocessing.py`
```python
"""
Preprocessing data cho các tasks khác nhau
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
            batched=True,  # Xử lý theo batch để nhanh hơn
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
            return_tensors=None     # Return Python lists, not tensors
        )
        
        # Add labels
        if self.label_column in examples:
            result["labels"] = examples[self.label_column]
        
        return result
```

**Giải thích:**
- `ABC`: Abstract Base Class - class cha không thể instantiate trực tiếp
- `@abstractmethod`: Method bắt buộc phải implement
- `batched=True`: Xử lý nhiều examples cùng lúc
- `padding=True`: Thêm padding tokens để cùng độ dài

---

## 🤖 Bước 4: Implement Model Wrappers

### Tại Sao Cần Wrappers?
```python
# Thay vì code phức tạp:
model = AutoModelForSequenceClassification.from_pretrained(...)
if use_quantization:
    model = quantize_model(model)
peft_config = LoraConfig(...)
model = get_peft_model(model, peft_config)

# Ta có wrapper đơn giản:
wrapper = PEFTModelWrapper(model_config, peft_config)
model = wrapper.load_model()
```

### 4.1 Tạo `models/__init__.py`
```python
"""
Export model classes
"""
from .base_model import BaseModelWrapper
from .peft_model import PEFTModelWrapper

__all__ = ["BaseModelWrapper", "PEFTModelWrapper"]
```

### 4.2 Tạo `models/base_model.py`
```python
"""
Wrapper cho base models
"""
import torch
from typing import Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import TaskType

from ..config.model_config import ModelConfig

class BaseModelWrapper:
    """Wrapper cho base model"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_tokenizer(self):
        """Load tokenizer"""
        if self.tokenizer is None:
            print(f"Loading tokenizer: {self.config.model_name_or_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path
            )
            
            # Add pad token nếu chưa có
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ Tokenizer loaded")
        
        return self.tokenizer
    
    def load_model(self, task_type: TaskType):
        """Load model based on task type"""
        if self.model is None:
            print(f"Loading model: {self.config.model_name_or_path}")
            
            # Prepare arguments
            model_kwargs = {}
            
            # Add quantization nếu cần
            if self.config.use_quantization:
                model_kwargs["quantization_config"] = self._get_quantization_config()
                model_kwargs["torch_dtype"] = torch.float16
            
            # Load model based on task
            if task_type == TaskType.SEQ_CLS:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name_or_path,
                    num_labels=self.config.num_labels,
                    **model_kwargs
                )
            
            print(f"✅ Model loaded")
        
        return self.model
    
    def _get_quantization_config(self):
        """Tạo quantization config"""
        if self.config.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.config.quantization_bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
```

**Giải thích:**
- `TaskType.SEQ_CLS`: Sequence Classification
- `quantization_config`: Cấu hình để giảm memory usage
- `torch.float16`: Half precision để tiết kiệm memory

---

## ⏰ Tạm Dừng - Checkpoint 1

Đến đây bạn đã hiểu:
1. ✅ Tại sao cần chia code thành nhiều files
2. ✅ Cách tạo config system
3. ✅ Cách load và preprocess data
4. ✅ Cách wrap models

**Tiếp theo**: Chúng ta sẽ implement PEFT wrapper, training system, và evaluation.

---

## 🧩 Bước 5: Implement PEFT Model Wrapper

### Tại Sao Cần PEFT?
```
Full Fine-tuning: Train toàn bộ 110M parameters của BERT
LoRA: Chỉ train 0.3M parameters mới (99.7% ít hơn!)

Kết quả: Gần như same accuracy, nhưng:
- Ít memory hơn
- Train nhanh hơn
- Dễ share model hơn (chỉ cần share LoRA weights)
```

### 5.1 Tạo `models/peft_model.py`
```python
"""
PEFT Model Wrapper
"""
import torch
from typing import Optional
from peft import get_peft_model, LoraConfig, TaskType

from .base_model import BaseModelWrapper
from ..config.model_config import ModelConfig, PEFTConfig

class PEFTModelWrapper:
    """Wrapper cho PEFT models"""

    def __init__(self, model_config: ModelConfig, peft_config: PEFTConfig):
        self.model_config = model_config
        self.peft_config = peft_config
        self.base_wrapper = BaseModelWrapper(model_config)
        self.peft_model = None

    def load_model(self):
        """Load và setup PEFT model"""
        if self.peft_model is None:
            print("Setting up PEFT model...")

            # Load base model
            base_model = self.base_wrapper.load_model(self.peft_config.task_type)

            # Create PEFT config
            peft_config = self._create_lora_config()

            # Apply PEFT
            self.peft_model = get_peft_model(base_model, peft_config)

            # Print trainable parameters
            self._print_trainable_parameters()

            print("✅ PEFT model ready")

        return self.peft_model

    def _create_lora_config(self):
        """Tạo LoRA configuration"""

        # Tự động detect target modules based on model type
        target_modules = self._get_target_modules()

        return LoraConfig(
            task_type=self.peft_config.task_type,
            r=self.peft_config.r,                    # Rank
            lora_alpha=self.peft_config.lora_alpha,  # Scaling
            lora_dropout=self.peft_config.lora_dropout,
            target_modules=target_modules,
            bias="none",  # Không train bias
        )

    def _get_target_modules(self):
        """Tự động chọn target modules based on model type"""
        model_name = self.model_config.model_name_or_path.lower()

        if "bert" in model_name:
            return ["query", "value"]  # BERT attention layers
        elif "llama" in model_name:
            return ["q_proj", "v_proj"]  # LLaMA attention layers
        elif "gpt" in model_name:
            return ["c_attn"]  # GPT attention layers
        else:
            return ["query", "value"]  # Default

    def _print_trainable_parameters(self):
        """In số lượng parameters trainable"""
        trainable_params = 0
        all_params = 0

        for _, param in self.peft_model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = 100 * trainable_params / all_params

        print(f"Trainable params: {trainable_params:,}")
        print(f"All params: {all_params:,}")
        print(f"Trainable%: {percentage:.2f}%")

    def get_tokenizer(self):
        """Get tokenizer"""
        return self.base_wrapper.load_tokenizer()

    def save_peft_model(self, save_directory: str):
        """Save chỉ PEFT weights (rất nhỏ!)"""
        if self.peft_model is not None:
            self.peft_model.save_pretrained(save_directory)
            print(f"✅ PEFT model saved to {save_directory}")
```

**Giải thích:**
- `get_peft_model()`: Function từ thư viện PEFT để wrap model
- `target_modules`: Layers nào sẽ được apply LoRA
- `r` (rank): Càng nhỏ càng ít parameters, nhưng có thể giảm performance
- `lora_alpha`: Scaling factor, thường = 2 * r

---

## 🏋️ Bước 6: Implement Training System

### Training Flow
```
1. Setup model + data
2. Create Trainer với callbacks
3. Train với monitoring
4. Save best model
```

### 6.1 Tạo `training/__init__.py`
```python
"""
Training module exports
"""
from .trainer import PEFTTrainer
from .utils import setup_logging, compute_metrics

__all__ = ["PEFTTrainer", "setup_logging", "compute_metrics"]
```

### 6.2 Tạo `training/utils.py`
```python
"""
Training utilities
"""
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def setup_logging():
    """Setup logging cho training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def compute_metrics(eval_pred):
    """Compute metrics cho evaluation"""
    predictions, labels = eval_pred

    # Convert logits to predictions
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        "accuracy": accuracy,
        "f1": f1,
    }
```

### 6.3 Tạo `training/trainer.py`
```python
"""
Main PEFT Trainer
"""
import os
from typing import Optional
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from ..models.peft_model import PEFTModelWrapper
from ..data.preprocessing import DataPreprocessor
from ..config.model_config import ModelConfig, PEFTConfig
from ..config.training_config import TrainingConfig
from .utils import setup_logging, compute_metrics

class PEFTTrainer:
    """Main trainer class"""

    def __init__(
        self,
        model_config: ModelConfig,
        peft_config: PEFTConfig,
        training_config: TrainingConfig,
        task_type: str = "classification"
    ):
        self.model_config = model_config
        self.peft_config = peft_config
        self.training_config = training_config
        self.task_type = task_type

        # Setup logging
        setup_logging()

        # Initialize components
        self.model_wrapper = None
        self.trainer = None

    def setup_model(self):
        """Setup PEFT model"""
        if self.model_wrapper is None:
            self.model_wrapper = PEFTModelWrapper(
                self.model_config,
                self.peft_config
            )
            self.model_wrapper.load_model()

        return self.model_wrapper

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        preprocessor: Optional[DataPreprocessor] = None
    ):
        """Main training function"""

        print("🚀 Starting PEFT training...")

        # Setup model
        self.setup_model()
        model = self.model_wrapper.peft_model
        tokenizer = self.model_wrapper.get_tokenizer()

        # Preprocess datasets
        if preprocessor is not None:
            print("📊 Preprocessing datasets...")
            train_dataset = preprocessor.preprocess_dataset(train_dataset)
            if eval_dataset is not None:
                eval_dataset = preprocessor.preprocess_dataset(eval_dataset)

        # Create training arguments
        training_args = TrainingArguments(
            **self.training_config.to_training_arguments()
        )

        # Create data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True
        )

        # Create trainer
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Start training
        print("🏋️ Training started...")
        train_result = self.trainer.train()

        # Save model
        self.save_model()

        print("✅ Training completed!")
        return train_result

    def save_model(self):
        """Save trained model"""
        save_dir = self.training_config.output_dir

        # Save PEFT model
        peft_save_dir = os.path.join(save_dir, "peft_model")
        self.model_wrapper.save_peft_model(peft_save_dir)

        # Save tokenizer
        tokenizer = self.model_wrapper.get_tokenizer()
        tokenizer.save_pretrained(save_dir)

        print(f"💾 Model saved to {save_dir}")
```

**Giải thích:**
- `TrainingArguments`: Cấu hình cho Hugging Face Trainer
- `DataCollatorWithPadding`: Tự động pad sequences trong batch
- `compute_metrics`: Function để tính accuracy, F1 score
- `self.trainer.train()`: Bắt đầu training loop

---

## 📊 Bước 7: Implement Evaluation System

### 7.1 Tạo `evaluation/__init__.py`
```python
"""
Evaluation module
"""
from .evaluator import ModelEvaluator

__all__ = ["ModelEvaluator"]
```

### 7.2 Tạo `evaluation/evaluator.py`
```python
"""
Model evaluation
"""
import torch
import numpy as np
from typing import Dict, Any
from datasets import Dataset

from ..models.peft_model import PEFTModelWrapper

class ModelEvaluator:
    """Model evaluator"""

    def __init__(self, model_wrapper: PEFTModelWrapper):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.peft_model
        self.tokenizer = model_wrapper.get_tokenizer()

    def evaluate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Evaluate model trên dataset"""

        print("📊 Evaluating model...")

        # Set model to eval mode
        self.model.eval()

        all_predictions = []
        all_labels = []

        # Process dataset
        for example in dataset:
            # Tokenize input
            inputs = self.tokenizer(
                example["text"],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1)

            all_predictions.append(prediction.item())
            all_labels.append(example["label"])

        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

        results = {
            "accuracy": accuracy,
            "num_examples": len(dataset),
        }

        print(f"✅ Evaluation completed: {results}")
        return results
```

---

## 🚀 Bước 8: Implement Inference Pipeline

### 8.1 Tạo `inference/__init__.py`
```python
"""
Inference module
"""
from .pipeline import InferencePipeline

__all__ = ["InferencePipeline"]
```

### 8.2 Tạo `inference/pipeline.py`
```python
"""
Production inference pipeline
"""
import torch
from typing import List, Union
from transformers import pipeline

from ..models.peft_model import PEFTModelWrapper

class InferencePipeline:
    """Easy-to-use inference pipeline"""

    def __init__(self, model_wrapper: PEFTModelWrapper):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.peft_model
        self.tokenizer = model_wrapper.get_tokenizer()

        # Create HF pipeline
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def classify_text(self, texts: Union[str, List[str]]):
        """Classify single text or list of texts"""

        if isinstance(texts, str):
            # Single text
            result = self.classifier(texts)
            return result[0]
        else:
            # Multiple texts
            results = self.classifier(texts)
            return results

    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load từ saved model"""
        # Implementation để load saved model
        pass
```

---

## 🎯 Bước 9: Tạo Example Đơn Giản

### 9.1 Tạo `examples/simple_example.py`
```python
"""
Ví dụ đơn giản nhất
"""
from datasets import load_dataset

# Import our modules
from config import ModelConfig, PEFTConfig, TrainingConfig
from data import TextClassificationPreprocessor
from training import PEFTTrainer
from peft import TaskType

def main():
    """Ví dụ train model classification đơn giản"""

    print("🚀 Starting simple LoRA example...")

    # 1. Setup configs
    model_config = ModelConfig(
        model_name_or_path="distilbert-base-uncased",  # Model nhỏ để test
        num_labels=2,
        max_length=128  # Ngắn để train nhanh
    )

    peft_config = PEFTConfig(
        peft_type="LORA",
        task_type=TaskType.SEQ_CLS,
        r=8,  # Rank nhỏ
        lora_alpha=16
    )

    training_config = TrainingConfig(
        output_dir="./simple_results",
        num_train_epochs=1,  # 1 epoch để test nhanh
        per_device_train_batch_size=16,
        learning_rate=1e-3,
        eval_steps=100,
        logging_steps=50
    )

    # 2. Load dataset
    print("📊 Loading dataset...")
    dataset = load_dataset("imdb")

    # Take small subset để test nhanh
    train_dataset = dataset["train"].select(range(100))
    eval_dataset = dataset["test"].select(range(50))

    # 3. Setup trainer
    trainer = PEFTTrainer(
        model_config=model_config,
        peft_config=peft_config,
        training_config=training_config
    )

    # 4. Setup preprocessor
    model_wrapper = trainer.setup_model()
    tokenizer = model_wrapper.get_tokenizer()

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

    print("✅ Example completed!")

if __name__ == "__main__":
    main()
```

---

## 🎉 Hoàn Thành - Bạn Đã Có Hệ Thống Hoàn Chỉnh!

### Tóm Tắt Những Gì Đã Implement:

1. ✅ **Config System**: Quản lý tham số dễ dàng
2. ✅ **Data Processing**: Load và preprocess data
3. ✅ **Model Wrappers**: Wrap models với PEFT
4. ✅ **Training System**: Train models với monitoring
5. ✅ **Evaluation**: Đánh giá model performance
6. ✅ **Inference**: Sử dụng model đã train
7. ✅ **Example**: Ví dụ hoàn chỉnh

### Cách Chạy:
```bash
cd lora-peft
python examples/simple_example.py
```

### Hiểu Được Gì:
- Tại sao chia code thành modules
- Cách PEFT/LoRA hoạt động
- Cách training pipeline hoạt động
- Cách integrate các components

### Bước Tiếp Theo:
1. Chạy example để thấy kết quả
2. Thử thay đổi parameters
3. Thử datasets khác
4. Implement thêm features

**Chúc mừng! Bạn đã hiểu và implement được một hệ thống LoRA/PEFT hoàn chỉnh từ số 0! 🎉**
