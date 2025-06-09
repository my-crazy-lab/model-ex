"""
Training configuration for LoRA/PEFT implementation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    
    # Output and logging
    output_dir: str = "./results"
    logging_dir: Optional[str] = None
    run_name: Optional[str] = None
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Optimization
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Logging
    logging_strategy: str = "steps"
    logging_steps: int = 100
    report_to: Optional[str] = None  # "wandb", "tensorboard", None
    
    # Data processing
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = True
    
    # Mixed precision and optimization
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # Reproducibility
    seed: int = 42
    data_seed: Optional[int] = None
    
    # Advanced options
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    
    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        
        if self.data_seed is None:
            self.data_seed = self.seed
    
    def to_training_arguments(self) -> Dict[str, Any]:
        """Convert to transformers TrainingArguments format"""
        return {
            "output_dir": self.output_dir,
            "logging_dir": self.logging_dir,
            "run_name": self.run_name,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "logging_strategy": self.logging_strategy,
            "logging_steps": self.logging_steps,
            "report_to": self.report_to,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "remove_unused_columns": self.remove_unused_columns,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "ignore_data_skip": self.ignore_data_skip,
        }


# Predefined training configurations
QUICK_TRAINING_CONFIG = TrainingConfig(
    num_train_epochs=1,
    per_device_train_batch_size=16,
    learning_rate=1e-3,
    eval_steps=100,
    save_steps=100,
    logging_steps=50,
)

STANDARD_TRAINING_CONFIG = TrainingConfig(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-4,
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
)

INTENSIVE_TRAINING_CONFIG = TrainingConfig(
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    eval_steps=1000,
    save_steps=1000,
    logging_steps=200,
    early_stopping_patience=5,
)
