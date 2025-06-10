"""
Training configuration for Adapter Fusion implementation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
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
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Optimization
    learning_rate: float = 1e-3  # Higher LR for adapters
    adapter_learning_rate: Optional[float] = None  # Separate LR for adapters
    fusion_learning_rate: Optional[float] = None   # Separate LR for fusion
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
    
    # Adapter-specific training
    freeze_base_model: bool = True
    train_adapters_only: bool = True
    adapter_warmup_steps: int = 0
    
    # Fusion-specific training
    fusion_training_strategy: str = "sequential"  # sequential, joint, continual
    freeze_adapters_during_fusion: bool = True
    fusion_warmup_steps: int = 0
    fusion_epochs: int = 2
    
    # Multi-task training
    multi_task_training: bool = False
    task_sampling_strategy: str = "proportional"  # proportional, uniform, temperature
    task_temperature: float = 1.0
    task_weights: Optional[Dict[str, float]] = None
    
    # Continual learning
    continual_learning: bool = False
    memory_replay: bool = False
    replay_buffer_size: int = 1000
    
    # Reproducibility
    seed: int = 42
    data_seed: Optional[int] = None
    
    # Advanced options
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    
    # Fusion-specific options
    save_adapters_separately: bool = True
    save_fusion_separately: bool = True
    adapter_checkpoint_dir: Optional[str] = None
    fusion_checkpoint_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        
        if self.data_seed is None:
            self.data_seed = self.seed
        
        if self.adapter_checkpoint_dir is None:
            self.adapter_checkpoint_dir = os.path.join(self.output_dir, "adapters")
        
        if self.fusion_checkpoint_dir is None:
            self.fusion_checkpoint_dir = os.path.join(self.output_dir, "fusion")
        
        # Set learning rates if not specified
        if self.adapter_learning_rate is None:
            self.adapter_learning_rate = self.learning_rate
        
        if self.fusion_learning_rate is None:
            self.fusion_learning_rate = self.learning_rate * 0.1  # Lower LR for fusion
    
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
    
    def get_fusion_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments specifically for fusion training"""
        args = self.to_training_arguments()
        args.update({
            "num_train_epochs": self.fusion_epochs,
            "learning_rate": self.fusion_learning_rate,
            "warmup_steps": self.fusion_warmup_steps,
            "output_dir": self.fusion_checkpoint_dir,
        })
        return args


# Predefined training configurations
QUICK_ADAPTER_TRAINING = TrainingConfig(
    num_train_epochs=1,
    per_device_train_batch_size=32,
    learning_rate=2e-3,
    eval_steps=100,
    save_steps=100,
    logging_steps=50,
    freeze_base_model=True,
    train_adapters_only=True
)

STANDARD_ADAPTER_TRAINING = TrainingConfig(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=1e-3,
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    freeze_base_model=True,
    train_adapters_only=True
)

INTENSIVE_ADAPTER_TRAINING = TrainingConfig(
    num_train_epochs=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    eval_steps=1000,
    save_steps=1000,
    logging_steps=200,
    early_stopping_patience=5,
    freeze_base_model=True,
    train_adapters_only=True
)

FUSION_TRAINING_CONFIG = TrainingConfig(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=1e-3,
    fusion_learning_rate=1e-4,
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    fusion_training_strategy="sequential",
    freeze_adapters_during_fusion=True,
    fusion_epochs=2,
    save_adapters_separately=True,
    save_fusion_separately=True
)

MULTI_TASK_TRAINING_CONFIG = TrainingConfig(
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=1e-3,
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    multi_task_training=True,
    task_sampling_strategy="proportional",
    freeze_base_model=True,
    train_adapters_only=True
)

CONTINUAL_LEARNING_CONFIG = TrainingConfig(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=1e-3,
    fusion_learning_rate=5e-5,
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    continual_learning=True,
    memory_replay=True,
    replay_buffer_size=1000,
    fusion_training_strategy="continual"
)
