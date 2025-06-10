"""
Configuration for multitask fine-tuning
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class TaskSamplingStrategy(Enum):
    """Task sampling strategies"""
    PROPORTIONAL = "proportional"  # Sample based on dataset size
    EQUAL = "equal"  # Equal sampling from all tasks
    TEMPERATURE = "temperature"  # Temperature-based sampling
    CURRICULUM = "curriculum"  # Curriculum learning
    DYNAMIC = "dynamic"  # Dynamic sampling based on performance


class LossWeightingStrategy(Enum):
    """Loss weighting strategies"""
    EQUAL = "equal"  # Equal weights for all tasks
    PROPORTIONAL = "proportional"  # Weights based on dataset size
    UNCERTAINTY = "uncertainty"  # Uncertainty-based weighting
    GRADIENT_NORM = "gradient_norm"  # Gradient norm balancing
    DYNAMIC = "dynamic"  # Dynamic weight adjustment
    MANUAL = "manual"  # Manually specified weights


class DataMixingStrategy(Enum):
    """Data mixing strategies"""
    BATCH_LEVEL = "batch_level"  # Mix tasks at batch level
    EXAMPLE_LEVEL = "example_level"  # Mix tasks at example level
    ROUND_ROBIN = "round_robin"  # Round-robin task selection
    RANDOM = "random"  # Random task selection


class ArchitectureType(Enum):
    """Model architecture types"""
    SHARED_BOTTOM = "shared_bottom"  # Shared encoder + task heads
    MULTI_GATE_MOE = "multi_gate_moe"  # Multi-gate mixture of experts
    CROSS_STITCH = "cross_stitch"  # Cross-stitch networks
    TASK_CLUSTERING = "task_clustering"  # Task clustering approach


@dataclass
class MultitaskConfig:
    """Configuration for multitask fine-tuning"""
    
    # Model configuration
    model_name_or_path: str = "bert-base-uncased"
    architecture_type: ArchitectureType = ArchitectureType.SHARED_BOTTOM
    
    # Task configuration
    tasks: Dict[str, Any] = field(default_factory=dict)
    primary_task: Optional[str] = None  # Main task for evaluation
    
    # Sampling configuration
    task_sampling_strategy: TaskSamplingStrategy = TaskSamplingStrategy.PROPORTIONAL
    sampling_temperature: float = 1.0
    sampling_alpha: float = 0.75  # For proportional sampling
    
    # Loss configuration
    loss_weighting_strategy: LossWeightingStrategy = LossWeightingStrategy.EQUAL
    task_weights: Optional[Dict[str, float]] = None
    loss_balancing_alpha: float = 0.16  # For uncertainty weighting
    
    # Data mixing configuration
    data_mixing_strategy: DataMixingStrategy = DataMixingStrategy.BATCH_LEVEL
    max_examples_per_task: Optional[int] = None
    
    # Training configuration
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Batch configuration
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    
    # Evaluation configuration
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Task-specific evaluation
    eval_all_tasks: bool = True
    eval_primary_task_only: bool = False
    
    # Advanced features
    use_task_embeddings: bool = False
    task_embedding_dim: int = 64
    
    # Gradient surgery
    use_gradient_surgery: bool = False
    gradient_surgery_alpha: float = 0.5
    
    # Progressive training
    use_progressive_training: bool = False
    progressive_schedule: Optional[Dict[str, List[str]]] = None
    
    # Meta-learning
    use_meta_learning: bool = False
    meta_learning_lr: float = 1e-3
    meta_batch_size: int = 4
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    hidden_dropout_rate: float = 0.1
    
    # Efficiency settings
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Output configuration
    output_dir: str = "./multitask_results"
    save_total_limit: int = 3
    
    def __post_init__(self):
        # Convert string enums to enum objects
        if isinstance(self.task_sampling_strategy, str):
            self.task_sampling_strategy = TaskSamplingStrategy(self.task_sampling_strategy)
        
        if isinstance(self.loss_weighting_strategy, str):
            self.loss_weighting_strategy = LossWeightingStrategy(self.loss_weighting_strategy)
        
        if isinstance(self.data_mixing_strategy, str):
            self.data_mixing_strategy = DataMixingStrategy(self.data_mixing_strategy)
        
        if isinstance(self.architecture_type, str):
            self.architecture_type = ArchitectureType(self.architecture_type)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if not self.tasks:
            raise ValueError("At least one task must be specified")
        
        if self.primary_task and self.primary_task not in self.tasks:
            raise ValueError(f"Primary task '{self.primary_task}' not found in tasks")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.sampling_temperature <= 0:
            raise ValueError("Sampling temperature must be positive")
        
        if self.task_weights:
            if set(self.task_weights.keys()) != set(self.tasks.keys()):
                raise ValueError("Task weights must be specified for all tasks")
            
            if any(w <= 0 for w in self.task_weights.values()):
                raise ValueError("All task weights must be positive")
    
    def get_task_names(self) -> List[str]:
        """Get list of task names"""
        return list(self.tasks.keys())
    
    def get_task_sampling_probabilities(self, dataset_sizes: Dict[str, int]) -> Dict[str, float]:
        """Get task sampling probabilities based on strategy"""
        if self.task_sampling_strategy == TaskSamplingStrategy.EQUAL:
            num_tasks = len(self.tasks)
            return {task: 1.0 / num_tasks for task in self.tasks}
        
        elif self.task_sampling_strategy == TaskSamplingStrategy.PROPORTIONAL:
            total_size = sum(dataset_sizes.values())
            probs = {}
            for task, size in dataset_sizes.items():
                # Apply alpha smoothing
                prob = (size / total_size) ** self.sampling_alpha
                probs[task] = prob
            
            # Normalize
            total_prob = sum(probs.values())
            return {task: prob / total_prob for task, prob in probs.items()}
        
        elif self.task_sampling_strategy == TaskSamplingStrategy.TEMPERATURE:
            import math
            sizes = list(dataset_sizes.values())
            # Apply temperature scaling
            scaled_sizes = [math.exp(s / self.sampling_temperature) for s in sizes]
            total_scaled = sum(scaled_sizes)
            
            probs = {}
            for task, scaled_size in zip(dataset_sizes.keys(), scaled_sizes):
                probs[task] = scaled_size / total_scaled
            
            return probs
        
        else:
            # Default to equal sampling
            num_tasks = len(self.tasks)
            return {task: 1.0 / num_tasks for task in self.tasks}
    
    def get_task_weights(self, dataset_sizes: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """Get task loss weights based on strategy"""
        if self.loss_weighting_strategy == LossWeightingStrategy.MANUAL and self.task_weights:
            return self.task_weights.copy()
        
        elif self.loss_weighting_strategy == LossWeightingStrategy.EQUAL:
            return {task: 1.0 for task in self.tasks}
        
        elif self.loss_weighting_strategy == LossWeightingStrategy.PROPORTIONAL and dataset_sizes:
            total_size = sum(dataset_sizes.values())
            weights = {}
            for task, size in dataset_sizes.items():
                # Inverse proportional weighting (smaller datasets get higher weights)
                weights[task] = total_size / (size * len(dataset_sizes))
            return weights
        
        else:
            # Default to equal weights
            return {task: 1.0 for task in self.tasks}
    
    def get_progressive_schedule(self) -> Dict[str, List[str]]:
        """Get progressive training schedule"""
        if self.progressive_schedule:
            return self.progressive_schedule
        
        # Default progressive schedule: start with one task, add one per phase
        task_names = self.get_task_names()
        schedule = {}
        
        for i, task in enumerate(task_names):
            phase_name = f"phase_{i + 1}"
            schedule[phase_name] = task_names[:i + 1]
        
        return schedule
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            elif isinstance(value, dict) and value:
                # Handle nested configs
                config_dict[key] = {}
                for k, v in value.items():
                    if hasattr(v, 'to_dict'):
                        config_dict[key][k] = v.to_dict()
                    else:
                        config_dict[key][k] = v
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MultitaskConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined multitask configurations
CLASSIFICATION_MULTITASK_CONFIG = MultitaskConfig(
    model_name_or_path="bert-base-uncased",
    task_sampling_strategy=TaskSamplingStrategy.PROPORTIONAL,
    loss_weighting_strategy=LossWeightingStrategy.EQUAL,
    data_mixing_strategy=DataMixingStrategy.BATCH_LEVEL,
    learning_rate=2e-5,
    per_device_train_batch_size=16
)

QA_SUMMARIZATION_CONFIG = MultitaskConfig(
    model_name_or_path="t5-base",
    task_sampling_strategy=TaskSamplingStrategy.TEMPERATURE,
    sampling_temperature=2.0,
    loss_weighting_strategy=LossWeightingStrategy.UNCERTAINTY,
    data_mixing_strategy=DataMixingStrategy.EXAMPLE_LEVEL,
    learning_rate=1e-4,
    per_device_train_batch_size=8
)

PROGRESSIVE_MULTITASK_CONFIG = MultitaskConfig(
    model_name_or_path="roberta-base",
    task_sampling_strategy=TaskSamplingStrategy.CURRICULUM,
    loss_weighting_strategy=LossWeightingStrategy.DYNAMIC,
    use_progressive_training=True,
    use_gradient_surgery=True,
    learning_rate=3e-5
)

META_LEARNING_CONFIG = MultitaskConfig(
    model_name_or_path="bert-base-uncased",
    task_sampling_strategy=TaskSamplingStrategy.DYNAMIC,
    loss_weighting_strategy=LossWeightingStrategy.GRADIENT_NORM,
    use_meta_learning=True,
    meta_learning_lr=1e-3,
    meta_batch_size=4,
    learning_rate=2e-5
)

EFFICIENT_MULTITASK_CONFIG = MultitaskConfig(
    model_name_or_path="distilbert-base-uncased",
    task_sampling_strategy=TaskSamplingStrategy.EQUAL,
    loss_weighting_strategy=LossWeightingStrategy.EQUAL,
    use_mixed_precision=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2
)
