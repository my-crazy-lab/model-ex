"""
Configuration for model distillation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class DistillationType(Enum):
    """Types of distillation"""
    LOGIT = "logit"  # Logit-based distillation
    FEATURE = "feature"  # Feature-based distillation
    ATTENTION = "attention"  # Attention transfer
    PROGRESSIVE = "progressive"  # Progressive distillation
    COMBINED = "combined"  # Multiple techniques


class LossType(Enum):
    """Loss function types"""
    KL_DIVERGENCE = "kl_div"
    MSE = "mse"
    COSINE_SIMILARITY = "cosine"
    L1 = "l1"
    HUBER = "huber"


class TemperatureSchedule(Enum):
    """Temperature scheduling strategies"""
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_ANNEALING = "cosine_annealing"
    ADAPTIVE = "adaptive"


@dataclass
class DistillationConfig:
    """Configuration for model distillation"""
    
    # Core distillation settings
    distillation_type: DistillationType = DistillationType.LOGIT
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for task loss (alpha + beta should = 1.0)
    
    # Loss function settings
    distillation_loss_type: LossType = LossType.KL_DIVERGENCE
    feature_loss_type: LossType = LossType.MSE
    attention_loss_type: LossType = LossType.MSE
    
    # Temperature scheduling
    temperature_schedule: TemperatureSchedule = TemperatureSchedule.CONSTANT
    initial_temperature: float = 4.0
    final_temperature: float = 1.0
    temperature_decay_steps: int = 1000
    
    # Feature distillation settings
    feature_distillation_layers: Optional[List[int]] = None
    feature_loss_weight: float = 0.5
    feature_projection_dim: Optional[int] = None
    normalize_features: bool = True
    
    # Attention distillation settings
    attention_distillation_layers: Optional[List[int]] = None
    attention_loss_weight: float = 0.3
    attention_head_selection: str = "all"  # all, random, top_k
    num_attention_heads: Optional[int] = None
    
    # Progressive distillation settings
    progressive_stages: int = 3
    progressive_schedule: str = "linear"  # linear, exponential
    stage_transition_steps: int = 1000
    
    # Layer mapping (teacher layer -> student layer)
    layer_mapping: Optional[Dict[int, int]] = None
    
    # Training settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Evaluation settings
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Advanced settings
    use_soft_targets_only: bool = False
    adaptive_temperature: bool = False
    temperature_adaptation_rate: float = 0.01
    
    # Multi-task distillation
    multi_task: bool = False
    task_weights: Optional[Dict[str, float]] = None
    shared_distillation: bool = True
    
    # Compression integration
    apply_quantization: bool = False
    quantization_bits: int = 8
    apply_pruning: bool = False
    pruning_ratio: float = 0.1
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    
    # Efficiency settings
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        # Convert string enums to enum objects
        if isinstance(self.distillation_type, str):
            self.distillation_type = DistillationType(self.distillation_type)
        
        if isinstance(self.distillation_loss_type, str):
            self.distillation_loss_type = LossType(self.distillation_loss_type)
        
        if isinstance(self.feature_loss_type, str):
            self.feature_loss_type = LossType(self.feature_loss_type)
        
        if isinstance(self.attention_loss_type, str):
            self.attention_loss_type = LossType(self.attention_loss_type)
        
        if isinstance(self.temperature_schedule, str):
            self.temperature_schedule = TemperatureSchedule(self.temperature_schedule)
        
        # Validate settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        if not (0 <= self.alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")
        
        if not (0 <= self.beta <= 1):
            raise ValueError("Beta must be between 0 and 1")
        
        if abs(self.alpha + self.beta - 1.0) > 1e-6:
            raise ValueError("Alpha + Beta should equal 1.0")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.progressive_stages <= 0:
            raise ValueError("Progressive stages must be positive")
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for different components"""
        weights = {
            "distillation": self.alpha,
            "task": self.beta
        }
        
        if self.distillation_type in [DistillationType.FEATURE, DistillationType.COMBINED]:
            weights["feature"] = self.feature_loss_weight
        
        if self.distillation_type in [DistillationType.ATTENTION, DistillationType.COMBINED]:
            weights["attention"] = self.attention_loss_weight
        
        return weights
    
    def get_temperature_at_step(self, step: int) -> float:
        """Get temperature value at given training step"""
        if self.temperature_schedule == TemperatureSchedule.CONSTANT:
            return self.temperature
        
        elif self.temperature_schedule == TemperatureSchedule.LINEAR_DECAY:
            progress = min(step / self.temperature_decay_steps, 1.0)
            return self.initial_temperature - progress * (self.initial_temperature - self.final_temperature)
        
        elif self.temperature_schedule == TemperatureSchedule.EXPONENTIAL_DECAY:
            decay_rate = (self.final_temperature / self.initial_temperature) ** (1.0 / self.temperature_decay_steps)
            return self.initial_temperature * (decay_rate ** step)
        
        elif self.temperature_schedule == TemperatureSchedule.COSINE_ANNEALING:
            import math
            progress = min(step / self.temperature_decay_steps, 1.0)
            return self.final_temperature + 0.5 * (self.initial_temperature - self.final_temperature) * (1 + math.cos(math.pi * progress))
        
        else:  # ADAPTIVE
            return self.temperature  # Will be updated adaptively during training
    
    def get_layer_mapping(self, teacher_layers: int, student_layers: int) -> Dict[int, int]:
        """Get layer mapping between teacher and student"""
        if self.layer_mapping is not None:
            return self.layer_mapping
        
        # Default uniform mapping
        mapping = {}
        for i in range(student_layers):
            teacher_layer = int(i * teacher_layers / student_layers)
            mapping[teacher_layer] = i
        
        return mapping
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                config_dict[key] = [item.value for item in value]
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DistillationConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined distillation configurations
LOGIT_DISTILLATION_CONFIG = DistillationConfig(
    distillation_type=DistillationType.LOGIT,
    temperature=4.0,
    alpha=0.7,
    beta=0.3,
    learning_rate=5e-5
)

FEATURE_DISTILLATION_CONFIG = DistillationConfig(
    distillation_type=DistillationType.FEATURE,
    temperature=4.0,
    alpha=0.5,
    beta=0.5,
    feature_loss_weight=0.5,
    normalize_features=True,
    learning_rate=5e-5
)

ATTENTION_DISTILLATION_CONFIG = DistillationConfig(
    distillation_type=DistillationType.ATTENTION,
    temperature=4.0,
    alpha=0.6,
    beta=0.4,
    attention_loss_weight=0.3,
    attention_head_selection="all",
    learning_rate=5e-5
)

PROGRESSIVE_DISTILLATION_CONFIG = DistillationConfig(
    distillation_type=DistillationType.PROGRESSIVE,
    temperature=4.0,
    alpha=0.7,
    beta=0.3,
    progressive_stages=3,
    progressive_schedule="linear",
    stage_transition_steps=1000,
    learning_rate=5e-5
)

COMBINED_DISTILLATION_CONFIG = DistillationConfig(
    distillation_type=DistillationType.COMBINED,
    temperature=4.0,
    alpha=0.6,
    beta=0.4,
    feature_loss_weight=0.3,
    attention_loss_weight=0.2,
    normalize_features=True,
    learning_rate=5e-5
)

LIGHTWEIGHT_DISTILLATION_CONFIG = DistillationConfig(
    distillation_type=DistillationType.LOGIT,
    temperature=3.0,
    alpha=0.8,
    beta=0.2,
    learning_rate=1e-4,
    mixed_precision=True,
    gradient_checkpointing=True
)

MOBILE_DISTILLATION_CONFIG = DistillationConfig(
    distillation_type=DistillationType.LOGIT,
    temperature=5.0,
    alpha=0.9,
    beta=0.1,
    apply_quantization=True,
    quantization_bits=8,
    apply_pruning=True,
    pruning_ratio=0.2,
    learning_rate=1e-4
)
