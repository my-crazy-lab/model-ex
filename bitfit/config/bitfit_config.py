"""
BitFit-specific configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class BiasType(Enum):
    """Types of bias parameters to train"""
    ATTENTION_BIAS = "attention_bias"
    FEEDFORWARD_BIAS = "feedforward_bias"
    LAYER_NORM_BIAS = "layer_norm_bias"
    CLASSIFIER_BIAS = "classifier_bias"
    EMBEDDING_BIAS = "embedding_bias"
    ALL_BIAS = "all_bias"


@dataclass
class BitFitConfig:
    """Configuration for BitFit training"""
    
    # Basic BitFit settings
    freeze_all_weights: bool = True
    train_bias_only: bool = True
    
    # Bias type selection
    bias_types_to_train: List[BiasType] = field(
        default_factory=lambda: [BiasType.ALL_BIAS]
    )
    
    # Specific bias training flags
    train_attention_bias: bool = True
    train_feedforward_bias: bool = True
    train_layer_norm_bias: bool = True
    train_classifier_bias: bool = True
    train_embedding_bias: bool = False  # Usually not needed
    
    # Learning rate settings
    bias_learning_rate: float = 1e-3
    use_different_lr_for_bias_types: bool = False
    attention_bias_lr: Optional[float] = None
    feedforward_bias_lr: Optional[float] = None
    layer_norm_bias_lr: Optional[float] = None
    classifier_bias_lr: Optional[float] = None
    
    # Optimization settings
    bias_weight_decay: float = 0.0  # Usually no weight decay for bias
    bias_gradient_clipping: bool = True
    max_bias_grad_norm: float = 1.0
    
    # Advanced settings
    selective_bias_training: bool = False
    bias_layers_to_train: Optional[List[int]] = None  # Specific layers
    exclude_bias_patterns: List[str] = field(default_factory=list)
    include_bias_patterns: List[str] = field(default_factory=list)
    
    # Initialization settings
    reinitialize_bias: bool = False
    bias_init_std: float = 0.02
    bias_init_method: str = "normal"  # normal, zeros, ones
    
    # Regularization
    bias_dropout: float = 0.0
    apply_bias_noise: bool = False
    bias_noise_std: float = 0.01
    
    # Monitoring and analysis
    track_bias_gradients: bool = True
    log_bias_statistics: bool = True
    save_bias_only: bool = True  # Save only bias parameters
    
    # Compatibility settings
    compatible_with_deepspeed: bool = True
    compatible_with_fsdp: bool = True
    
    def __post_init__(self):
        # Convert string enums to enum objects
        if isinstance(self.bias_types_to_train[0], str):
            self.bias_types_to_train = [
                BiasType(bias_type) for bias_type in self.bias_types_to_train
            ]
        
        # Set individual learning rates if not specified
        if self.use_different_lr_for_bias_types:
            if self.attention_bias_lr is None:
                self.attention_bias_lr = self.bias_learning_rate
            if self.feedforward_bias_lr is None:
                self.feedforward_bias_lr = self.bias_learning_rate
            if self.layer_norm_bias_lr is None:
                self.layer_norm_bias_lr = self.bias_learning_rate * 0.5  # Lower for layer norm
            if self.classifier_bias_lr is None:
                self.classifier_bias_lr = self.bias_learning_rate * 2.0  # Higher for classifier
        
        # Validate settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if not self.train_bias_only and self.freeze_all_weights:
            raise ValueError("Cannot freeze all weights if not training bias only")
        
        if self.bias_learning_rate <= 0:
            raise ValueError("bias_learning_rate must be positive")
        
        if self.max_bias_grad_norm <= 0:
            raise ValueError("max_bias_grad_norm must be positive")
        
        if self.bias_init_method not in ["normal", "zeros", "ones", "uniform"]:
            raise ValueError("bias_init_method must be one of: normal, zeros, ones, uniform")
    
    def get_bias_learning_rates(self) -> Dict[str, float]:
        """Get learning rates for different bias types"""
        if not self.use_different_lr_for_bias_types:
            return {"default": self.bias_learning_rate}
        
        return {
            "attention_bias": self.attention_bias_lr,
            "feedforward_bias": self.feedforward_bias_lr,
            "layer_norm_bias": self.layer_norm_bias_lr,
            "classifier_bias": self.classifier_bias_lr,
        }
    
    def should_train_bias_type(self, bias_type: BiasType) -> bool:
        """Check if specific bias type should be trained"""
        if BiasType.ALL_BIAS in self.bias_types_to_train:
            return True
        
        return bias_type in self.bias_types_to_train
    
    def get_bias_patterns(self) -> Dict[str, List[str]]:
        """Get patterns for identifying bias parameters"""
        patterns = {
            "include": self.include_bias_patterns.copy(),
            "exclude": self.exclude_bias_patterns.copy()
        }
        
        # Add default patterns based on bias types
        if self.train_attention_bias:
            patterns["include"].extend([
                "attention.*.bias",
                "self_attn.*.bias",
                "attn.*.bias"
            ])
        
        if self.train_feedforward_bias:
            patterns["include"].extend([
                "intermediate.*.bias",
                "output.*.bias",
                "feed_forward.*.bias",
                "ffn.*.bias"
            ])
        
        if self.train_layer_norm_bias:
            patterns["include"].extend([
                "LayerNorm.bias",
                "layer_norm.bias",
                "norm.bias"
            ])
        
        if self.train_classifier_bias:
            patterns["include"].extend([
                "classifier.bias",
                "cls.bias",
                "head.bias"
            ])
        
        if self.train_embedding_bias:
            patterns["include"].extend([
                "embeddings.*.bias",
                "embed.*.bias"
            ])
        
        return patterns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list) and value and isinstance(value[0], Enum):
                config_dict[key] = [item.value for item in value]
            elif isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BitFitConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined BitFit configurations
STANDARD_BITFIT = BitFitConfig(
    freeze_all_weights=True,
    train_bias_only=True,
    bias_learning_rate=1e-3,
    train_attention_bias=True,
    train_feedforward_bias=True,
    train_layer_norm_bias=True,
    train_classifier_bias=True
)

MINIMAL_BITFIT = BitFitConfig(
    freeze_all_weights=True,
    train_bias_only=True,
    bias_learning_rate=5e-4,
    train_attention_bias=False,
    train_feedforward_bias=True,
    train_layer_norm_bias=False,
    train_classifier_bias=True
)

AGGRESSIVE_BITFIT = BitFitConfig(
    freeze_all_weights=True,
    train_bias_only=True,
    bias_learning_rate=2e-3,
    use_different_lr_for_bias_types=True,
    attention_bias_lr=1e-3,
    feedforward_bias_lr=2e-3,
    classifier_bias_lr=5e-3,
    bias_gradient_clipping=True,
    max_bias_grad_norm=0.5
)

SELECTIVE_BITFIT = BitFitConfig(
    freeze_all_weights=True,
    train_bias_only=True,
    selective_bias_training=True,
    bias_layers_to_train=[6, 7, 8, 9, 10, 11],  # Last 6 layers
    bias_learning_rate=1e-3
)

EXPERIMENTAL_BITFIT = BitFitConfig(
    freeze_all_weights=True,
    train_bias_only=True,
    bias_learning_rate=1e-3,
    reinitialize_bias=True,
    bias_init_method="normal",
    bias_init_std=0.01,
    apply_bias_noise=True,
    bias_noise_std=0.005,
    track_bias_gradients=True
)
