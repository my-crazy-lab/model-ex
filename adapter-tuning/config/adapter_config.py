"""
Adapter-specific configuration for Adapter Tuning implementation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union


@dataclass
class AdapterConfig:
    """Configuration for adapter modules"""
    
    # Basic adapter parameters
    adapter_size: int = 64  # Bottleneck dimension
    adapter_dropout: float = 0.1
    adapter_activation: str = "relu"  # relu, gelu, swish, tanh
    
    # Adapter placement
    adapter_layers: Optional[List[int]] = None  # Which layers to add adapters (None = all)
    adapter_location: str = "both"  # "attention", "feedforward", "both"
    
    # Adapter architecture
    adapter_type: str = "bottleneck"  # bottleneck, parallel, sequential
    use_residual: bool = True
    use_layer_norm: bool = False
    
    # Advanced options
    adapter_reduction_factor: int = 16  # For automatic sizing: hidden_size // reduction_factor
    adapter_init_range: float = 1e-3
    adapter_scaling: float = 1.0
    
    # Multi-adapter settings
    adapter_names: Optional[List[str]] = None
    adapter_fusion: bool = False
    fusion_type: str = "attention"  # attention, average, weighted
    
    # Task-specific adapters
    task_adapters: Optional[Dict[str, "AdapterConfig"]] = None
    shared_adapters: bool = False
    
    # Training settings
    freeze_base_model: bool = True
    train_adapter_only: bool = True
    adapter_learning_rate: Optional[float] = None
    
    def __post_init__(self):
        # Set default adapter names if not provided
        if self.adapter_names is None:
            self.adapter_names = ["default"]
        
        # Validate adapter activation
        valid_activations = ["relu", "gelu", "swish", "tanh", "silu"]
        if self.adapter_activation not in valid_activations:
            raise ValueError(f"adapter_activation must be one of {valid_activations}")
        
        # Validate adapter location
        valid_locations = ["attention", "feedforward", "both"]
        if self.adapter_location not in valid_locations:
            raise ValueError(f"adapter_location must be one of {valid_locations}")
        
        # Validate adapter type
        valid_types = ["bottleneck", "parallel", "sequential"]
        if self.adapter_type not in valid_types:
            raise ValueError(f"adapter_type must be one of {valid_types}")
    
    def get_adapter_size(self, hidden_size: int) -> int:
        """Calculate adapter size based on hidden size and reduction factor"""
        if self.adapter_size > 0:
            return self.adapter_size
        else:
            return max(1, hidden_size // self.adapter_reduction_factor)
    
    def should_add_adapter(self, layer_idx: int) -> bool:
        """Check if adapter should be added to specific layer"""
        if self.adapter_layers is None:
            return True
        return layer_idx in self.adapter_layers
    
    def get_adapter_config_for_task(self, task_name: str) -> "AdapterConfig":
        """Get adapter config for specific task"""
        if self.task_adapters and task_name in self.task_adapters:
            return self.task_adapters[task_name]
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "adapter_size": self.adapter_size,
            "adapter_dropout": self.adapter_dropout,
            "adapter_activation": self.adapter_activation,
            "adapter_layers": self.adapter_layers,
            "adapter_location": self.adapter_location,
            "adapter_type": self.adapter_type,
            "use_residual": self.use_residual,
            "use_layer_norm": self.use_layer_norm,
            "adapter_reduction_factor": self.adapter_reduction_factor,
            "adapter_init_range": self.adapter_init_range,
            "adapter_scaling": self.adapter_scaling,
            "adapter_names": self.adapter_names,
            "adapter_fusion": self.adapter_fusion,
            "fusion_type": self.fusion_type,
            "freeze_base_model": self.freeze_base_model,
            "train_adapter_only": self.train_adapter_only,
            "adapter_learning_rate": self.adapter_learning_rate,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AdapterConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined adapter configurations
SMALL_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=32,
    adapter_dropout=0.1,
    adapter_activation="relu"
)

MEDIUM_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=64,
    adapter_dropout=0.1,
    adapter_activation="relu"
)

LARGE_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=128,
    adapter_dropout=0.1,
    adapter_activation="relu"
)

# Task-specific adapter configurations
CLASSIFICATION_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=64,
    adapter_dropout=0.1,
    adapter_activation="relu",
    adapter_location="both"
)

NER_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=32,
    adapter_dropout=0.1,
    adapter_activation="relu",
    adapter_location="feedforward"
)

QA_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=128,
    adapter_dropout=0.1,
    adapter_activation="gelu",
    adapter_location="both"
)

GENERATION_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=64,
    adapter_dropout=0.1,
    adapter_activation="gelu",
    adapter_location="feedforward"
)

# Multi-task adapter configuration
MULTI_TASK_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=64,
    adapter_dropout=0.1,
    adapter_activation="relu",
    adapter_names=["task1", "task2", "task3"],
    adapter_fusion=True,
    fusion_type="attention"
)

# Efficient adapter configuration (minimal parameters)
EFFICIENT_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=16,
    adapter_dropout=0.05,
    adapter_activation="relu",
    adapter_location="feedforward",
    adapter_reduction_factor=32
)
