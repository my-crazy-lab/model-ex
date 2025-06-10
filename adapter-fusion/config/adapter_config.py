"""
Adapter configuration for Adapter Fusion implementation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class AdapterConfig:
    """Configuration for individual adapters"""
    
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
    
    # Task-specific settings
    task_name: Optional[str] = None
    task_type: str = "classification"
    
    # Training settings
    freeze_base_model: bool = True
    train_adapter_only: bool = True
    adapter_learning_rate: Optional[float] = None
    
    # Fusion-specific settings
    fusion_compatible: bool = True
    adapter_id: Optional[str] = None
    
    def __post_init__(self):
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
        
        # Set adapter_id if not provided
        if self.adapter_id is None and self.task_name is not None:
            self.adapter_id = f"{self.task_name}_adapter"
    
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
            "task_name": self.task_name,
            "task_type": self.task_type,
            "freeze_base_model": self.freeze_base_model,
            "train_adapter_only": self.train_adapter_only,
            "adapter_learning_rate": self.adapter_learning_rate,
            "fusion_compatible": self.fusion_compatible,
            "adapter_id": self.adapter_id,
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
SENTIMENT_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=64,
    adapter_dropout=0.1,
    adapter_activation="relu",
    adapter_location="both",
    task_name="sentiment",
    task_type="classification"
)

NLI_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=64,
    adapter_dropout=0.1,
    adapter_activation="gelu",
    adapter_location="both",
    task_name="nli",
    task_type="classification"
)

NER_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=32,
    adapter_dropout=0.1,
    adapter_activation="relu",
    adapter_location="feedforward",
    task_name="ner",
    task_type="token_classification"
)

QA_ADAPTER_CONFIG = AdapterConfig(
    adapter_size=128,
    adapter_dropout=0.1,
    adapter_activation="gelu",
    adapter_location="both",
    task_name="qa",
    task_type="question_answering"
)

# Fusion-optimized configurations
FUSION_OPTIMIZED_CONFIG = AdapterConfig(
    adapter_size=64,
    adapter_dropout=0.05,  # Lower dropout for fusion
    adapter_activation="gelu",
    adapter_location="both",
    use_layer_norm=True,
    fusion_compatible=True
)

# Efficient configurations for large-scale fusion
EFFICIENT_FUSION_CONFIG = AdapterConfig(
    adapter_size=32,
    adapter_dropout=0.1,
    adapter_activation="relu",
    adapter_location="feedforward",
    adapter_reduction_factor=32,
    fusion_compatible=True
)

# Multi-task adapter configurations
MULTI_TASK_ADAPTER_CONFIGS = {
    "sentiment": AdapterConfig(
        adapter_size=64,
        task_name="sentiment",
        task_type="classification",
        adapter_id="sentiment_adapter"
    ),
    "nli": AdapterConfig(
        adapter_size=64,
        task_name="nli", 
        task_type="classification",
        adapter_id="nli_adapter"
    ),
    "ner": AdapterConfig(
        adapter_size=32,
        task_name="ner",
        task_type="token_classification",
        adapter_location="feedforward",
        adapter_id="ner_adapter"
    ),
    "qa": AdapterConfig(
        adapter_size=128,
        task_name="qa",
        task_type="question_answering",
        adapter_activation="gelu",
        adapter_id="qa_adapter"
    )
}
