"""
Fusion-specific configuration for Adapter Fusion implementation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class FusionMethod(Enum):
    """Supported fusion methods"""
    ATTENTION = "attention"
    WEIGHTED = "weighted"
    GATING = "gating"
    CONCAT = "concat"
    HIERARCHICAL = "hierarchical"


class FusionStrategy(Enum):
    """Fusion training strategies"""
    SEQUENTIAL = "sequential"  # Train adapters first, then fusion
    JOINT = "joint"           # Train adapters and fusion together
    CONTINUAL = "continual"   # Add new adapters incrementally


@dataclass
class FusionConfig:
    """Configuration for adapter fusion"""
    
    # Basic fusion settings
    fusion_method: Union[FusionMethod, str] = FusionMethod.ATTENTION
    fusion_strategy: Union[FusionStrategy, str] = FusionStrategy.SEQUENTIAL
    
    # Adapter management
    adapter_names: List[str] = field(default_factory=list)
    adapter_paths: List[str] = field(default_factory=list)
    task_names: List[str] = field(default_factory=list)
    
    # Fusion layer configuration
    fusion_hidden_size: Optional[int] = None  # Auto-detect from adapters
    fusion_dropout: float = 0.1
    fusion_activation: str = "relu"
    
    # Attention-based fusion
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    use_attention_bias: bool = True
    attention_temperature: float = 1.0
    
    # Weighted fusion
    learnable_weights: bool = True
    weight_initialization: str = "uniform"  # uniform, normal, xavier
    weight_constraint: Optional[str] = "softmax"  # softmax, sigmoid, none
    
    # Gating fusion
    gate_activation: str = "sigmoid"  # sigmoid, tanh, relu
    gate_bias: bool = True
    gate_hidden_size: Optional[int] = None
    
    # Hierarchical fusion
    fusion_layers: List[int] = field(default_factory=lambda: [6, 12])  # Which layers to fuse
    layer_fusion_method: str = "attention"
    cross_layer_fusion: bool = False
    
    # Dynamic fusion
    dynamic_fusion: bool = False
    input_dependent_weights: bool = False
    context_size: int = 128
    
    # Training settings
    freeze_adapters_during_fusion: bool = True
    fusion_learning_rate: float = 1e-4
    adapter_learning_rate: float = 1e-5
    
    # Regularization
    fusion_weight_decay: float = 0.01
    adapter_dropout_during_fusion: float = 0.0
    fusion_layer_norm: bool = True
    
    # Task routing
    automatic_task_detection: bool = False
    task_embedding_size: int = 64
    use_task_embeddings: bool = False
    
    # Advanced options
    sparse_fusion: bool = False
    top_k_adapters: Optional[int] = None
    fusion_temperature: float = 1.0
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        # Convert string enums to enum objects
        if isinstance(self.fusion_method, str):
            self.fusion_method = FusionMethod(self.fusion_method)
        
        if isinstance(self.fusion_strategy, str):
            self.fusion_strategy = FusionStrategy(self.fusion_strategy)
        
        # Validate configurations
        self._validate_config()
        
        # Set defaults based on fusion method
        self._set_method_defaults()
    
    def _validate_config(self):
        """Validate fusion configuration"""
        # Check adapter configuration
        if self.adapter_paths and self.adapter_names:
            if len(self.adapter_paths) != len(self.adapter_names):
                raise ValueError("adapter_paths and adapter_names must have same length")
        
        # Check task configuration
        if self.task_names and self.adapter_names:
            if len(self.task_names) != len(self.adapter_names):
                raise ValueError("task_names and adapter_names must have same length")
        
        # Validate fusion method specific settings
        if self.fusion_method == FusionMethod.ATTENTION:
            if self.num_attention_heads <= 0:
                raise ValueError("num_attention_heads must be positive")
        
        if self.fusion_method == FusionMethod.WEIGHTED:
            valid_inits = ["uniform", "normal", "xavier"]
            if self.weight_initialization not in valid_inits:
                raise ValueError(f"weight_initialization must be one of {valid_inits}")
        
        if self.sparse_fusion and self.top_k_adapters is None:
            raise ValueError("top_k_adapters must be specified when sparse_fusion=True")
    
    def _set_method_defaults(self):
        """Set default values based on fusion method"""
        if self.fusion_method == FusionMethod.ATTENTION:
            if self.fusion_hidden_size is None:
                self.fusion_hidden_size = 768  # Default BERT size
        
        elif self.fusion_method == FusionMethod.GATING:
            if self.gate_hidden_size is None:
                self.gate_hidden_size = self.fusion_hidden_size or 768
        
        elif self.fusion_method == FusionMethod.HIERARCHICAL:
            if not self.fusion_layers:
                self.fusion_layers = [6, 12]  # Default for BERT-base
    
    def get_num_adapters(self) -> int:
        """Get number of adapters to fuse"""
        if self.adapter_names:
            return len(self.adapter_names)
        elif self.adapter_paths:
            return len(self.adapter_paths)
        elif self.task_names:
            return len(self.task_names)
        else:
            return 0
    
    def get_adapter_config(self, adapter_name: str) -> Dict[str, Any]:
        """Get configuration for specific adapter"""
        return {
            "name": adapter_name,
            "freeze_during_fusion": self.freeze_adapters_during_fusion,
            "dropout_during_fusion": self.adapter_dropout_during_fusion,
            "learning_rate": self.adapter_learning_rate,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FusionConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined fusion configurations
ATTENTION_FUSION_CONFIG = FusionConfig(
    fusion_method=FusionMethod.ATTENTION,
    num_attention_heads=8,
    attention_dropout=0.1,
    fusion_dropout=0.1,
    learnable_weights=True
)

WEIGHTED_FUSION_CONFIG = FusionConfig(
    fusion_method=FusionMethod.WEIGHTED,
    learnable_weights=True,
    weight_initialization="uniform",
    weight_constraint="softmax",
    fusion_dropout=0.1
)

GATING_FUSION_CONFIG = FusionConfig(
    fusion_method=FusionMethod.GATING,
    gate_activation="sigmoid",
    gate_bias=True,
    fusion_dropout=0.1
)

HIERARCHICAL_FUSION_CONFIG = FusionConfig(
    fusion_method=FusionMethod.HIERARCHICAL,
    fusion_layers=[3, 6, 9, 12],
    layer_fusion_method="attention",
    cross_layer_fusion=True,
    fusion_dropout=0.1
)

DYNAMIC_FUSION_CONFIG = FusionConfig(
    fusion_method=FusionMethod.ATTENTION,
    dynamic_fusion=True,
    input_dependent_weights=True,
    context_size=128,
    fusion_dropout=0.1
)

# Multi-task specific configurations
MULTI_TASK_FUSION_CONFIG = FusionConfig(
    fusion_method=FusionMethod.ATTENTION,
    fusion_strategy=FusionStrategy.SEQUENTIAL,
    automatic_task_detection=True,
    use_task_embeddings=True,
    task_embedding_size=64,
    fusion_dropout=0.1
)

CONTINUAL_LEARNING_CONFIG = FusionConfig(
    fusion_method=FusionMethod.WEIGHTED,
    fusion_strategy=FusionStrategy.CONTINUAL,
    freeze_adapters_during_fusion=True,
    learnable_weights=True,
    fusion_dropout=0.1
)
