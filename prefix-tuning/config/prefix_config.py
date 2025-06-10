"""
Configuration for prefix tuning
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class PrefixInitMethod(Enum):
    """Prefix initialization methods"""
    RANDOM = "random"
    UNIFORM = "uniform"
    NORMAL = "normal"
    XAVIER = "xavier"
    KAIMING = "kaiming"
    FROM_VOCAB = "from_vocab"
    FROM_TEXT = "from_text"


class PrefixPosition(Enum):
    """Where to apply prefixes"""
    INPUT_ONLY = "input_only"  # Only at input layer
    ALL_LAYERS = "all_layers"  # At all transformer layers
    SELECTED_LAYERS = "selected_layers"  # At specific layers


class ReparameterizationType(Enum):
    """Reparameterization strategies"""
    NONE = "none"  # Direct optimization
    MLP = "mlp"  # MLP reparameterization
    LSTM = "lstm"  # LSTM reparameterization
    TRANSFORMER = "transformer"  # Transformer reparameterization


@dataclass
class PrefixConfig:
    """Configuration for prefix tuning"""
    
    # Basic prefix settings
    prefix_length: int = 10
    prefix_hidden_size: Optional[int] = None  # If None, use model hidden size
    prefix_dropout: float = 0.1
    
    # Initialization settings
    init_method: PrefixInitMethod = PrefixInitMethod.RANDOM
    init_std: float = 0.02
    init_range: float = 0.1
    init_text: Optional[str] = None  # For text-based initialization
    init_vocab_size: int = 1000  # For vocab-based initialization
    
    # Position settings
    prefix_position: PrefixPosition = PrefixPosition.ALL_LAYERS
    selected_layers: Optional[List[int]] = None  # For selected layers
    
    # Reparameterization settings
    reparameterization: bool = True
    reparameterization_type: ReparameterizationType = ReparameterizationType.MLP
    reparameterization_hidden_size: int = 512
    reparameterization_num_layers: int = 2
    reparameterization_activation: str = "tanh"
    
    # Attention settings
    prefix_attention_heads: Optional[int] = None  # If None, use model num_heads
    prefix_key_value_heads: Optional[int] = None  # For GQA models
    
    # Training settings
    prefix_learning_rate: float = 1e-3
    prefix_weight_decay: float = 0.0
    prefix_gradient_clipping: float = 1.0
    
    # Advanced settings
    use_past_key_values: bool = True  # Use past key values for efficiency
    prefix_projection: bool = False  # Project prefix to different dimension
    prefix_projection_hidden_size: int = 256
    
    # Layer-specific settings
    different_prefix_per_layer: bool = True
    shared_prefix_across_heads: bool = False
    
    # Optimization settings
    freeze_base_model: bool = True
    trainable_base_modules: List[str] = field(default_factory=list)
    
    # Task-specific settings
    task_type: str = "classification"  # classification, generation, qa
    num_labels: Optional[int] = None
    
    # Efficiency settings
    low_cpu_mem_usage: bool = True
    use_cache: bool = True
    
    # Experimental settings
    prefix_tuning_variant: str = "standard"  # standard, p_tuning_v2, adaptive
    adaptive_prefix_length: bool = False
    max_prefix_length: int = 50
    min_prefix_length: int = 1
    
    def __post_init__(self):
        # Convert string enums to enum objects
        if isinstance(self.init_method, str):
            self.init_method = PrefixInitMethod(self.init_method)
        
        if isinstance(self.prefix_position, str):
            self.prefix_position = PrefixPosition(self.prefix_position)
        
        if isinstance(self.reparameterization_type, str):
            self.reparameterization_type = ReparameterizationType(self.reparameterization_type)
        
        # Validate settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if self.prefix_length <= 0:
            raise ValueError("prefix_length must be positive")
        
        if self.prefix_dropout < 0 or self.prefix_dropout > 1:
            raise ValueError("prefix_dropout must be between 0 and 1")
        
        if self.prefix_learning_rate <= 0:
            raise ValueError("prefix_learning_rate must be positive")
        
        if self.prefix_position == PrefixPosition.SELECTED_LAYERS and not self.selected_layers:
            raise ValueError("selected_layers must be specified when using SELECTED_LAYERS position")
        
        if self.init_method == PrefixInitMethod.FROM_TEXT and not self.init_text:
            raise ValueError("init_text must be specified when using FROM_TEXT initialization")
    
    def get_reparameterization_config(self) -> Dict[str, Any]:
        """Get reparameterization configuration"""
        if not self.reparameterization:
            return {}
        
        return {
            "type": self.reparameterization_type.value,
            "hidden_size": self.reparameterization_hidden_size,
            "num_layers": self.reparameterization_num_layers,
            "activation": self.reparameterization_activation
        }
    
    def get_prefix_dimensions(self, model_config) -> Dict[str, int]:
        """Get prefix dimensions based on model config"""
        hidden_size = self.prefix_hidden_size or model_config.hidden_size
        num_heads = self.prefix_attention_heads or model_config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        return {
            "prefix_length": self.prefix_length,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "key_value_heads": self.prefix_key_value_heads or num_heads
        }
    
    def get_trainable_modules(self) -> List[str]:
        """Get list of trainable modules"""
        trainable_modules = ["prefix_embeddings"]
        
        if self.reparameterization:
            trainable_modules.append("prefix_reparameterization")
        
        if self.prefix_projection:
            trainable_modules.append("prefix_projection")
        
        # Add any additional trainable base modules
        trainable_modules.extend(self.trainable_base_modules)
        
        return trainable_modules
    
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PrefixConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined prefix configurations
STANDARD_PREFIX_CONFIG = PrefixConfig(
    prefix_length=10,
    prefix_dropout=0.1,
    reparameterization=True,
    reparameterization_type=ReparameterizationType.MLP,
    prefix_learning_rate=1e-3
)

LIGHTWEIGHT_PREFIX_CONFIG = PrefixConfig(
    prefix_length=5,
    prefix_dropout=0.0,
    reparameterization=False,
    prefix_learning_rate=5e-4,
    prefix_position=PrefixPosition.INPUT_ONLY
)

DEEP_PREFIX_CONFIG = PrefixConfig(
    prefix_length=20,
    prefix_dropout=0.2,
    reparameterization=True,
    reparameterization_type=ReparameterizationType.TRANSFORMER,
    reparameterization_num_layers=3,
    prefix_learning_rate=2e-3,
    different_prefix_per_layer=True
)

GENERATION_PREFIX_CONFIG = PrefixConfig(
    prefix_length=15,
    prefix_dropout=0.1,
    reparameterization=True,
    reparameterization_type=ReparameterizationType.LSTM,
    task_type="generation",
    use_past_key_values=True,
    prefix_learning_rate=1e-3
)

CLASSIFICATION_PREFIX_CONFIG = PrefixConfig(
    prefix_length=8,
    prefix_dropout=0.1,
    reparameterization=True,
    reparameterization_type=ReparameterizationType.MLP,
    task_type="classification",
    prefix_learning_rate=1e-3
)

P_TUNING_V2_CONFIG = PrefixConfig(
    prefix_length=10,
    prefix_dropout=0.1,
    reparameterization=True,
    reparameterization_type=ReparameterizationType.MLP,
    prefix_position=PrefixPosition.ALL_LAYERS,
    different_prefix_per_layer=True,
    prefix_tuning_variant="p_tuning_v2",
    prefix_learning_rate=1e-3
)

ADAPTIVE_PREFIX_CONFIG = PrefixConfig(
    prefix_length=10,
    prefix_dropout=0.1,
    reparameterization=True,
    adaptive_prefix_length=True,
    max_prefix_length=30,
    min_prefix_length=3,
    prefix_tuning_variant="adaptive",
    prefix_learning_rate=1e-3
)
