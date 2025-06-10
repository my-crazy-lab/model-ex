"""
Configuration for model quantization
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class QuantizationType(Enum):
    """Types of quantization"""
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"
    DYNAMIC = "dynamic"
    STATIC = "static"


class QuantizationBackend(Enum):
    """Quantization backends"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    BITSANDBYTES = "bitsandbytes"
    NEURAL_COMPRESSOR = "neural_compressor"


class CalibrationMethod(Enum):
    """Calibration methods for quantization"""
    MINMAX = "minmax"
    ENTROPY = "entropy"
    PERCENTILE = "percentile"
    MSE = "mse"
    KL_DIVERGENCE = "kl_divergence"


class QuantizationScheme(Enum):
    """Quantization schemes"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class QuantizationGranularity(Enum):
    """Quantization granularity"""
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    
    # Core quantization settings
    quantization_type: QuantizationType = QuantizationType.INT8
    backend: QuantizationBackend = QuantizationBackend.PYTORCH
    
    # Quantization parameters
    bits: int = 8
    scheme: QuantizationScheme = QuantizationScheme.ASYMMETRIC
    granularity: QuantizationGranularity = QuantizationGranularity.PER_TENSOR
    
    # Calibration settings
    calibration_method: CalibrationMethod = CalibrationMethod.MINMAX
    calibration_dataset: Optional[str] = None
    num_calibration_samples: int = 100
    calibration_batch_size: int = 32
    
    # Layer-specific settings
    quantize_embeddings: bool = True
    quantize_attention: bool = True
    quantize_feed_forward: bool = True
    quantize_output: bool = True
    
    # Layers to skip quantization
    skip_layers: Optional[List[str]] = None
    
    # Advanced settings
    use_fake_quantization: bool = False  # For QAT
    symmetric_weights: bool = True
    symmetric_activations: bool = False
    
    # Optimization settings
    optimize_for_inference: bool = True
    fuse_operations: bool = True
    
    # Quality preservation
    preserve_accuracy: bool = True
    accuracy_threshold: float = 0.01  # Max acceptable accuracy drop
    
    # BitsAndBytes specific settings
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
    
    # Neural Compressor settings
    nc_approach: str = "post_training_static_quant"
    nc_accuracy_criterion: float = 0.01
    nc_timeout: int = 0
    nc_max_trials: int = 100
    
    # Output settings
    output_dir: str = "./quantized_model"
    save_original_model: bool = True
    
    def __post_init__(self):
        # Convert string enums to enum objects
        if isinstance(self.quantization_type, str):
            self.quantization_type = QuantizationType(self.quantization_type)
        
        if isinstance(self.backend, str):
            self.backend = QuantizationBackend(self.backend)
        
        if isinstance(self.calibration_method, str):
            self.calibration_method = CalibrationMethod(self.calibration_method)
        
        if isinstance(self.scheme, str):
            self.scheme = QuantizationScheme(self.scheme)
        
        if isinstance(self.granularity, str):
            self.granularity = QuantizationGranularity(self.granularity)
        
        # Validate settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if self.bits not in [4, 8, 16]:
            raise ValueError("Bits must be 4, 8, or 16")
        
        if self.num_calibration_samples <= 0:
            raise ValueError("Number of calibration samples must be positive")
        
        if not 0 <= self.accuracy_threshold <= 1:
            raise ValueError("Accuracy threshold must be between 0 and 1")
        
        # Backend-specific validations
        if self.backend == QuantizationBackend.BITSANDBYTES:
            if not (self.load_in_8bit or self.load_in_4bit):
                self.load_in_8bit = True  # Default to 8-bit
        
        if self.quantization_type == QuantizationType.INT4 and self.bits != 4:
            self.bits = 4
        
        if self.quantization_type == QuantizationType.INT8 and self.bits != 8:
            self.bits = 8
    
    def get_pytorch_qconfig(self):
        """Get PyTorch quantization configuration"""
        import torch
        
        if self.quantization_type == QuantizationType.DYNAMIC:
            return torch.quantization.default_dynamic_qconfig
        elif self.quantization_type == QuantizationType.STATIC:
            return torch.quantization.get_default_qconfig('fbgemm')
        else:
            # Custom qconfig
            from torch.quantization import QConfig
            from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
            
            if self.calibration_method == CalibrationMethod.MINMAX:
                activation_observer = MinMaxObserver
                weight_observer = MinMaxObserver
            else:
                activation_observer = MovingAverageMinMaxObserver
                weight_observer = MovingAverageMinMaxObserver
            
            return QConfig(
                activation=activation_observer.with_args(
                    dtype=torch.quint8 if self.bits == 8 else torch.qint8,
                    qscheme=torch.per_tensor_affine if self.granularity == QuantizationGranularity.PER_TENSOR else torch.per_channel_affine
                ),
                weight=weight_observer.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric if self.scheme == QuantizationScheme.SYMMETRIC else torch.per_tensor_affine
                )
            )
    
    def get_bitsandbytes_config(self):
        """Get BitsAndBytes quantization configuration"""
        try:
            from transformers import BitsAndBytesConfig
            
            if self.load_in_4bit:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=self.bnb_4bit_quant_type
                )
            elif self.load_in_8bit:
                return BitsAndBytesConfig(
                    load_in_8bit=True
                )
            else:
                return None
        except ImportError:
            raise ImportError("BitsAndBytes not available. Install with: pip install bitsandbytes")
    
    def get_neural_compressor_config(self):
        """Get Neural Compressor configuration"""
        try:
            from neural_compressor import PostTrainingQuantConfig
            
            return PostTrainingQuantConfig(
                approach=self.nc_approach,
                backend=self.backend.value,
                accuracy_criterion={
                    'relative': self.nc_accuracy_criterion,
                    'higher_is_better': True
                },
                timeout=self.nc_timeout,
                max_trials=self.nc_max_trials
            )
        except ImportError:
            raise ImportError("Neural Compressor not available. Install with: pip install neural-compressor")
    
    def get_compression_ratio(self) -> float:
        """Get theoretical compression ratio"""
        if self.quantization_type == QuantizationType.INT8:
            return 4.0  # FP32 → INT8
        elif self.quantization_type == QuantizationType.INT4:
            return 8.0  # FP32 → INT4
        elif self.quantization_type == QuantizationType.FP16:
            return 2.0  # FP32 → FP16
        elif self.quantization_type == QuantizationType.BF16:
            return 2.0  # FP32 → BF16
        else:
            return 1.0  # No compression
    
    def get_expected_speedup(self) -> float:
        """Get expected inference speedup"""
        speedup_map = {
            QuantizationType.INT8: 2.5,
            QuantizationType.INT4: 4.0,
            QuantizationType.FP16: 1.8,
            QuantizationType.BF16: 1.8,
            QuantizationType.DYNAMIC: 2.0,
            QuantizationType.STATIC: 3.0
        }
        return speedup_map.get(self.quantization_type, 1.0)
    
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined quantization configurations
INT8_QUANTIZATION_CONFIG = QuantizationConfig(
    quantization_type=QuantizationType.INT8,
    backend=QuantizationBackend.PYTORCH,
    calibration_method=CalibrationMethod.MINMAX,
    num_calibration_samples=100
)

INT4_QUANTIZATION_CONFIG = QuantizationConfig(
    quantization_type=QuantizationType.INT4,
    backend=QuantizationBackend.BITSANDBYTES,
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

DYNAMIC_QUANTIZATION_CONFIG = QuantizationConfig(
    quantization_type=QuantizationType.DYNAMIC,
    backend=QuantizationBackend.PYTORCH,
    optimize_for_inference=True,
    fuse_operations=True
)

FP16_MIXED_PRECISION_CONFIG = QuantizationConfig(
    quantization_type=QuantizationType.FP16,
    backend=QuantizationBackend.PYTORCH,
    optimize_for_inference=True
)

NEURAL_COMPRESSOR_CONFIG = QuantizationConfig(
    quantization_type=QuantizationType.INT8,
    backend=QuantizationBackend.NEURAL_COMPRESSOR,
    nc_approach="post_training_static_quant",
    nc_accuracy_criterion=0.01,
    nc_max_trials=100
)

MOBILE_OPTIMIZED_CONFIG = QuantizationConfig(
    quantization_type=QuantizationType.INT8,
    backend=QuantizationBackend.PYTORCH,
    optimize_for_inference=True,
    fuse_operations=True,
    preserve_accuracy=True,
    accuracy_threshold=0.02
)

EDGE_DEPLOYMENT_CONFIG = QuantizationConfig(
    quantization_type=QuantizationType.INT4,
    backend=QuantizationBackend.BITSANDBYTES,
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    optimize_for_inference=True,
    preserve_accuracy=False  # Prioritize size over accuracy
)
