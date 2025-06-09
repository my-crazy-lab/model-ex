"""
Model configuration for LoRA/PEFT implementation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from peft import TaskType


@dataclass
class ModelConfig:
    """Configuration for base model"""
    
    # Model identification
    model_name_or_path: str = "bert-base-uncased"
    tokenizer_name_or_path: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # Model parameters
    num_labels: int = 2
    max_length: int = 512
    
    # Quantization settings
    use_quantization: bool = False
    quantization_bits: int = 4  # 4 or 8
    
    # Device settings
    device_map: Optional[str] = "auto"
    torch_dtype: str = "auto"
    
    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path


@dataclass
class PEFTConfig:
    """Configuration for PEFT methods"""
    
    # PEFT method selection
    peft_type: str = "LORA"  # LORA, PREFIX_TUNING, PROMPT_TUNING, IA3
    task_type: TaskType = TaskType.SEQ_CLS
    
    # LoRA specific parameters
    r: int = 16  # Rank
    lora_alpha: int = 32  # LoRA scaling parameter
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"  # none, all, lora_only
    
    # Prefix Tuning parameters
    num_virtual_tokens: int = 20
    prefix_projection: bool = False
    
    # Prompt Tuning parameters
    num_transformer_submodules: Optional[int] = None
    
    # IA3 parameters
    feedforward_modules: Optional[List[str]] = None
    
    # General parameters
    inference_mode: bool = False
    
    def get_target_modules_for_model(self, model_name: str) -> List[str]:
        """Get default target modules for different model types"""
        model_name = model_name.lower()
        
        if "bert" in model_name:
            return ["query", "value"]
        elif "llama" in model_name:
            return ["q_proj", "v_proj"]
        elif "mistral" in model_name:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "falcon" in model_name:
            return ["query_key_value"]
        elif "t5" in model_name:
            return ["q", "v"]
        elif "gpt" in model_name:
            return ["c_attn"]
        else:
            # Default for transformer models
            return ["query", "value"]
    
    def __post_init__(self):
        if self.target_modules is None and self.peft_type == "LORA":
            # Will be set based on model type during initialization
            pass


# Predefined configurations for common use cases
CLASSIFICATION_CONFIG = PEFTConfig(
    peft_type="LORA",
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)

GENERATION_CONFIG = PEFTConfig(
    peft_type="LORA",
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)

QA_CONFIG = PEFTConfig(
    peft_type="LORA",
    task_type=TaskType.QUESTION_ANS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)

PREFIX_TUNING_CONFIG = PEFTConfig(
    peft_type="PREFIX_TUNING",
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=20,
    prefix_projection=False,
)

PROMPT_TUNING_CONFIG = PEFTConfig(
    peft_type="PROMPT_TUNING",
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=20,
)
