"""
Configuration for RLHF training
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class RLHFAlgorithm(Enum):
    """RLHF algorithms"""
    PPO = "ppo"
    DPO = "dpo"
    RLAIF = "rlaif"
    REWARD_MIXING = "reward_mixing"


class RewardModelType(Enum):
    """Types of reward models"""
    PREFERENCE = "preference"
    CONSTITUTIONAL = "constitutional"
    MULTI_OBJECTIVE = "multi_objective"
    ENSEMBLE = "ensemble"


class PolicyUpdateMethod(Enum):
    """Policy update methods"""
    CLIPPED = "clipped"
    ADAPTIVE_KL = "adaptive_kl"
    NATURAL = "natural"
    TRUST_REGION = "trust_region"


class FeedbackSource(Enum):
    """Sources of feedback"""
    HUMAN = "human"
    AI = "ai"
    MIXED = "mixed"
    CONSTITUTIONAL = "constitutional"


@dataclass
class RLHFConfig:
    """Configuration for RLHF training"""
    
    # Model configuration
    model_name_or_path: str = "gpt2"
    tokenizer_name_or_path: Optional[str] = None
    
    # RLHF algorithm
    algorithm: RLHFAlgorithm = RLHFAlgorithm.PPO
    reward_model_type: RewardModelType = RewardModelType.PREFERENCE
    
    # Reward model configuration
    reward_model_path: Optional[str] = None
    reward_model_name: Optional[str] = None
    use_multiple_rewards: bool = False
    reward_weights: Optional[Dict[str, float]] = None
    
    # Training configuration
    learning_rate: float = 1e-5
    batch_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    
    # PPO specific parameters
    ppo_epochs: int = 4
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    vf_coef: float = 0.1
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # KL penalty
    kl_penalty: str = "kl"  # "kl" or "abs"
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    adaptive_kl: bool = True
    
    # DPO specific parameters
    dpo_beta: float = 0.1
    dpo_label_smoothing: float = 0.0
    dpo_loss_type: str = "sigmoid"  # "sigmoid" or "hinge"
    
    # Generation parameters
    max_length: int = 512
    min_length: int = 10
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    
    # Training parameters
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Memory optimization
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    use_qlora: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    
    # Distributed training
    deepspeed: Optional[str] = None
    local_rank: int = -1
    
    # Data configuration
    max_prompt_length: int = 256
    max_response_length: int = 256
    
    # Evaluation configuration
    eval_batch_size: int = 16
    eval_accumulation_steps: int = 1
    
    # Safety and filtering
    use_safety_filter: bool = True
    safety_threshold: float = 0.5
    content_filter: bool = True
    
    # Feedback collection
    feedback_source: FeedbackSource = FeedbackSource.HUMAN
    feedback_collection_rate: float = 0.1
    active_learning: bool = False
    
    # Constitutional AI
    use_constitutional_ai: bool = False
    constitution_path: Optional[str] = None
    self_critique_enabled: bool = False
    
    # Output configuration
    output_dir: str = "./rlhf_output"
    run_name: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        # Convert string enums to enum objects
        if isinstance(self.algorithm, str):
            self.algorithm = RLHFAlgorithm(self.algorithm)
        
        if isinstance(self.reward_model_type, str):
            self.reward_model_type = RewardModelType(self.reward_model_type)
        
        if isinstance(self.feedback_source, str):
            self.feedback_source = FeedbackSource(self.feedback_source)
        
        # Set default tokenizer path
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
        
        # Set default LoRA target modules based on model
        if self.use_lora and self.lora_target_modules is None:
            if "gpt" in self.model_name_or_path.lower():
                self.lora_target_modules = ["c_attn", "c_proj", "c_fc"]
            elif "llama" in self.model_name_or_path.lower():
                self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            elif "t5" in self.model_name_or_path.lower():
                self.lora_target_modules = ["q", "v", "k", "o", "wi", "wo"]
            else:
                self.lora_target_modules = ["query", "value", "key", "dense"]
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.mini_batch_size > self.batch_size:
            raise ValueError("Mini batch size cannot be larger than batch size")
        
        if not 0 <= self.clip_range <= 1:
            raise ValueError("Clip range must be between 0 and 1")
        
        if self.dpo_beta <= 0:
            raise ValueError("DPO beta must be positive")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature should be between 0 and 2")
        
        if self.use_lora and self.use_qlora:
            raise ValueError("Cannot use both LoRA and QLoRA simultaneously")
        
        if self.reward_weights and not self.use_multiple_rewards:
            raise ValueError("Reward weights specified but use_multiple_rewards is False")
    
    def get_ppo_config(self):
        """Get PPO-specific configuration"""
        return {
            "learning_rate": self.learning_rate,
            "mini_batch_size": self.mini_batch_size,
            "batch_size": self.batch_size,
            "ppo_epochs": self.ppo_epochs,
            "gamma": 1.0,
            "lam": 0.95,
            "cliprange": self.clip_range,
            "cliprange_value": self.clip_range_vf,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "init_kl_coef": self.init_kl_coef,
            "adap_kl_ctrl": self.adaptive_kl,
        }
    
    def get_generation_config(self):
        """Get generation configuration"""
        return {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "pad_token_id": None,  # Will be set based on tokenizer
            "eos_token_id": None,  # Will be set based on tokenizer
        }
    
    def get_lora_config(self):
        """Get LoRA configuration"""
        if not self.use_lora:
            return None
        
        from peft import LoraConfig, TaskType
        
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    
    def get_qlora_config(self):
        """Get QLoRA configuration"""
        if not self.use_qlora:
            return None
        
        from transformers import BitsAndBytesConfig
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    def get_reward_weights(self) -> Dict[str, float]:
        """Get reward weights for multi-objective training"""
        if self.reward_weights:
            return self.reward_weights
        
        # Default weights for common reward types
        default_weights = {
            "helpfulness": 0.4,
            "safety": 0.3,
            "factuality": 0.2,
            "engagement": 0.1
        }
        
        return default_weights
    
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RLHFConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined RLHF configurations
CHATBOT_RLHF_CONFIG = RLHFConfig(
    model_name_or_path="microsoft/DialoGPT-medium",
    algorithm=RLHFAlgorithm.PPO,
    learning_rate=1e-5,
    batch_size=8,
    ppo_epochs=4,
    max_length=512,
    use_safety_filter=True,
    feedback_source=FeedbackSource.HUMAN
)

SUMMARIZATION_RLHF_CONFIG = RLHFConfig(
    model_name_or_path="facebook/bart-large-cnn",
    algorithm=RLHFAlgorithm.DPO,
    dpo_beta=0.1,
    learning_rate=5e-6,
    batch_size=4,
    max_length=256,
    use_multiple_rewards=True,
    reward_weights={"factuality": 0.5, "conciseness": 0.3, "readability": 0.2}
)

CODE_GENERATION_RLHF_CONFIG = RLHFConfig(
    model_name_or_path="Salesforce/codegen-350M-mono",
    algorithm=RLHFAlgorithm.PPO,
    learning_rate=2e-5,
    batch_size=16,
    ppo_epochs=2,
    max_length=1024,
    use_constitutional_ai=True,
    feedback_source=FeedbackSource.AI
)

EFFICIENT_RLHF_CONFIG = RLHFConfig(
    model_name_or_path="gpt2",
    algorithm=RLHFAlgorithm.DPO,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    learning_rate=1e-4,
    batch_size=4,
    fp16=True
)

CONSTITUTIONAL_AI_CONFIG = RLHFConfig(
    model_name_or_path="gpt2",
    algorithm=RLHFAlgorithm.PPO,
    use_constitutional_ai=True,
    self_critique_enabled=True,
    reward_model_type=RewardModelType.CONSTITUTIONAL,
    feedback_source=FeedbackSource.CONSTITUTIONAL,
    learning_rate=1e-5,
    batch_size=8
)

MULTI_OBJECTIVE_CONFIG = RLHFConfig(
    model_name_or_path="gpt2",
    algorithm=RLHFAlgorithm.REWARD_MIXING,
    use_multiple_rewards=True,
    reward_model_type=RewardModelType.MULTI_OBJECTIVE,
    reward_weights={
        "helpfulness": 0.35,
        "safety": 0.35,
        "factuality": 0.20,
        "engagement": 0.10
    },
    learning_rate=1e-5,
    batch_size=8
)
