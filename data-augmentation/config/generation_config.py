"""
Configuration for synthetic data generation using LLMs
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class GenerationModel(Enum):
    """Supported generation models"""
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    T5_SMALL = "t5-small"
    T5_BASE = "t5-base"
    T5_LARGE = "t5-large"
    FLAN_T5_BASE = "google/flan-t5-base"
    FLAN_T5_LARGE = "google/flan-t5-large"


class GenerationStrategy(Enum):
    """Generation strategies"""
    DIVERSE = "diverse"  # Maximize diversity
    QUALITY = "quality"  # Maximize quality
    BALANCED = "balanced"  # Balance quality and diversity
    CREATIVE = "creative"  # Maximize creativity
    FACTUAL = "factual"  # Focus on factual accuracy


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation"""
    
    # Model settings
    model_name: Union[GenerationModel, str] = GenerationModel.GPT_3_5_TURBO
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Generation parameters
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 150
    num_samples: int = 10
    
    # Generation strategy
    generation_strategy: GenerationStrategy = GenerationStrategy.BALANCED
    
    # Prompt engineering
    system_prompt: Optional[str] = None
    instruction_template: str = "{instruction}"
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    use_chain_of_thought: bool = False
    
    # Quality control
    min_length: int = 10
    max_length: int = 500
    filter_duplicates: bool = True
    filter_near_duplicates: bool = True
    similarity_threshold: float = 0.85
    
    # Diversity settings
    diversity_penalty: float = 0.0
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    
    # Batch processing
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Task-specific settings
    task_type: str = "classification"  # classification, qa, ner, summarization
    preserve_format: bool = True
    output_format: str = "text"  # text, json, structured
    
    # Language settings
    language: str = "en"
    target_languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Cost control
    max_cost_per_sample: float = 0.01  # USD
    cost_tracking: bool = True
    
    # Reproducibility
    seed: Optional[int] = None
    deterministic: bool = False
    
    def __post_init__(self):
        # Convert string enums to enum objects
        if isinstance(self.model_name, str):
            try:
                self.model_name = GenerationModel(self.model_name)
            except ValueError:
                # Keep as string if not in enum (for custom models)
                pass
        
        if isinstance(self.generation_strategy, str):
            self.generation_strategy = GenerationStrategy(self.generation_strategy)
        
        # Validate parameters
        self._validate_parameters()
        
        # Set strategy-specific defaults
        self._apply_strategy_defaults()
    
    def _validate_parameters(self):
        """Validate generation parameters"""
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
    
    def _apply_strategy_defaults(self):
        """Apply strategy-specific parameter defaults"""
        if self.generation_strategy == GenerationStrategy.DIVERSE:
            self.temperature = max(self.temperature, 1.0)
            self.top_p = max(self.top_p, 0.9)
            self.diversity_penalty = max(self.diversity_penalty, 0.5)
        
        elif self.generation_strategy == GenerationStrategy.QUALITY:
            self.temperature = min(self.temperature, 0.7)
            self.top_p = min(self.top_p, 0.8)
            self.repetition_penalty = max(self.repetition_penalty, 1.2)
        
        elif self.generation_strategy == GenerationStrategy.CREATIVE:
            self.temperature = max(self.temperature, 1.2)
            self.top_p = max(self.top_p, 0.95)
            self.top_k = max(self.top_k, 100)
        
        elif self.generation_strategy == GenerationStrategy.FACTUAL:
            self.temperature = min(self.temperature, 0.3)
            self.top_p = min(self.top_p, 0.7)
            self.repetition_penalty = max(self.repetition_penalty, 1.1)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model-specific parameters"""
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add model-specific parameters
        if isinstance(self.model_name, GenerationModel):
            model_name = self.model_name.value
        else:
            model_name = self.model_name
        
        if "gpt" in model_name.lower():
            params.update({
                "top_p": self.top_p,
                "frequency_penalty": self.diversity_penalty,
                "presence_penalty": self.repetition_penalty - 1.0,
            })
        elif "t5" in model_name.lower():
            params.update({
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "length_penalty": self.length_penalty,
            })
        elif "claude" in model_name.lower():
            params.update({
                "top_p": self.top_p,
                "top_k": self.top_k,
            })
        
        return params
    
    def get_prompt_template(self) -> str:
        """Get formatted prompt template"""
        template_parts = []
        
        if self.system_prompt:
            template_parts.append(f"System: {self.system_prompt}")
        
        if self.few_shot_examples:
            template_parts.append("Examples:")
            for i, example in enumerate(self.few_shot_examples, 1):
                if "input" in example and "output" in example:
                    template_parts.append(f"{i}. Input: {example['input']}")
                    template_parts.append(f"   Output: {example['output']}")
        
        template_parts.append(self.instruction_template)
        
        return "\n".join(template_parts)
    
    def estimate_cost(self, num_requests: int) -> float:
        """Estimate generation cost"""
        if not isinstance(self.model_name, GenerationModel):
            return 0.0  # Unknown cost for custom models
        
        # Rough cost estimates (as of 2024)
        cost_per_1k_tokens = {
            GenerationModel.GPT_3_5_TURBO: 0.002,
            GenerationModel.GPT_4: 0.03,
            GenerationModel.GPT_4_TURBO: 0.01,
            GenerationModel.CLAUDE_3_HAIKU: 0.00025,
            GenerationModel.CLAUDE_3_SONNET: 0.003,
        }
        
        if self.model_name in cost_per_1k_tokens:
            tokens_per_request = self.max_tokens + 50  # Estimate input tokens
            total_tokens = num_requests * tokens_per_request
            return (total_tokens / 1000) * cost_per_1k_tokens[self.model_name]
        
        return 0.0
    
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GenerationConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined generation configurations
FAST_GENERATION = GenerationConfig(
    model_name=GenerationModel.GPT_3_5_TURBO,
    temperature=0.7,
    max_tokens=100,
    num_samples=5,
    generation_strategy=GenerationStrategy.QUALITY
)

DIVERSE_GENERATION = GenerationConfig(
    model_name=GenerationModel.GPT_3_5_TURBO,
    temperature=1.2,
    top_p=0.95,
    max_tokens=150,
    num_samples=10,
    generation_strategy=GenerationStrategy.DIVERSE
)

HIGH_QUALITY_GENERATION = GenerationConfig(
    model_name=GenerationModel.GPT_4,
    temperature=0.5,
    top_p=0.8,
    max_tokens=200,
    num_samples=5,
    generation_strategy=GenerationStrategy.QUALITY
)

CREATIVE_GENERATION = GenerationConfig(
    model_name=GenerationModel.GPT_3_5_TURBO,
    temperature=1.5,
    top_p=0.95,
    top_k=100,
    max_tokens=200,
    num_samples=8,
    generation_strategy=GenerationStrategy.CREATIVE
)

FACTUAL_GENERATION = GenerationConfig(
    model_name=GenerationModel.GPT_4,
    temperature=0.2,
    top_p=0.7,
    max_tokens=150,
    num_samples=3,
    generation_strategy=GenerationStrategy.FACTUAL
)

T5_GENERATION = GenerationConfig(
    model_name=GenerationModel.FLAN_T5_BASE,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    max_tokens=128,
    num_samples=5,
    generation_strategy=GenerationStrategy.BALANCED
)
