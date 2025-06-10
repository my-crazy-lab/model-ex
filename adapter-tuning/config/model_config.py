"""
Model configuration for Adapter Tuning implementation
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


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
    
    # Task configuration
    task_type: str = "classification"  # classification, ner, qa, generation
    problem_type: Optional[str] = None  # single_label_classification, multi_label_classification, regression
    
    # Model loading options
    torch_dtype: str = "auto"
    device_map: Optional[str] = "auto"
    trust_remote_code: bool = False
    
    # Additional model arguments
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
        
        # Set default problem type based on task type
        if self.problem_type is None:
            if self.task_type == "classification":
                self.problem_type = "single_label_classification"
            elif self.task_type == "regression":
                self.problem_type = "regression"
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model loading arguments"""
        kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            **self.model_kwargs
        }
        
        if self.torch_dtype != "auto":
            import torch
            kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
        
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        
        return kwargs


# Predefined model configurations for common use cases
BERT_BASE_CONFIG = ModelConfig(
    model_name_or_path="bert-base-uncased",
    max_length=512,
    task_type="classification"
)

BERT_LARGE_CONFIG = ModelConfig(
    model_name_or_path="bert-large-uncased",
    max_length=512,
    task_type="classification"
)

ROBERTA_BASE_CONFIG = ModelConfig(
    model_name_or_path="roberta-base",
    max_length=512,
    task_type="classification"
)

DISTILBERT_CONFIG = ModelConfig(
    model_name_or_path="distilbert-base-uncased",
    max_length=512,
    task_type="classification"
)

GPT2_CONFIG = ModelConfig(
    model_name_or_path="gpt2",
    max_length=1024,
    task_type="generation"
)

T5_BASE_CONFIG = ModelConfig(
    model_name_or_path="t5-base",
    max_length=512,
    task_type="generation"
)

# Task-specific configurations
SENTIMENT_ANALYSIS_CONFIG = ModelConfig(
    model_name_or_path="bert-base-uncased",
    num_labels=2,
    task_type="classification",
    problem_type="single_label_classification"
)

NER_CONFIG = ModelConfig(
    model_name_or_path="bert-base-uncased",
    num_labels=9,  # Common NER tag count
    task_type="ner",
    problem_type="single_label_classification"
)

QA_CONFIG = ModelConfig(
    model_name_or_path="bert-base-uncased",
    max_length=384,
    task_type="qa"
)

MULTI_LABEL_CONFIG = ModelConfig(
    model_name_or_path="bert-base-uncased",
    num_labels=10,
    task_type="classification",
    problem_type="multi_label_classification"
)
