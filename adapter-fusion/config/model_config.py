"""
Model configuration for Adapter Fusion implementation
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
    
    # Multi-task settings
    multi_task: bool = False
    task_names: List[str] = field(default_factory=list)
    task_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
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
        
        # Validate multi-task configuration
        if self.multi_task and not self.task_names:
            raise ValueError("task_names must be provided when multi_task=True")
    
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
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Get configuration for specific task"""
        if task_name in self.task_configs:
            return self.task_configs[task_name]
        
        # Return default config
        return {
            "task_type": self.task_type,
            "num_labels": self.num_labels,
            "problem_type": self.problem_type
        }
    
    def add_task(self, task_name: str, task_config: Dict[str, Any]):
        """Add a new task configuration"""
        if not self.multi_task:
            self.multi_task = True
        
        if task_name not in self.task_names:
            self.task_names.append(task_name)
        
        self.task_configs[task_name] = task_config


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

# Multi-task configurations
GLUE_MULTI_TASK_CONFIG = ModelConfig(
    model_name_or_path="bert-base-uncased",
    max_length=512,
    multi_task=True,
    task_names=["sst2", "cola", "mrpc", "qqp"],
    task_configs={
        "sst2": {"task_type": "classification", "num_labels": 2},
        "cola": {"task_type": "classification", "num_labels": 2},
        "mrpc": {"task_type": "classification", "num_labels": 2},
        "qqp": {"task_type": "classification", "num_labels": 2},
    }
)

NLP_MULTI_TASK_CONFIG = ModelConfig(
    model_name_or_path="bert-base-uncased",
    max_length=512,
    multi_task=True,
    task_names=["sentiment", "nli", "ner", "qa"],
    task_configs={
        "sentiment": {"task_type": "classification", "num_labels": 2},
        "nli": {"task_type": "classification", "num_labels": 3},
        "ner": {"task_type": "token_classification", "num_labels": 9},
        "qa": {"task_type": "question_answering"},
    }
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
    task_type="token_classification",
    problem_type="single_label_classification"
)

QA_CONFIG = ModelConfig(
    model_name_or_path="bert-base-uncased",
    max_length=384,
    task_type="question_answering"
)

NLI_CONFIG = ModelConfig(
    model_name_or_path="bert-base-uncased",
    num_labels=3,  # entailment, contradiction, neutral
    task_type="classification",
    problem_type="single_label_classification"
)
