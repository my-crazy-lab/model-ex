"""
Configuration for lifelong learning techniques
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class LifelongTechnique(Enum):
    """Lifelong learning techniques"""
    EWC = "ewc"  # Elastic Weight Consolidation
    REHEARSAL = "rehearsal"  # Experience Replay
    L2_REGULARIZATION = "l2_reg"  # L2 Regularization
    SYNAPTIC_INTELLIGENCE = "si"  # Synaptic Intelligence
    PROGRESSIVE = "progressive"  # Progressive Neural Networks
    PACKNET = "packnet"  # PackNet
    META_LEARNING = "meta"  # Meta-learning
    COMBINED = "combined"  # Multiple techniques


class MemoryStrategy(Enum):
    """Memory buffer strategies"""
    RANDOM = "random"
    BALANCED = "balanced"
    UNCERTAINTY = "uncertainty"
    GRADIENT_BASED = "gradient"
    HERDING = "herding"
    FIFO = "fifo"  # First In First Out
    RESERVOIR = "reservoir"  # Reservoir sampling


class EvaluationStrategy(Enum):
    """Evaluation strategies"""
    AFTER_EACH_TASK = "after_each"
    FINAL_ONLY = "final_only"
    PERIODIC = "periodic"
    CONTINUOUS = "continuous"


@dataclass
class LifelongConfig:
    """Configuration for lifelong learning"""
    
    # Core technique selection
    technique: LifelongTechnique = LifelongTechnique.EWC
    combine_techniques: bool = False
    combined_techniques: List[LifelongTechnique] = field(default_factory=list)
    
    # EWC (Elastic Weight Consolidation) settings
    ewc_lambda: float = 1000.0
    ewc_gamma: float = 1.0  # Decay factor for multi-task EWC
    fisher_estimation_samples: int = 1000
    fisher_estimation_method: str = "diagonal"  # diagonal, full, kfac
    online_ewc: bool = False
    
    # Experience Replay settings
    memory_size: int = 1000
    memory_strategy: MemoryStrategy = MemoryStrategy.RANDOM
    replay_batch_size: int = 32
    replay_frequency: int = 1  # How often to replay
    balanced_replay: bool = True
    
    # L2 Regularization settings
    l2_lambda: float = 0.01
    selective_l2: bool = True  # Only on important weights
    
    # Synaptic Intelligence settings
    si_c: float = 0.1  # SI regularization strength
    si_xi: float = 1.0  # SI damping parameter
    
    # Progressive Networks settings
    progressive_columns: int = 4  # Max columns
    lateral_connections: bool = True
    column_capacity: int = 512  # Hidden units per column
    
    # PackNet settings
    packnet_pruning_ratio: float = 0.8
    packnet_retrain_epochs: int = 5
    
    # Meta-learning settings
    meta_inner_lr: float = 0.01
    meta_outer_lr: float = 0.001
    meta_adaptation_steps: int = 5
    meta_batch_size: int = 16
    
    # General training settings
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs_per_task: int = 10
    early_stopping_patience: int = 3
    
    # Evaluation settings
    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.AFTER_EACH_TASK
    evaluation_frequency: int = 100  # Steps
    compute_forgetting_metrics: bool = True
    compute_transfer_metrics: bool = True
    
    # Memory and efficiency
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    max_memory_gb: float = 8.0
    
    # Task management
    max_tasks: int = 10
    task_incremental: bool = True  # vs class incremental
    domain_incremental: bool = False
    
    # Regularization and stability
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    dropout: float = 0.1
    
    # Monitoring and logging
    log_frequency: int = 100
    save_checkpoints: bool = True
    checkpoint_frequency: int = 1000
    track_weight_changes: bool = True
    track_gradient_norms: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        # Convert string enums to enum objects
        if isinstance(self.technique, str):
            self.technique = LifelongTechnique(self.technique)
        
        if isinstance(self.memory_strategy, str):
            self.memory_strategy = MemoryStrategy(self.memory_strategy)
        
        if isinstance(self.evaluation_strategy, str):
            self.evaluation_strategy = EvaluationStrategy(self.evaluation_strategy)
        
        # Convert combined techniques
        if isinstance(self.combined_techniques, list) and self.combined_techniques:
            if isinstance(self.combined_techniques[0], str):
                self.combined_techniques = [
                    LifelongTechnique(tech) for tech in self.combined_techniques
                ]
        
        # Validate settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if self.ewc_lambda < 0:
            raise ValueError("ewc_lambda must be non-negative")
        
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.epochs_per_task <= 0:
            raise ValueError("epochs_per_task must be positive")
        
        if self.combine_techniques and not self.combined_techniques:
            raise ValueError("combined_techniques must be specified when combine_techniques=True")
    
    def get_technique_config(self) -> Dict[str, Any]:
        """Get configuration for the selected technique"""
        if self.technique == LifelongTechnique.EWC:
            return {
                "lambda": self.ewc_lambda,
                "gamma": self.ewc_gamma,
                "fisher_samples": self.fisher_estimation_samples,
                "fisher_method": self.fisher_estimation_method,
                "online": self.online_ewc
            }
        
        elif self.technique == LifelongTechnique.REHEARSAL:
            return {
                "memory_size": self.memory_size,
                "strategy": self.memory_strategy.value,
                "batch_size": self.replay_batch_size,
                "frequency": self.replay_frequency,
                "balanced": self.balanced_replay
            }
        
        elif self.technique == LifelongTechnique.L2_REGULARIZATION:
            return {
                "lambda": self.l2_lambda,
                "selective": self.selective_l2
            }
        
        elif self.technique == LifelongTechnique.SYNAPTIC_INTELLIGENCE:
            return {
                "c": self.si_c,
                "xi": self.si_xi
            }
        
        elif self.technique == LifelongTechnique.PROGRESSIVE:
            return {
                "columns": self.progressive_columns,
                "lateral_connections": self.lateral_connections,
                "column_capacity": self.column_capacity
            }
        
        elif self.technique == LifelongTechnique.PACKNET:
            return {
                "pruning_ratio": self.packnet_pruning_ratio,
                "retrain_epochs": self.packnet_retrain_epochs
            }
        
        elif self.technique == LifelongTechnique.META_LEARNING:
            return {
                "inner_lr": self.meta_inner_lr,
                "outer_lr": self.meta_outer_lr,
                "adaptation_steps": self.meta_adaptation_steps,
                "batch_size": self.meta_batch_size
            }
        
        return {}
    
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LifelongConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined lifelong learning configurations
EWC_CONFIG = LifelongConfig(
    technique=LifelongTechnique.EWC,
    ewc_lambda=1000.0,
    fisher_estimation_samples=1000,
    learning_rate=1e-4,
    epochs_per_task=10
)

REHEARSAL_CONFIG = LifelongConfig(
    technique=LifelongTechnique.REHEARSAL,
    memory_size=1000,
    memory_strategy=MemoryStrategy.BALANCED,
    replay_batch_size=32,
    learning_rate=1e-4,
    epochs_per_task=10
)

PROGRESSIVE_CONFIG = LifelongConfig(
    technique=LifelongTechnique.PROGRESSIVE,
    progressive_columns=4,
    lateral_connections=True,
    column_capacity=512,
    learning_rate=1e-4,
    epochs_per_task=10
)

COMBINED_CONFIG = LifelongConfig(
    technique=LifelongTechnique.COMBINED,
    combine_techniques=True,
    combined_techniques=[LifelongTechnique.EWC, LifelongTechnique.REHEARSAL],
    ewc_lambda=500.0,  # Reduced since combined
    memory_size=500,   # Reduced since combined
    learning_rate=1e-4,
    epochs_per_task=10
)

META_LEARNING_CONFIG = LifelongConfig(
    technique=LifelongTechnique.META_LEARNING,
    meta_inner_lr=0.01,
    meta_outer_lr=0.001,
    meta_adaptation_steps=5,
    learning_rate=1e-4,
    epochs_per_task=5  # Fewer epochs due to meta-learning
)

LIGHTWEIGHT_CONFIG = LifelongConfig(
    technique=LifelongTechnique.L2_REGULARIZATION,
    l2_lambda=0.01,
    selective_l2=True,
    learning_rate=1e-4,
    epochs_per_task=5,
    memory_size=100,  # Small memory footprint
    batch_size=16
)
