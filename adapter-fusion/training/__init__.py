"""
Training module for Adapter Fusion
"""

from .fusion_trainer import FusionTrainer
from .adapter_trainer import AdapterTrainer
from .multi_task_trainer import MultiTaskTrainer

__all__ = ["FusionTrainer", "AdapterTrainer", "MultiTaskTrainer"]
