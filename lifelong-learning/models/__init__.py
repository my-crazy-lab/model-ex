"""
Models module for lifelong learning
"""

from .lifelong_model import LifelongModel
from .task_heads import TaskSpecificHeads, MultiTaskHead
from .shared_backbone import SharedBackbone

__all__ = ["LifelongModel", "TaskSpecificHeads", "MultiTaskHead", "SharedBackbone"]
