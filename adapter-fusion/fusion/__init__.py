"""
Fusion module for Adapter Fusion implementation
"""

from .fusion_layer import (
    FusionLayer,
    AttentionFusion,
    WeightedFusion,
    GatingFusion,
    HierarchicalFusion
)
from .adapter_manager import AdapterManager
from .fusion_model import FusionModel

__all__ = [
    "FusionLayer",
    "AttentionFusion",
    "WeightedFusion", 
    "GatingFusion",
    "HierarchicalFusion",
    "AdapterManager",
    "FusionModel"
]
