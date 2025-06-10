"""
Lifelong learning techniques module
"""

from .ewc import ElasticWeightConsolidation, FisherInformationMatrix
from .rehearsal import ExperienceReplay, MemoryBuffer
from .regularization import L2Regularization, SynapticIntelligence
from .progressive import ProgressiveNeuralNetworks
from .meta_learning import MetaLearner

__all__ = [
    "ElasticWeightConsolidation",
    "FisherInformationMatrix", 
    "ExperienceReplay",
    "MemoryBuffer",
    "L2Regularization",
    "SynapticIntelligence",
    "ProgressiveNeuralNetworks",
    "MetaLearner"
]
