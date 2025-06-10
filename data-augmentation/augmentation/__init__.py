"""
Augmentation module for text data augmentation
"""

from .text_augmentation import TextAugmenter, SynonymReplacer, RandomOperations
from .llm_generation import LLMGenerator
from .back_translation import BackTranslator
from .template_generation import TemplateGenerator

__all__ = [
    "TextAugmenter",
    "SynonymReplacer", 
    "RandomOperations",
    "LLMGenerator",
    "BackTranslator",
    "TemplateGenerator"
]
