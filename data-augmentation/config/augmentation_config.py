"""
Configuration for text augmentation techniques
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class AugmentationType(Enum):
    """Types of augmentation techniques"""
    SYNONYM_REPLACEMENT = "synonym_replacement"
    RANDOM_INSERTION = "random_insertion"
    RANDOM_DELETION = "random_deletion"
    RANDOM_SWAP = "random_swap"
    BACK_TRANSLATION = "back_translation"
    PARAPHRASE = "paraphrase"
    CONTEXTUAL_WORD_EMBEDDING = "contextual_word_embedding"


@dataclass
class AugmentationConfig:
    """Configuration for text augmentation"""
    
    # Basic augmentation settings
    augmentation_types: List[AugmentationType] = field(
        default_factory=lambda: [
            AugmentationType.SYNONYM_REPLACEMENT,
            AugmentationType.RANDOM_INSERTION,
            AugmentationType.RANDOM_DELETION,
            AugmentationType.RANDOM_SWAP
        ]
    )
    
    # Augmentation probabilities
    synonym_replacement_prob: float = 0.1
    random_insertion_prob: float = 0.1
    random_deletion_prob: float = 0.1
    random_swap_prob: float = 0.1
    
    # Number of augmentations per sample
    num_augmentations_per_sample: int = 1
    max_augmentations_per_sample: int = 5
    
    # Synonym replacement settings
    synonym_replacement_ratio: float = 0.1  # Fraction of words to replace
    use_wordnet: bool = True
    use_word2vec: bool = False
    word2vec_model_path: Optional[str] = None
    
    # Random insertion settings
    random_insertion_ratio: float = 0.1  # Fraction of words to insert
    insertion_word_types: List[str] = field(
        default_factory=lambda: ["noun", "verb", "adjective", "adverb"]
    )
    
    # Random deletion settings
    random_deletion_ratio: float = 0.1  # Fraction of words to delete
    min_words_after_deletion: int = 1
    
    # Random swap settings
    random_swap_ratio: float = 0.1  # Fraction of word pairs to swap
    
    # Back translation settings
    back_translation_languages: List[str] = field(
        default_factory=lambda: ["fr", "de", "es", "it"]
    )
    translation_service: str = "google"  # google, deepl, azure
    
    # Contextual word embedding settings
    contextual_model_name: str = "bert-base-uncased"
    contextual_top_k: int = 5
    contextual_temperature: float = 1.0
    
    # Paraphrase settings
    paraphrase_model_name: str = "tuner007/pegasus_paraphrase"
    paraphrase_num_beams: int = 10
    paraphrase_num_return_sequences: int = 3
    paraphrase_temperature: float = 1.5
    
    # Quality control
    preserve_labels: bool = True
    preserve_named_entities: bool = True
    preserve_special_tokens: bool = True
    min_text_length: int = 5
    max_text_length: int = 512
    
    # Language settings
    language: str = "en"
    preserve_case: bool = True
    preserve_punctuation: bool = True
    
    # Advanced settings
    use_pos_tagging: bool = True
    pos_model: str = "en_core_web_sm"  # spaCy model
    exclude_pos_tags: List[str] = field(
        default_factory=lambda: ["PROPN", "NUM"]  # Proper nouns, numbers
    )
    
    # Filtering settings
    similarity_threshold: float = 0.9  # Avoid too similar augmentations
    fluency_threshold: float = 0.7  # Minimum fluency score
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        # Convert string enums to enum objects
        if isinstance(self.augmentation_types[0], str):
            self.augmentation_types = [
                AugmentationType(aug_type) for aug_type in self.augmentation_types
            ]
        
        # Validate probabilities
        self._validate_probabilities()
        
        # Validate ratios
        self._validate_ratios()
    
    def _validate_probabilities(self):
        """Validate probability values"""
        probs = [
            self.synonym_replacement_prob,
            self.random_insertion_prob,
            self.random_deletion_prob,
            self.random_swap_prob
        ]
        
        for prob in probs:
            if not 0 <= prob <= 1:
                raise ValueError(f"Probability must be between 0 and 1, got {prob}")
    
    def _validate_ratios(self):
        """Validate ratio values"""
        ratios = [
            self.synonym_replacement_ratio,
            self.random_insertion_ratio,
            self.random_deletion_ratio,
            self.random_swap_ratio
        ]
        
        for ratio in ratios:
            if not 0 <= ratio <= 1:
                raise ValueError(f"Ratio must be between 0 and 1, got {ratio}")
    
    def get_augmentation_weights(self) -> Dict[AugmentationType, float]:
        """Get weights for different augmentation types"""
        weights = {}
        
        if AugmentationType.SYNONYM_REPLACEMENT in self.augmentation_types:
            weights[AugmentationType.SYNONYM_REPLACEMENT] = self.synonym_replacement_prob
        
        if AugmentationType.RANDOM_INSERTION in self.augmentation_types:
            weights[AugmentationType.RANDOM_INSERTION] = self.random_insertion_prob
        
        if AugmentationType.RANDOM_DELETION in self.augmentation_types:
            weights[AugmentationType.RANDOM_DELETION] = self.random_deletion_prob
        
        if AugmentationType.RANDOM_SWAP in self.augmentation_types:
            weights[AugmentationType.RANDOM_SWAP] = self.random_swap_prob
        
        return weights
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list) and value and isinstance(value[0], Enum):
                config_dict[key] = [item.value for item in value]
            elif isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AugmentationConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Predefined augmentation configurations
LIGHT_AUGMENTATION = AugmentationConfig(
    synonym_replacement_prob=0.05,
    random_insertion_prob=0.05,
    random_deletion_prob=0.05,
    random_swap_prob=0.05,
    num_augmentations_per_sample=1
)

MEDIUM_AUGMENTATION = AugmentationConfig(
    synonym_replacement_prob=0.1,
    random_insertion_prob=0.1,
    random_deletion_prob=0.1,
    random_swap_prob=0.1,
    num_augmentations_per_sample=2
)

HEAVY_AUGMENTATION = AugmentationConfig(
    synonym_replacement_prob=0.2,
    random_insertion_prob=0.15,
    random_deletion_prob=0.15,
    random_swap_prob=0.15,
    num_augmentations_per_sample=3
)

SYNONYM_ONLY_CONFIG = AugmentationConfig(
    augmentation_types=[AugmentationType.SYNONYM_REPLACEMENT],
    synonym_replacement_prob=0.2,
    num_augmentations_per_sample=2
)

BACK_TRANSLATION_CONFIG = AugmentationConfig(
    augmentation_types=[AugmentationType.BACK_TRANSLATION],
    back_translation_languages=["fr", "de", "es"],
    num_augmentations_per_sample=3
)

CONTEXTUAL_CONFIG = AugmentationConfig(
    augmentation_types=[AugmentationType.CONTEXTUAL_WORD_EMBEDDING],
    contextual_model_name="bert-base-uncased",
    contextual_top_k=5,
    num_augmentations_per_sample=2
)

PARAPHRASE_CONFIG = AugmentationConfig(
    augmentation_types=[AugmentationType.PARAPHRASE],
    paraphrase_model_name="tuner007/pegasus_paraphrase",
    paraphrase_num_return_sequences=3,
    num_augmentations_per_sample=3
)
