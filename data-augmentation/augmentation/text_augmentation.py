"""
Core text augmentation techniques
"""

import random
import re
import nltk
import spacy
from typing import List, Optional, Dict, Any
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import numpy as np

from ..config.augmentation_config import AugmentationConfig, AugmentationType

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class SynonymReplacer:
    """Handles synonym replacement augmentation"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.use_wordnet = config.use_wordnet
        self.replacement_ratio = config.synonym_replacement_ratio
        
        # Load spaCy model for POS tagging if needed
        if config.use_pos_tagging:
            try:
                self.nlp = spacy.load(config.pos_model)
            except OSError:
                print(f"Warning: spaCy model {config.pos_model} not found. Using NLTK POS tagging.")
                self.nlp = None
        else:
            self.nlp = None
    
    def get_synonyms(self, word: str, pos: Optional[str] = None) -> List[str]:
        """Get synonyms for a word using WordNet"""
        if not self.use_wordnet:
            return []
        
        synonyms = set()
        
        # Get WordNet synsets
        synsets = wordnet.synsets(word)
        
        # Filter by POS if provided
        if pos:
            pos_map = {
                'NOUN': wordnet.NOUN,
                'VERB': wordnet.VERB,
                'ADJ': wordnet.ADJ,
                'ADV': wordnet.ADV
            }
            if pos in pos_map:
                synsets = [s for s in synsets if s.pos() == pos_map[pos]]
        
        # Extract synonyms
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def should_replace_word(self, word: str, pos: Optional[str] = None) -> bool:
        """Determine if a word should be replaced"""
        # Skip if word is too short
        if len(word) < 3:
            return False
        
        # Skip proper nouns and numbers if configured
        if pos and pos in self.config.exclude_pos_tags:
            return False
        
        # Skip special tokens
        if self.config.preserve_special_tokens and (
            word.startswith('@') or word.startswith('#') or 
            word.startswith('http') or word.isnumeric()
        ):
            return False
        
        return True
    
    def replace_synonyms(self, text: str) -> str:
        """Replace words with synonyms"""
        if self.nlp:
            # Use spaCy for tokenization and POS tagging
            doc = self.nlp(text)
            words = [(token.text, token.pos_) for token in doc]
        else:
            # Use NLTK for tokenization and POS tagging
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            words = [(word, pos) for word, pos in pos_tags]
        
        # Calculate number of words to replace
        num_words_to_replace = max(1, int(len(words) * self.replacement_ratio))
        
        # Randomly select words to replace
        replaceable_indices = [
            i for i, (word, pos) in enumerate(words)
            if self.should_replace_word(word, pos)
        ]
        
        if not replaceable_indices:
            return text
        
        indices_to_replace = random.sample(
            replaceable_indices,
            min(num_words_to_replace, len(replaceable_indices))
        )
        
        # Replace selected words
        new_words = words.copy()
        for idx in indices_to_replace:
            word, pos = words[idx]
            synonyms = self.get_synonyms(word, pos)
            
            if synonyms:
                new_word = random.choice(synonyms)
                new_words[idx] = (new_word, pos)
        
        # Reconstruct text (simple approach)
        return ' '.join([word for word, _ in new_words])


class RandomOperations:
    """Handles random insertion, deletion, and swap operations"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
        # Load word lists for insertion
        self.insertion_words = self._load_insertion_words()
    
    def _load_insertion_words(self) -> Dict[str, List[str]]:
        """Load words for random insertion by POS type"""
        # This is a simplified version - in practice, you'd load from files
        return {
            "noun": ["thing", "person", "place", "idea", "concept"],
            "verb": ["is", "was", "seems", "appears", "looks"],
            "adjective": ["good", "bad", "nice", "great", "small"],
            "adverb": ["very", "quite", "really", "somewhat", "rather"]
        }
    
    def random_insertion(self, text: str) -> str:
        """Randomly insert words into text"""
        words = text.split()
        
        if len(words) < 2:
            return text
        
        num_insertions = max(1, int(len(words) * self.config.random_insertion_ratio))
        
        for _ in range(num_insertions):
            # Choose random position
            pos = random.randint(0, len(words))
            
            # Choose random word type and word
            word_type = random.choice(self.config.insertion_word_types)
            if word_type in self.insertion_words:
                new_word = random.choice(self.insertion_words[word_type])
                words.insert(pos, new_word)
        
        return ' '.join(words)
    
    def random_deletion(self, text: str) -> str:
        """Randomly delete words from text"""
        words = text.split()
        
        if len(words) <= self.config.min_words_after_deletion:
            return text
        
        num_deletions = max(1, int(len(words) * self.config.random_deletion_ratio))
        max_deletions = len(words) - self.config.min_words_after_deletion
        num_deletions = min(num_deletions, max_deletions)
        
        # Randomly select indices to delete
        indices_to_delete = random.sample(range(len(words)), num_deletions)
        
        # Remove words (in reverse order to maintain indices)
        for idx in sorted(indices_to_delete, reverse=True):
            del words[idx]
        
        return ' '.join(words)
    
    def random_swap(self, text: str) -> str:
        """Randomly swap adjacent words"""
        words = text.split()
        
        if len(words) < 2:
            return text
        
        num_swaps = max(1, int(len(words) * self.config.random_swap_ratio))
        
        for _ in range(num_swaps):
            # Choose random adjacent pair
            idx = random.randint(0, len(words) - 2)
            
            # Swap words
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)


class TextAugmenter:
    """Main text augmentation class"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
        # Initialize augmentation components
        self.synonym_replacer = SynonymReplacer(config)
        self.random_operations = RandomOperations(config)
        
        # Set random seed for reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
    
    def augment_single(self, text: str, augmentation_type: AugmentationType) -> str:
        """Apply single augmentation technique"""
        if augmentation_type == AugmentationType.SYNONYM_REPLACEMENT:
            return self.synonym_replacer.replace_synonyms(text)
        
        elif augmentation_type == AugmentationType.RANDOM_INSERTION:
            return self.random_operations.random_insertion(text)
        
        elif augmentation_type == AugmentationType.RANDOM_DELETION:
            return self.random_operations.random_deletion(text)
        
        elif augmentation_type == AugmentationType.RANDOM_SWAP:
            return self.random_operations.random_swap(text)
        
        else:
            # For other types (back translation, paraphrase, etc.)
            # These would be implemented in separate modules
            return text
    
    def should_apply_augmentation(self, augmentation_type: AugmentationType) -> bool:
        """Determine if augmentation should be applied based on probability"""
        weights = self.config.get_augmentation_weights()
        prob = weights.get(augmentation_type, 0.0)
        return random.random() < prob
    
    def augment(
        self,
        text: str,
        num_augmentations: Optional[int] = None,
        augmentation_types: Optional[List[AugmentationType]] = None
    ) -> List[str]:
        """
        Augment text using configured techniques
        
        Args:
            text: Input text to augment
            num_augmentations: Number of augmentations to generate
            augmentation_types: Specific augmentation types to use
            
        Returns:
            List of augmented texts
        """
        if num_augmentations is None:
            num_augmentations = self.config.num_augmentations_per_sample
        
        if augmentation_types is None:
            augmentation_types = self.config.augmentation_types
        
        # Validate input
        if len(text.strip()) < self.config.min_text_length:
            return [text]  # Return original if too short
        
        augmented_texts = []
        
        for _ in range(min(num_augmentations, self.config.max_augmentations_per_sample)):
            augmented_text = text
            
            # Apply augmentations
            for aug_type in augmentation_types:
                if self.should_apply_augmentation(aug_type):
                    augmented_text = self.augment_single(augmented_text, aug_type)
            
            # Validate augmented text
            if self._is_valid_augmentation(text, augmented_text):
                augmented_texts.append(augmented_text)
        
        # Remove duplicates if configured
        if len(augmented_texts) > 1:
            augmented_texts = self._remove_duplicates(augmented_texts)
        
        return augmented_texts
    
    def _is_valid_augmentation(self, original: str, augmented: str) -> bool:
        """Check if augmentation is valid"""
        # Check length constraints
        if len(augmented) < self.config.min_text_length:
            return False
        
        if len(augmented) > self.config.max_text_length:
            return False
        
        # Check similarity (avoid too similar augmentations)
        similarity = self._calculate_similarity(original, augmented)
        if similarity > self.config.similarity_threshold:
            return False
        
        return True
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple Jaccard similarity)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _remove_duplicates(self, texts: List[str]) -> List[str]:
        """Remove duplicate texts"""
        seen = set()
        unique_texts = []
        
        for text in texts:
            text_lower = text.lower().strip()
            if text_lower not in seen:
                seen.add(text_lower)
                unique_texts.append(text)
        
        return unique_texts
    
    def augment_dataset(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None
    ) -> Dict[str, List[Any]]:
        """
        Augment entire dataset
        
        Args:
            texts: List of input texts
            labels: Optional list of labels
            
        Returns:
            Dictionary with augmented texts and labels
        """
        augmented_texts = []
        augmented_labels = []
        
        for i, text in enumerate(texts):
            # Add original
            augmented_texts.append(text)
            if labels is not None:
                augmented_labels.append(labels[i])
            
            # Add augmentations
            aug_texts = self.augment(text)
            augmented_texts.extend(aug_texts)
            
            if labels is not None:
                # Repeat label for each augmentation
                augmented_labels.extend([labels[i]] * len(aug_texts))
        
        result = {"texts": augmented_texts}
        if labels is not None:
            result["labels"] = augmented_labels
        
        return result
