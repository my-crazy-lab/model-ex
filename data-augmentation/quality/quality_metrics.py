"""
Quality metrics for synthetic data assessment
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class FluencyScorer:
    """Measures text fluency using language model perplexity"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the language model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.model.eval()
            logger.info(f"Initialized fluency model: {self.model_name}")
        
        except Exception as e:
            logger.error(f"Failed to initialize fluency model: {e}")
            raise
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of text"""
        if not text.strip():
            return float('inf')
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Calculate loss
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            # Convert loss to perplexity
            perplexity = torch.exp(loss).item()
            
            return perplexity
        
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')
    
    def score_fluency(self, text: str) -> float:
        """Score fluency (0-1, higher is better)"""
        perplexity = self.calculate_perplexity(text)
        
        # Convert perplexity to 0-1 score (lower perplexity = higher fluency)
        # Using sigmoid transformation
        fluency_score = 1 / (1 + math.log(max(perplexity, 1.0)))
        
        return min(max(fluency_score, 0.0), 1.0)
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """Score fluency for multiple texts"""
        scores = []
        for text in texts:
            score = self.score_fluency(text)
            scores.append(score)
        return scores


class CoherenceScorer:
    """Measures text coherence using sentence embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized coherence model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize coherence model: {e}")
            raise
    
    def calculate_sentence_similarity(self, sentences: List[str]) -> float:
        """Calculate average similarity between consecutive sentences"""
        if len(sentences) < 2:
            return 1.0  # Single sentence is perfectly coherent
        
        # Get sentence embeddings
        embeddings = self.model.encode(sentences)
        
        # Calculate similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def score_coherence(self, text: str) -> float:
        """Score text coherence (0-1, higher is better)"""
        if not text.strip():
            return 0.0
        
        try:
            # Split into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) < 2:
                return 1.0  # Single sentence
            
            # Calculate coherence
            coherence = self.calculate_sentence_similarity(sentences)
            
            # Normalize to 0-1 range
            coherence_score = (coherence + 1) / 2  # Cosine similarity is in [-1, 1]
            
            return min(max(coherence_score, 0.0), 1.0)
        
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            return 0.0
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """Score coherence for multiple texts"""
        scores = []
        for text in texts:
            score = self.score_coherence(text)
            scores.append(score)
        return scores


class GrammaticalityScorer:
    """Measures grammatical correctness"""
    
    def __init__(self):
        # This is a simplified version - in practice, you'd use a grammar checker
        self.stop_words = set(stopwords.words('english'))
    
    def count_grammar_errors(self, text: str) -> int:
        """Count basic grammar errors (simplified)"""
        errors = 0
        
        # Check for basic issues
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            
            # Check if sentence starts with capital letter
            if sentence and not sentence[0].isupper():
                errors += 1
            
            # Check if sentence ends with punctuation
            if sentence and sentence[-1] not in '.!?':
                errors += 1
            
            # Check for very short sentences (might be fragments)
            if len(words) < 3:
                errors += 1
            
            # Check for repeated words
            for i in range(len(words) - 1):
                if words[i] == words[i + 1] and words[i] not in self.stop_words:
                    errors += 1
        
        return errors
    
    def score_grammaticality(self, text: str) -> float:
        """Score grammatical correctness (0-1, higher is better)"""
        if not text.strip():
            return 0.0
        
        errors = self.count_grammar_errors(text)
        sentences = len(sent_tokenize(text))
        
        if sentences == 0:
            return 0.0
        
        # Calculate error rate
        error_rate = errors / sentences
        
        # Convert to score (fewer errors = higher score)
        score = 1 / (1 + error_rate)
        
        return min(max(score, 0.0), 1.0)


class QualityMetrics:
    """Comprehensive quality assessment"""
    
    def __init__(
        self,
        fluency_model: str = "gpt2",
        coherence_model: str = "all-MiniLM-L6-v2",
        use_gpu: bool = True
    ):
        self.fluency_scorer = FluencyScorer(fluency_model)
        self.coherence_scorer = CoherenceScorer(coherence_model)
        self.grammaticality_scorer = GrammaticalityScorer()
    
    def assess_quality(self, text: str) -> Dict[str, float]:
        """Comprehensive quality assessment"""
        return {
            "fluency": self.fluency_scorer.score_fluency(text),
            "coherence": self.coherence_scorer.score_coherence(text),
            "grammaticality": self.grammaticality_scorer.score_grammaticality(text)
        }
    
    def assess_batch(self, texts: List[str]) -> Dict[str, List[float]]:
        """Assess quality for multiple texts"""
        fluency_scores = self.fluency_scorer.score_batch(texts)
        coherence_scores = self.coherence_scorer.score_batch(texts)
        grammaticality_scores = [
            self.grammaticality_scorer.score_grammaticality(text) for text in texts
        ]
        
        return {
            "fluency": fluency_scores,
            "coherence": coherence_scores,
            "grammaticality": grammaticality_scores
        }
    
    def calculate_overall_score(
        self,
        quality_scores: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate weighted overall quality score"""
        if weights is None:
            weights = {"fluency": 0.4, "coherence": 0.3, "grammaticality": 0.3}
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, score in quality_scores.items():
            if metric in weights:
                overall_score += score * weights[metric]
                total_weight += weights[metric]
        
        if total_weight > 0:
            overall_score /= total_weight
        
        return overall_score
    
    def filter_by_quality(
        self,
        texts: List[str],
        min_fluency: float = 0.5,
        min_coherence: float = 0.5,
        min_grammaticality: float = 0.5,
        min_overall: float = 0.6
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """Filter texts by quality thresholds"""
        quality_scores = self.assess_batch(texts)
        
        filtered_texts = []
        filtered_scores = []
        
        for i, text in enumerate(texts):
            scores = {
                "fluency": quality_scores["fluency"][i],
                "coherence": quality_scores["coherence"][i],
                "grammaticality": quality_scores["grammaticality"][i]
            }
            
            overall_score = self.calculate_overall_score(scores)
            scores["overall"] = overall_score
            
            # Check thresholds
            if (scores["fluency"] >= min_fluency and
                scores["coherence"] >= min_coherence and
                scores["grammaticality"] >= min_grammaticality and
                overall_score >= min_overall):
                
                filtered_texts.append(text)
                filtered_scores.append(scores)
        
        return filtered_texts, filtered_scores
    
    def rank_by_quality(
        self,
        texts: List[str],
        metric: str = "overall"
    ) -> List[Tuple[str, float]]:
        """Rank texts by quality metric"""
        quality_scores = self.assess_batch(texts)
        
        if metric == "overall":
            scores = [
                self.calculate_overall_score({
                    "fluency": quality_scores["fluency"][i],
                    "coherence": quality_scores["coherence"][i],
                    "grammaticality": quality_scores["grammaticality"][i]
                })
                for i in range(len(texts))
            ]
        else:
            scores = quality_scores.get(metric, [0.0] * len(texts))
        
        # Create (text, score) pairs and sort by score
        text_score_pairs = list(zip(texts, scores))
        text_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return text_score_pairs
