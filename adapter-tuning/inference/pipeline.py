"""
Inference pipeline for Adapter Tuning implementation
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import pipeline, Pipeline, AutoTokenizer
import logging

from ..adapters.adapter_model import AdapterModel
from ..config.model_config import ModelConfig
from ..config.adapter_config import AdapterConfig

logger = logging.getLogger(__name__)


class AdapterInferencePipeline:
    """High-level inference pipeline for adapter models"""
    
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        adapter_config: Optional[AdapterConfig] = None,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path or model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Load configurations if not provided
        if model_config is None or adapter_config is None:
            model_config, adapter_config = self._load_configs_from_path(model_path)
        
        self.model_config = model_config
        self.adapter_config = adapter_config
        
        # Load model and tokenizer
        self.adapter_model = self._load_adapter_model()
        self.tokenizer = self._load_tokenizer()
        
        # Determine task type
        self.task_type = self.model_config.task_type
        
        logger.info(f"Adapter inference pipeline initialized for task: {self.task_type}")
    
    def _load_configs_from_path(self, model_path: str):
        """Load model and adapter configs from saved path"""
        import os
        import json
        
        # Try to load from configs directory
        config_path = os.path.join(model_path, "configs", "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            model_config_dict = config_dict.get("model_config", {})
            adapter_config_dict = config_dict.get("adapter_config", {})
            
            model_config = ModelConfig(**model_config_dict)
            adapter_config = AdapterConfig.from_dict(adapter_config_dict)
            
            return model_config, adapter_config
        
        # Try to load adapter config directly
        adapter_config_path = os.path.join(self.adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config_dict = json.load(f)
            adapter_config = AdapterConfig.from_dict(adapter_config_dict)
        else:
            logger.warning("No adapter config found, using default")
            adapter_config = AdapterConfig()
        
        # Use default model config
        logger.warning("No model config found, using default")
        model_config = ModelConfig()
        
        return model_config, adapter_config
    
    def _load_adapter_model(self) -> AdapterModel:
        """Load the adapter model"""
        # Create adapter model
        adapter_model = AdapterModel(self.model_config, self.adapter_config)
        
        # Load adapter weights if available
        try:
            adapter_model.load_adapters(self.adapter_path)
            logger.info(f"Loaded adapters from {self.adapter_path}")
        except Exception as e:
            logger.warning(f"Could not load adapters: {e}")
        
        # Set to evaluation mode
        adapter_model.eval()
        adapter_model.to(self.device)
        
        return adapter_model
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer_name_or_path,
            cache_dir=self.model_config.cache_dir
        )
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        return tokenizer
    
    def classify_text(
        self,
        texts: Union[str, List[str]],
        return_all_scores: bool = False,
        top_k: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Classify text(s) using the adapter model"""
        
        if self.task_type not in ["classification", "text_classification"]:
            raise ValueError(f"classify_text not supported for task type: {self.task_type}")
        
        # Create classification pipeline
        classifier = pipeline(
            "text-classification",
            model=self.adapter_model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=return_all_scores,
            top_k=top_k,
        )
        
        # Handle single text vs list of texts
        if isinstance(texts, str):
            result = classifier(texts)
            return result[0] if isinstance(result, list) else result
        else:
            results = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_results = classifier(batch)
                results.extend(batch_results)
            return results
    
    def classify_tokens(
        self,
        texts: Union[str, List[str]],
        aggregation_strategy: str = "simple"
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Classify tokens using the adapter model (for NER, POS tagging)"""
        
        if self.task_type not in ["token_classification", "ner"]:
            raise ValueError(f"classify_tokens not supported for task type: {self.task_type}")
        
        # Create token classification pipeline
        token_classifier = pipeline(
            "token-classification",
            model=self.adapter_model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy=aggregation_strategy,
        )
        
        # Handle single text vs list of texts
        if isinstance(texts, str):
            return token_classifier(texts)
        else:
            results = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_results = token_classifier(batch)
                results.extend(batch_results)
            return results
    
    def answer_question(
        self,
        questions: Union[str, List[str]],
        contexts: Union[str, List[str]],
        max_answer_length: int = 30,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Answer questions using the adapter model"""
        
        if self.task_type != "qa":
            raise ValueError(f"answer_question not supported for task type: {self.task_type}")
        
        # Create question answering pipeline
        qa_pipeline = pipeline(
            "question-answering",
            model=self.adapter_model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )
        
        # Handle single question vs list of questions
        if isinstance(questions, str):
            if isinstance(contexts, str):
                result = qa_pipeline(question=questions, context=contexts, **kwargs)
                return result
            else:
                raise ValueError("If questions is a string, contexts must also be a string")
        else:
            if not isinstance(contexts, list) or len(questions) != len(contexts):
                raise ValueError("Questions and contexts must be lists of the same length")
            
            results = []
            for i in range(0, len(questions), self.batch_size):
                batch_questions = questions[i:i + self.batch_size]
                batch_contexts = contexts[i:i + self.batch_size]
                
                for q, c in zip(batch_questions, batch_contexts):
                    result = qa_pipeline(question=q, context=c, **kwargs)
                    results.append(result)
            
            return results
    
    def predict_custom(
        self,
        inputs: Union[str, List[str], Dict[str, Any]],
        **kwargs
    ) -> Union[Any, List[Any]]:
        """Custom prediction method for any task"""
        
        # Tokenize inputs
        if isinstance(inputs, str):
            inputs = [inputs]
        elif isinstance(inputs, dict):
            # Handle pre-tokenized inputs
            pass
        
        results = []
        
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            
            # Tokenize batch
            if isinstance(batch[0], str):
                tokenized = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.model_config.max_length,
                )
            else:
                tokenized = batch
            
            # Move to device
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.adapter_model(**tokenized)
                
                if self.task_type == "classification":
                    predictions = torch.softmax(outputs.logits, dim=-1)
                    predictions = predictions.cpu().numpy()
                elif self.task_type == "token_classification":
                    predictions = torch.softmax(outputs.logits, dim=-1)
                    predictions = predictions.cpu().numpy()
                else:
                    predictions = outputs.logits.cpu().numpy()
                
                results.extend(predictions)
        
        return results[0] if len(inputs) == 1 else results
    
    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        layer: int = -1,
        pooling: str = "mean"
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get embeddings from the adapter model"""
        
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.model_config.max_length,
            )
            
            # Move to device
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.adapter_model.base_model(**tokenized, output_hidden_states=True)
                
                # Get embeddings from specified layer
                embeddings = outputs.hidden_states[layer]
                
                # Apply pooling
                if pooling == "mean":
                    # Mean pooling (excluding padding tokens)
                    attention_mask = tokenized["attention_mask"].unsqueeze(-1)
                    embeddings = (embeddings * attention_mask).sum(1) / attention_mask.sum(1)
                elif pooling == "cls":
                    # Use CLS token embedding
                    embeddings = embeddings[:, 0, :]
                elif pooling == "max":
                    # Max pooling
                    embeddings = embeddings.max(1)[0]
                else:
                    raise ValueError(f"Unsupported pooling method: {pooling}")
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        return all_embeddings[0] if len(texts) == 1 else all_embeddings
    
    def switch_adapter(self, adapter_path: str):
        """Switch to a different adapter"""
        try:
            self.adapter_model.load_adapters(adapter_path)
            self.adapter_path = adapter_path
            logger.info(f"Switched to adapter: {adapter_path}")
        except Exception as e:
            logger.error(f"Failed to switch adapter: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.adapter_model.adapter_info
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        adapter_path: Optional[str] = None,
        **kwargs
    ) -> "AdapterInferencePipeline":
        """Load pipeline from pretrained model"""
        return cls(model_path, adapter_path, **kwargs)
