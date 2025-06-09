"""
Inference pipeline for LoRA/PEFT implementation
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import pipeline, Pipeline
import logging

from ..models.peft_model import PEFTModelWrapper
from ..config.model_config import ModelConfig, PEFTConfig

logger = logging.getLogger(__name__)


class InferencePipeline:
    """High-level inference pipeline for PEFT models"""
    
    def __init__(
        self,
        model_path: str,
        model_config: Optional[ModelConfig] = None,
        peft_config: Optional[PEFTConfig] = None,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Load model
        if model_config is None:
            # Try to load from saved config
            model_config = self._load_config_from_path(model_path)
        
        self.model_wrapper = PEFTModelWrapper(model_config, peft_config)
        self.model_wrapper.load_peft_model(model_path)
        self.model_wrapper.prepare_for_inference()
        
        # Get model and tokenizer
        self.model = self.model_wrapper.peft_model
        self.tokenizer = self.model_wrapper.get_tokenizer()
        
        logger.info(f"Inference pipeline initialized for model: {model_path}")
    
    def _load_config_from_path(self, model_path: str) -> ModelConfig:
        """Load model config from saved path"""
        import os
        import json
        
        config_path = os.path.join(model_path, "configs", "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            model_config_dict = config_dict.get("model_config", {})
            return ModelConfig(**model_config_dict)
        else:
            # Use default config
            logger.warning("No config found, using default ModelConfig")
            return ModelConfig()
    
    def classify_text(
        self,
        texts: Union[str, List[str]],
        return_all_scores: bool = False,
        top_k: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Classify text(s) using the model"""
        
        # Create classification pipeline
        classifier = pipeline(
            "text-classification",
            model=self.model,
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
    
    def generate_text(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text using the model"""
        
        # Create text generation pipeline
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs
        }
        
        # Handle single prompt vs list of prompts
        if isinstance(prompts, str):
            result = generator(prompts, **generation_kwargs)
            if num_return_sequences == 1:
                return result[0]["generated_text"]
            else:
                return [r["generated_text"] for r in result]
        else:
            results = []
            for i in range(0, len(prompts), self.batch_size):
                batch = prompts[i:i + self.batch_size]
                batch_results = generator(batch, **generation_kwargs)
                
                if num_return_sequences == 1:
                    results.extend([r[0]["generated_text"] for r in batch_results])
                else:
                    results.extend([[seq["generated_text"] for seq in r] for r in batch_results])
            
            return results
    
    def answer_question(
        self,
        questions: Union[str, List[str]],
        contexts: Union[str, List[str]],
        max_answer_length: int = 30,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Answer questions using the model"""
        
        # Create question answering pipeline
        qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
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
        task_type: str = "classification",
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
                    max_length=512,
                )
            else:
                tokenized = batch
            
            # Move to device
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**tokenized)
                
                if task_type == "classification":
                    predictions = torch.softmax(outputs.logits, dim=-1)
                    predictions = predictions.cpu().numpy()
                elif task_type == "regression":
                    predictions = outputs.logits.cpu().numpy()
                elif task_type == "generation":
                    # For generation, use generate method
                    generated = self.model.generate(
                        **tokenized,
                        max_new_tokens=kwargs.get("max_new_tokens", 50),
                        do_sample=kwargs.get("do_sample", False),
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    # Remove input tokens
                    generated = generated[:, tokenized["input_ids"].shape[1]:]
                    predictions = [
                        self.tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in generated
                    ]
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
        """Get embeddings from the model"""
        
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
                max_length=512,
            )
            
            # Move to device
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**tokenized, output_hidden_states=True)
                
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_wrapper.get_model_info()
    
    def save_pipeline(self, save_path: str):
        """Save the pipeline for later use"""
        self.model_wrapper.save_peft_model(save_path)
        logger.info(f"Pipeline saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        **kwargs
    ) -> "InferencePipeline":
        """Load pipeline from pretrained model"""
        return cls(model_path, **kwargs)
