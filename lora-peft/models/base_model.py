"""
Base model wrapper for LoRA/PEFT implementation
"""

import torch
from typing import Optional, Dict, Any, Union
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import TaskType
import logging

from ..config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class BaseModelWrapper:
    """Wrapper class for base models with quantization support"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer"""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer: {self.config.tokenizer_name_or_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name_or_path,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            logger.info(f"Tokenizer loaded successfully. Vocab size: {len(self.tokenizer)}")
        
        return self.tokenizer
    
    def load_model(self, task_type: TaskType) -> PreTrainedModel:
        """Load model based on task type"""
        if self.model is None:
            logger.info(f"Loading model: {self.config.model_name_or_path}")
            
            # Prepare model loading arguments
            model_kwargs = {
                "cache_dir": self.config.cache_dir,
                "trust_remote_code": True,
                "device_map": self.config.device_map,
            }
            
            # Add quantization config if enabled
            if self.config.use_quantization:
                model_kwargs["quantization_config"] = self._get_quantization_config()
                model_kwargs["torch_dtype"] = torch.float16
            else:
                if self.config.torch_dtype != "auto":
                    model_kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)
            
            # Load model based on task type
            if task_type == TaskType.SEQ_CLS:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name_or_path,
                    num_labels=self.config.num_labels,
                    **model_kwargs
                )
            elif task_type == TaskType.CAUSAL_LM:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name_or_path,
                    **model_kwargs
                )
            elif task_type == TaskType.QUESTION_ANS:
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    self.config.model_name_or_path,
                    **model_kwargs
                )
            elif task_type == TaskType.SEQ_2_SEQ_LM:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name_or_path,
                    **model_kwargs
                )
            else:
                # Default to AutoModel for other tasks
                self.model = AutoModel.from_pretrained(
                    self.config.model_name_or_path,
                    **model_kwargs
                )
            
            # Resize token embeddings if tokenizer was modified
            if self.tokenizer is not None:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
            
        return self.model
    
    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """Get quantization configuration"""
        if self.config.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            raise ValueError(f"Unsupported quantization bits: {self.config.quantization_bits}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {"status": "Model not loaded"}
        
        info = {
            "model_name": self.config.model_name_or_path,
            "num_parameters": self.model.num_parameters(),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "device": next(self.model.parameters()).device,
            "dtype": next(self.model.parameters()).dtype,
            "quantization": self.config.use_quantization,
        }
        
        if self.config.use_quantization:
            info["quantization_bits"] = self.config.quantization_bits
        
        return info
    
    def save_model(self, save_directory: str):
        """Save model and tokenizer"""
        if self.model is not None:
            self.model.save_pretrained(save_directory)
            logger.info(f"Model saved to {save_directory}")
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
            logger.info(f"Tokenizer saved to {save_directory}")
    
    def load_from_checkpoint(self, checkpoint_path: str, task_type: TaskType):
        """Load model from checkpoint"""
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Update config to load from checkpoint
        original_path = self.config.model_name_or_path
        self.config.model_name_or_path = checkpoint_path
        
        try:
            self.load_tokenizer()
            self.load_model(task_type)
        except Exception as e:
            # Restore original path if loading fails
            self.config.model_name_or_path = original_path
            raise e
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        if self.model is not None:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    
    def prepare_for_training(self):
        """Prepare model for training"""
        if self.model is not None:
            # Enable training mode
            self.model.train()
            
            # Enable gradient computation for all parameters
            for param in self.model.parameters():
                param.requires_grad = True
            
            logger.info("Model prepared for training")
    
    def prepare_for_inference(self):
        """Prepare model for inference"""
        if self.model is not None:
            # Enable evaluation mode
            self.model.eval()
            
            # Disable gradient computation
            for param in self.model.parameters():
                param.requires_grad = False
            
            logger.info("Model prepared for inference")
