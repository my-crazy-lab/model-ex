"""
PEFT model wrapper for LoRA/PEFT implementation
"""

import torch
from typing import Optional, Dict, Any, List
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    IA3Config,
    TaskType,
    PeftModel,
    PeftConfig
)
import logging

from .base_model import BaseModelWrapper
from ..config.model_config import ModelConfig, PEFTConfig

logger = logging.getLogger(__name__)


class PEFTModelWrapper:
    """Wrapper class for PEFT models"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        peft_config: PEFTConfig
    ):
        self.model_config = model_config
        self.peft_config = peft_config
        self.base_model_wrapper = BaseModelWrapper(model_config)
        self.peft_model: Optional[PeftModel] = None
        
    def load_model(self) -> PeftModel:
        """Load and configure PEFT model"""
        if self.peft_model is None:
            # Load tokenizer first
            tokenizer = self.base_model_wrapper.load_tokenizer()
            
            # Load base model
            base_model = self.base_model_wrapper.load_model(self.peft_config.task_type)
            
            # Configure PEFT
            peft_config = self._create_peft_config()
            
            # Create PEFT model
            self.peft_model = get_peft_model(base_model, peft_config)
            
            # Print trainable parameters
            self._print_trainable_parameters()
            
            logger.info("PEFT model created successfully")
        
        return self.peft_model
    
    def _create_peft_config(self) -> PeftConfig:
        """Create PEFT configuration based on method"""
        if self.peft_config.peft_type == "LORA":
            return self._create_lora_config()
        elif self.peft_config.peft_type == "PREFIX_TUNING":
            return self._create_prefix_tuning_config()
        elif self.peft_config.peft_type == "PROMPT_TUNING":
            return self._create_prompt_tuning_config()
        elif self.peft_config.peft_type == "IA3":
            return self._create_ia3_config()
        else:
            raise ValueError(f"Unsupported PEFT type: {self.peft_config.peft_type}")
    
    def _create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration"""
        # Set target modules if not specified
        target_modules = self.peft_config.target_modules
        if target_modules is None:
            target_modules = self.peft_config.get_target_modules_for_model(
                self.model_config.model_name_or_path
            )
        
        return LoraConfig(
            task_type=self.peft_config.task_type,
            inference_mode=self.peft_config.inference_mode,
            r=self.peft_config.r,
            lora_alpha=self.peft_config.lora_alpha,
            lora_dropout=self.peft_config.lora_dropout,
            target_modules=target_modules,
            bias=self.peft_config.bias,
        )
    
    def _create_prefix_tuning_config(self) -> PrefixTuningConfig:
        """Create Prefix Tuning configuration"""
        return PrefixTuningConfig(
            task_type=self.peft_config.task_type,
            inference_mode=self.peft_config.inference_mode,
            num_virtual_tokens=self.peft_config.num_virtual_tokens,
            prefix_projection=self.peft_config.prefix_projection,
        )
    
    def _create_prompt_tuning_config(self) -> PromptTuningConfig:
        """Create Prompt Tuning configuration"""
        return PromptTuningConfig(
            task_type=self.peft_config.task_type,
            inference_mode=self.peft_config.inference_mode,
            num_virtual_tokens=self.peft_config.num_virtual_tokens,
            num_transformer_submodules=self.peft_config.num_transformer_submodules,
        )
    
    def _create_ia3_config(self) -> IA3Config:
        """Create IA3 configuration"""
        # Set target modules if not specified
        target_modules = self.peft_config.target_modules
        if target_modules is None:
            target_modules = self.peft_config.get_target_modules_for_model(
                self.model_config.model_name_or_path
            )
        
        feedforward_modules = self.peft_config.feedforward_modules
        if feedforward_modules is None:
            # Default feedforward modules for different models
            model_name = self.model_config.model_name_or_path.lower()
            if "bert" in model_name:
                feedforward_modules = ["intermediate.dense"]
            elif "llama" in model_name:
                feedforward_modules = ["mlp.gate_proj", "mlp.up_proj"]
            elif "t5" in model_name:
                feedforward_modules = ["DenseReluDense.wi"]
            else:
                feedforward_modules = ["mlp"]
        
        return IA3Config(
            task_type=self.peft_config.task_type,
            inference_mode=self.peft_config.inference_mode,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
        )
    
    def _print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        if self.peft_model is not None:
            trainable_params = 0
            all_param = 0
            
            for _, param in self.peft_model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            percentage = 100 * trainable_params / all_param
            
            logger.info(
                f"Trainable params: {trainable_params:,} || "
                f"All params: {all_param:,} || "
                f"Trainable%: {percentage:.2f}%"
            )
    
    def save_peft_model(self, save_directory: str):
        """Save PEFT model"""
        if self.peft_model is not None:
            self.peft_model.save_pretrained(save_directory)
            logger.info(f"PEFT model saved to {save_directory}")
        else:
            raise ValueError("PEFT model not loaded")
    
    def load_peft_model(self, peft_model_path: str):
        """Load PEFT model from checkpoint"""
        logger.info(f"Loading PEFT model from: {peft_model_path}")
        
        # Load base model first
        base_model = self.base_model_wrapper.load_model(self.peft_config.task_type)
        
        # Load PEFT model
        from peft import PeftModel
        self.peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
        
        logger.info("PEFT model loaded successfully")
        return self.peft_model
    
    def merge_and_unload(self):
        """Merge PEFT weights with base model and unload"""
        if self.peft_model is not None:
            merged_model = self.peft_model.merge_and_unload()
            logger.info("PEFT weights merged with base model")
            return merged_model
        else:
            raise ValueError("PEFT model not loaded")
    
    def get_peft_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get PEFT model state dict"""
        if self.peft_model is not None:
            return get_peft_model_state_dict(self.peft_model)
        else:
            raise ValueError("PEFT model not loaded")
    
    def set_peft_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Set PEFT model state dict"""
        if self.peft_model is not None:
            set_peft_model_state_dict(self.peft_model, state_dict)
            logger.info("PEFT state dict loaded")
        else:
            raise ValueError("PEFT model not loaded")
    
    def enable_adapters(self):
        """Enable PEFT adapters"""
        if self.peft_model is not None:
            self.peft_model.enable_adapters()
            logger.info("PEFT adapters enabled")
    
    def disable_adapters(self):
        """Disable PEFT adapters"""
        if self.peft_model is not None:
            self.peft_model.disable_adapters()
            logger.info("PEFT adapters disabled")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        base_info = self.base_model_wrapper.get_model_info()
        
        if self.peft_model is not None:
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.peft_model.parameters())
            
            peft_info = {
                "peft_type": self.peft_config.peft_type,
                "task_type": self.peft_config.task_type.value,
                "trainable_parameters": trainable_params,
                "trainable_percentage": 100 * trainable_params / all_params,
                "peft_config": self.peft_config.__dict__,
            }
            
            base_info.update(peft_info)
        
        return base_info
    
    def get_tokenizer(self):
        """Get tokenizer"""
        return self.base_model_wrapper.load_tokenizer()
    
    def prepare_for_training(self):
        """Prepare model for training"""
        if self.peft_model is not None:
            self.peft_model.train()
            logger.info("PEFT model prepared for training")
    
    def prepare_for_inference(self):
        """Prepare model for inference"""
        if self.peft_model is not None:
            self.peft_model.eval()
            logger.info("PEFT model prepared for inference")
