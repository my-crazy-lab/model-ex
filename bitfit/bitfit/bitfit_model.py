"""
BitFit model wrapper implementation
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoModelForSequenceClassification, 
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    PreTrainedModel
)

from ..config.model_config import ModelConfig
from ..config.bitfit_config import BitFitConfig
from .parameter_utils import ParameterUtils, BiasParameterManager

logger = logging.getLogger(__name__)


class BitFitModel(nn.Module):
    """
    BitFit model wrapper that handles bias-only fine-tuning
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        bitfit_config: BitFitConfig,
        base_model: Optional[PreTrainedModel] = None
    ):
        super().__init__()
        
        self.model_config = model_config
        self.bitfit_config = bitfit_config
        
        # Load or use provided base model
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = self._load_base_model()
        
        # Initialize parameter manager
        self.parameter_manager = BiasParameterManager(bitfit_config)
        
        # Setup BitFit training
        self._setup_bitfit_training()
        
        # Store original state for analysis
        self.original_param_count = ParameterUtils.count_parameters(self.base_model)
        
        logger.info("BitFit model initialized successfully")
    
    def _load_base_model(self) -> PreTrainedModel:
        """Load the base pre-trained model"""
        model_kwargs = self.model_config.get_model_kwargs()
        
        if self.model_config.task_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name_or_path,
                num_labels=self.model_config.num_labels,
                **model_kwargs
            )
        elif self.model_config.task_type == "token_classification":
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_config.model_name_or_path,
                num_labels=self.model_config.num_labels,
                **model_kwargs
            )
        elif self.model_config.task_type == "question_answering":
            model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_config.model_name_or_path,
                **model_kwargs
            )
        else:
            model = AutoModel.from_pretrained(
                self.model_config.model_name_or_path,
                **model_kwargs
            )
        
        return model
    
    def _setup_bitfit_training(self):
        """Setup the model for BitFit training"""
        logger.info("Setting up BitFit training...")
        
        # Setup bias parameter training
        trainable_counts = self.parameter_manager.setup_bias_training(self.base_model)
        
        # Log setup results
        total_trainable = sum(trainable_counts.values())
        logger.info(f"Total trainable bias parameters: {total_trainable:,}")
        
        for bias_type, count in trainable_counts.items():
            if count > 0:
                logger.info(f"  {bias_type}: {count:,} parameters")
        
        # Print parameter summary
        ParameterUtils.print_parameter_summary(self.base_model, "BitFit Model")
    
    def forward(self, *args, **kwargs):
        """Forward pass through the base model"""
        return self.base_model(*args, **kwargs)
    
    def get_parameter_efficiency(self) -> Dict[str, float]:
        """Get parameter efficiency metrics"""
        return ParameterUtils.get_parameter_efficiency(self.base_model)
    
    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all trainable parameters (bias only)"""
        trainable_params = {}
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param
        return trainable_params
    
    def get_bias_statistics(self) -> Dict[str, Any]:
        """Get statistics about bias parameters"""
        bias_params = self.parameter_manager.identify_bias_parameters(self.base_model)
        
        stats = {
            "bias_parameter_counts": {},
            "bias_parameter_norms": {},
            "bias_gradient_norms": {}
        }
        
        for bias_type, param_names in bias_params.items():
            if not param_names:
                continue
            
            param_count = 0
            param_norm = 0.0
            grad_norm = 0.0
            
            for param_name in param_names:
                param = dict(self.base_model.named_parameters())[param_name]
                param_count += param.numel()
                param_norm += param.norm().item()
                
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
            
            stats["bias_parameter_counts"][bias_type] = param_count
            stats["bias_parameter_norms"][bias_type] = param_norm
            stats["bias_gradient_norms"][bias_type] = grad_norm
        
        return stats
    
    def apply_bias_noise(self):
        """Apply noise to bias parameters during training"""
        if not self.bitfit_config.apply_bias_noise:
            return
        
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if param.requires_grad and 'bias' in name:
                    noise = torch.randn_like(param) * self.bitfit_config.bias_noise_std
                    param.add_(noise)
    
    def save_bitfit_model(self, save_path: str):
        """Save BitFit model (bias parameters only if configured)"""
        os.makedirs(save_path, exist_ok=True)
        
        if self.bitfit_config.save_bias_only:
            # Save only bias parameters
            bias_state_dict = {}
            for name, param in self.base_model.named_parameters():
                if param.requires_grad and 'bias' in name:
                    bias_state_dict[name] = param.cpu()
            
            torch.save(bias_state_dict, os.path.join(save_path, "bias_parameters.bin"))
            logger.info(f"Saved {len(bias_state_dict)} bias parameters to {save_path}")
        else:
            # Save full model
            self.base_model.save_pretrained(save_path)
            logger.info(f"Saved full model to {save_path}")
        
        # Save configurations
        with open(os.path.join(save_path, "model_config.json"), 'w') as f:
            json.dump(self.model_config.__dict__, f, indent=2)
        
        with open(os.path.join(save_path, "bitfit_config.json"), 'w') as f:
            json.dump(self.bitfit_config.to_dict(), f, indent=2)
        
        # Save parameter statistics
        efficiency = self.get_parameter_efficiency()
        with open(os.path.join(save_path, "parameter_stats.json"), 'w') as f:
            json.dump(efficiency, f, indent=2)
    
    def load_bitfit_model(self, load_path: str):
        """Load BitFit model"""
        bias_params_path = os.path.join(load_path, "bias_parameters.bin")
        
        if os.path.exists(bias_params_path):
            # Load bias parameters only
            bias_state_dict = torch.load(bias_params_path, map_location="cpu")
            
            # Load bias parameters into model
            model_state_dict = self.base_model.state_dict()
            for name, param in bias_state_dict.items():
                if name in model_state_dict:
                    model_state_dict[name] = param
            
            self.base_model.load_state_dict(model_state_dict)
            logger.info(f"Loaded {len(bias_state_dict)} bias parameters from {load_path}")
        else:
            # Load full model
            self.base_model = self.base_model.from_pretrained(load_path)
            logger.info(f"Loaded full model from {load_path}")
        
        # Re-setup BitFit training
        self._setup_bitfit_training()
    
    def compare_with_full_finetuning(self) -> Dict[str, Any]:
        """Compare BitFit efficiency with full fine-tuning"""
        current_counts = ParameterUtils.count_parameters(self.base_model)
        
        comparison = {
            "full_finetuning": {
                "trainable_params": self.original_param_count["total"],
                "memory_mb": self.original_param_count["total"] * 4 / (1024 * 1024),  # float32
                "relative_efficiency": 100.0
            },
            "bitfit": {
                "trainable_params": current_counts["trainable"],
                "memory_mb": current_counts["trainable"] * 4 / (1024 * 1024),
                "relative_efficiency": current_counts["trainable"] / self.original_param_count["total"] * 100
            }
        }
        
        comparison["improvement"] = {
            "parameter_reduction": self.original_param_count["total"] / max(current_counts["trainable"], 1),
            "memory_reduction": comparison["full_finetuning"]["memory_mb"] / max(comparison["bitfit"]["memory_mb"], 0.001),
            "efficiency_gain": 100.0 / comparison["bitfit"]["relative_efficiency"]
        }
        
        return comparison
    
    def print_comparison(self):
        """Print comparison with full fine-tuning"""
        comparison = self.compare_with_full_finetuning()
        
        print("\nBitFit vs Full Fine-tuning Comparison:")
        print("=" * 50)
        print(f"Full Fine-tuning:")
        print(f"  Trainable parameters: {comparison['full_finetuning']['trainable_params']:,}")
        print(f"  Memory usage: {comparison['full_finetuning']['memory_mb']:.1f} MB")
        print()
        print(f"BitFit:")
        print(f"  Trainable parameters: {comparison['bitfit']['trainable_params']:,}")
        print(f"  Memory usage: {comparison['bitfit']['memory_mb']:.1f} MB")
        print(f"  Parameter efficiency: {comparison['bitfit']['relative_efficiency']:.4f}%")
        print()
        print(f"Improvements:")
        print(f"  Parameter reduction: {comparison['improvement']['parameter_reduction']:.1f}x")
        print(f"  Memory reduction: {comparison['improvement']['memory_reduction']:.1f}x")
        print(f"  Efficiency gain: {comparison['improvement']['efficiency_gain']:.1f}x")
        print("=" * 50)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_config: Optional[ModelConfig] = None,
        bitfit_config: Optional[BitFitConfig] = None
    ) -> "BitFitModel":
        """Load BitFit model from pretrained path"""
        
        # Load configurations if not provided
        if model_config is None:
            config_path = os.path.join(model_path, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                model_config = ModelConfig(**config_dict)
            else:
                raise ValueError(f"No model config found at {config_path}")
        
        if bitfit_config is None:
            config_path = os.path.join(model_path, "bitfit_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                bitfit_config = BitFitConfig.from_dict(config_dict)
            else:
                raise ValueError(f"No BitFit config found at {config_path}")
        
        # Create model
        model = cls(model_config, bitfit_config)
        
        # Load parameters
        model.load_bitfit_model(model_path)
        
        return model
