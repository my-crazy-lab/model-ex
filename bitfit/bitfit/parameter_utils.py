"""
Parameter utilities for BitFit implementation
"""

import re
import logging
from typing import Dict, List, Tuple, Set, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..config.bitfit_config import BitFitConfig, BiasType

logger = logging.getLogger(__name__)


class ParameterUtils:
    """Utilities for parameter management in BitFit"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count different types of parameters"""
        total_params = 0
        trainable_params = 0
        bias_params = 0
        weight_params = 0
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            
            if 'bias' in name:
                bias_params += param_count
            else:
                weight_params += param_count
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "bias": bias_params,
            "weight": weight_params,
            "frozen": total_params - trainable_params
        }
    
    @staticmethod
    def get_parameter_efficiency(model: nn.Module) -> Dict[str, float]:
        """Calculate parameter efficiency metrics"""
        counts = ParameterUtils.count_parameters(model)
        
        if counts["total"] == 0:
            return {"efficiency": 0.0, "bias_ratio": 0.0, "trainable_ratio": 0.0}
        
        return {
            "efficiency": counts["trainable"] / counts["total"] * 100,
            "bias_ratio": counts["bias"] / counts["total"] * 100,
            "trainable_ratio": counts["trainable"] / counts["total"] * 100,
            "reduction_factor": counts["total"] / max(counts["trainable"], 1)
        }
    
    @staticmethod
    def print_parameter_summary(model: nn.Module, model_name: str = "Model"):
        """Print detailed parameter summary"""
        counts = ParameterUtils.count_parameters(model)
        efficiency = ParameterUtils.get_parameter_efficiency(model)
        
        print(f"\n{model_name} Parameter Summary:")
        print("=" * 50)
        print(f"Total parameters: {counts['total']:,}")
        print(f"Trainable parameters: {counts['trainable']:,}")
        print(f"Frozen parameters: {counts['frozen']:,}")
        print(f"Bias parameters: {counts['bias']:,}")
        print(f"Weight parameters: {counts['weight']:,}")
        print()
        print(f"Parameter efficiency: {efficiency['efficiency']:.4f}%")
        print(f"Bias ratio: {efficiency['bias_ratio']:.4f}%")
        print(f"Reduction factor: {efficiency['reduction_factor']:.1f}x")
        print("=" * 50)


class BiasParameterManager:
    """Manages bias parameter identification and manipulation"""
    
    def __init__(self, config: BitFitConfig):
        self.config = config
        self.bias_patterns = config.get_bias_patterns()
    
    def identify_bias_parameters(self, model: nn.Module) -> Dict[str, List[str]]:
        """Identify bias parameters based on configuration"""
        bias_params = {
            "attention_bias": [],
            "feedforward_bias": [],
            "layer_norm_bias": [],
            "classifier_bias": [],
            "embedding_bias": [],
            "other_bias": []
        }
        
        for name, param in model.named_parameters():
            if 'bias' not in name:
                continue
            
            # Check if parameter should be excluded
            if self._should_exclude_parameter(name):
                continue
            
            # Check if parameter should be included
            if not self._should_include_parameter(name):
                continue
            
            # Categorize bias parameter
            bias_type = self._categorize_bias_parameter(name)
            bias_params[bias_type].append(name)
        
        return bias_params
    
    def _should_exclude_parameter(self, param_name: str) -> bool:
        """Check if parameter should be excluded"""
        for pattern in self.bias_patterns["exclude"]:
            if re.search(pattern, param_name):
                return True
        return False
    
    def _should_include_parameter(self, param_name: str) -> bool:
        """Check if parameter should be included"""
        if not self.bias_patterns["include"]:
            return True  # Include all if no specific patterns
        
        for pattern in self.bias_patterns["include"]:
            if re.search(pattern, param_name):
                return True
        return False
    
    def _categorize_bias_parameter(self, param_name: str) -> str:
        """Categorize bias parameter by type"""
        name_lower = param_name.lower()
        
        # Attention bias patterns
        attention_patterns = [
            r'attention.*bias', r'self_attn.*bias', r'attn.*bias',
            r'query.*bias', r'key.*bias', r'value.*bias', r'out_proj.*bias'
        ]
        for pattern in attention_patterns:
            if re.search(pattern, name_lower):
                return "attention_bias"
        
        # Feedforward bias patterns
        feedforward_patterns = [
            r'intermediate.*bias', r'output.*bias', r'feed_forward.*bias',
            r'ffn.*bias', r'mlp.*bias', r'fc.*bias', r'dense.*bias'
        ]
        for pattern in feedforward_patterns:
            if re.search(pattern, name_lower):
                return "feedforward_bias"
        
        # Layer norm bias patterns
        layernorm_patterns = [
            r'layernorm.*bias', r'layer_norm.*bias', r'norm.*bias',
            r'ln.*bias', r'batch_norm.*bias'
        ]
        for pattern in layernorm_patterns:
            if re.search(pattern, name_lower):
                return "layer_norm_bias"
        
        # Classifier bias patterns
        classifier_patterns = [
            r'classifier.*bias', r'cls.*bias', r'head.*bias',
            r'prediction.*bias', r'score.*bias'
        ]
        for pattern in classifier_patterns:
            if re.search(pattern, name_lower):
                return "classifier_bias"
        
        # Embedding bias patterns
        embedding_patterns = [
            r'embeddings.*bias', r'embed.*bias', r'word_embed.*bias',
            r'position_embed.*bias', r'token_embed.*bias'
        ]
        for pattern in embedding_patterns:
            if re.search(pattern, name_lower):
                return "embedding_bias"
        
        return "other_bias"
    
    def freeze_non_bias_parameters(self, model: nn.Module) -> int:
        """Freeze all non-bias parameters"""
        frozen_count = 0
        
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param.requires_grad = False
                frozen_count += param.numel()
        
        logger.info(f"Frozen {frozen_count:,} non-bias parameters")
        return frozen_count
    
    def setup_bias_training(self, model: nn.Module) -> Dict[str, int]:
        """Setup bias parameters for training based on configuration"""
        bias_params = self.identify_bias_parameters(model)
        trainable_counts = {}
        
        # First, freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Then, enable training for selected bias types
        for bias_type, param_names in bias_params.items():
            if not self._should_train_bias_type(bias_type):
                continue
            
            trainable_count = 0
            for param_name in param_names:
                # Check layer-specific training if configured
                if not self._should_train_layer(param_name):
                    continue
                
                # Get parameter and enable training
                param = dict(model.named_parameters())[param_name]
                param.requires_grad = True
                trainable_count += param.numel()
                
                # Apply initialization if configured
                if self.config.reinitialize_bias:
                    self._initialize_bias_parameter(param)
            
            trainable_counts[bias_type] = trainable_count
            if trainable_count > 0:
                logger.info(f"Enabled training for {trainable_count:,} {bias_type} parameters")
        
        return trainable_counts
    
    def _should_train_bias_type(self, bias_type: str) -> bool:
        """Check if bias type should be trained"""
        type_mapping = {
            "attention_bias": self.config.train_attention_bias,
            "feedforward_bias": self.config.train_feedforward_bias,
            "layer_norm_bias": self.config.train_layer_norm_bias,
            "classifier_bias": self.config.train_classifier_bias,
            "embedding_bias": self.config.train_embedding_bias,
            "other_bias": True  # Train other bias by default
        }
        
        return type_mapping.get(bias_type, False)
    
    def _should_train_layer(self, param_name: str) -> bool:
        """Check if parameter's layer should be trained"""
        if not self.config.selective_bias_training:
            return True
        
        if self.config.bias_layers_to_train is None:
            return True
        
        # Extract layer number from parameter name
        layer_match = re.search(r'layer\.(\d+)', param_name)
        if layer_match:
            layer_num = int(layer_match.group(1))
            return layer_num in self.config.bias_layers_to_train
        
        # If no layer number found, include by default
        return True
    
    def _initialize_bias_parameter(self, param: torch.Tensor):
        """Initialize bias parameter based on configuration"""
        with torch.no_grad():
            if self.config.bias_init_method == "zeros":
                param.zero_()
            elif self.config.bias_init_method == "ones":
                param.fill_(1.0)
            elif self.config.bias_init_method == "normal":
                param.normal_(mean=0.0, std=self.config.bias_init_std)
            elif self.config.bias_init_method == "uniform":
                param.uniform_(-self.config.bias_init_std, self.config.bias_init_std)
    
    def get_bias_parameter_groups(self, model: nn.Module) -> List[Dict]:
        """Get parameter groups for optimizer with different learning rates"""
        bias_params = self.identify_bias_parameters(model)
        learning_rates = self.config.get_bias_learning_rates()
        
        if not self.config.use_different_lr_for_bias_types:
            # Single learning rate for all bias parameters
            bias_param_list = []
            for param_names in bias_params.values():
                for param_name in param_names:
                    param = dict(model.named_parameters())[param_name]
                    if param.requires_grad:
                        bias_param_list.append(param)
            
            return [{"params": bias_param_list, "lr": learning_rates["default"]}]
        
        # Different learning rates for different bias types
        param_groups = []
        
        for bias_type, param_names in bias_params.items():
            if not param_names:
                continue
            
            lr_key = bias_type
            if lr_key not in learning_rates:
                lr_key = "default"
            
            if lr_key not in learning_rates:
                continue
            
            param_list = []
            for param_name in param_names:
                param = dict(model.named_parameters())[param_name]
                if param.requires_grad:
                    param_list.append(param)
            
            if param_list:
                param_groups.append({
                    "params": param_list,
                    "lr": learning_rates[lr_key],
                    "name": bias_type
                })
        
        return param_groups
