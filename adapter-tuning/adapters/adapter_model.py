"""
Model wrapper with adapter integration
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedModel
)
import logging

from .adapter_layer import BottleneckAdapter, ParallelAdapter, MultiAdapter
from ..config.model_config import ModelConfig
from ..config.adapter_config import AdapterConfig

logger = logging.getLogger(__name__)


class AdapterModel(nn.Module):
    """
    Model wrapper that adds adapters to pre-trained transformers
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        adapter_config: AdapterConfig,
        model: Optional[PreTrainedModel] = None
    ):
        super().__init__()
        self.model_config = model_config
        self.adapter_config = adapter_config
        
        # Load base model if not provided
        if model is None:
            self.base_model = self._load_base_model()
        else:
            self.base_model = model
        
        # Add adapters to the model
        self._add_adapters()
        
        # Freeze base model parameters if specified
        if adapter_config.freeze_base_model:
            self._freeze_base_model()
        
        # Store adapter information
        self.adapter_info = self._get_adapter_info()
        
        logger.info(f"AdapterModel initialized with {self.adapter_info['total_adapter_params']:,} adapter parameters")
    
    def _load_base_model(self) -> PreTrainedModel:
        """Load the base pre-trained model"""
        model_kwargs = self.model_config.get_model_kwargs()
        
        if self.model_config.task_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name_or_path,
                num_labels=self.model_config.num_labels,
                **model_kwargs
            )
        elif self.model_config.task_type == "ner":
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_config.model_name_or_path,
                num_labels=self.model_config.num_labels,
                **model_kwargs
            )
        elif self.model_config.task_type == "qa":
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
    
    def _add_adapters(self):
        """Add adapters to the model"""
        # Get model architecture info
        if hasattr(self.base_model, 'bert'):
            encoder = self.base_model.bert.encoder
        elif hasattr(self.base_model, 'roberta'):
            encoder = self.base_model.roberta.encoder
        elif hasattr(self.base_model, 'distilbert'):
            encoder = self.base_model.distilbert.transformer
        elif hasattr(self.base_model, 'encoder'):
            encoder = self.base_model.encoder
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.base_model)}")
        
        # Add adapters to transformer layers
        for layer_idx, layer in enumerate(encoder.layer):
            if self.adapter_config.should_add_adapter(layer_idx):
                self._add_adapter_to_layer(layer, layer_idx)
    
    def _add_adapter_to_layer(self, layer, layer_idx: int):
        """Add adapter to a specific transformer layer"""
        # Get hidden size from the layer
        if hasattr(layer, 'output') and hasattr(layer.output, 'dense'):
            hidden_size = layer.output.dense.out_features
        elif hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense'):
            hidden_size = layer.intermediate.dense.out_features
        else:
            # Fallback: try to infer from attention
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
                hidden_size = layer.attention.self.query.in_features
            else:
                raise ValueError(f"Cannot determine hidden size for layer {layer_idx}")
        
        # Calculate adapter size
        adapter_size = self.adapter_config.get_adapter_size(hidden_size)
        
        # Create adapters based on configuration
        if self.adapter_config.adapter_location in ["attention", "both"]:
            attention_adapter = self._create_adapter(hidden_size, adapter_size, f"attention_{layer_idx}")
            layer.attention_adapter = attention_adapter
        
        if self.adapter_config.adapter_location in ["feedforward", "both"]:
            feedforward_adapter = self._create_adapter(hidden_size, adapter_size, f"feedforward_{layer_idx}")
            layer.feedforward_adapter = feedforward_adapter
        
        # Modify layer forward method to include adapters
        self._modify_layer_forward(layer)
    
    def _create_adapter(self, hidden_size: int, adapter_size: int, name: str):
        """Create an adapter instance"""
        if self.adapter_config.adapter_type == "bottleneck":
            return BottleneckAdapter(
                input_size=hidden_size,
                adapter_size=adapter_size,
                dropout=self.adapter_config.adapter_dropout,
                activation=self.adapter_config.adapter_activation,
                use_residual=self.adapter_config.use_residual,
                use_layer_norm=self.adapter_config.use_layer_norm,
                init_range=self.adapter_config.adapter_init_range,
                scaling=self.adapter_config.adapter_scaling
            )
        elif self.adapter_config.adapter_type == "parallel":
            return ParallelAdapter(
                input_size=hidden_size,
                adapter_size=adapter_size,
                dropout=self.adapter_config.adapter_dropout,
                activation=self.adapter_config.adapter_activation,
                use_residual=self.adapter_config.use_residual,
                use_layer_norm=self.adapter_config.use_layer_norm,
                init_range=self.adapter_config.adapter_init_range,
                scaling=self.adapter_config.adapter_scaling
            )
        else:
            raise ValueError(f"Unsupported adapter type: {self.adapter_config.adapter_type}")
    
    def _modify_layer_forward(self, layer):
        """Modify layer forward method to include adapter calls"""
        original_forward = layer.forward
        
        def forward_with_adapters(hidden_states, attention_mask=None, **kwargs):
            # Original layer forward pass
            outputs = original_forward(hidden_states, attention_mask=attention_mask, **kwargs)
            
            # Extract hidden states (first element of outputs)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs
            
            # Apply attention adapter if present
            if hasattr(layer, 'attention_adapter'):
                hidden_states = layer.attention_adapter(hidden_states)
            
            # Apply feedforward adapter if present
            if hasattr(layer, 'feedforward_adapter'):
                hidden_states = layer.feedforward_adapter(hidden_states)
            
            # Return in the same format as original outputs
            if isinstance(outputs, tuple):
                return (hidden_states,) + outputs[1:]
            else:
                return hidden_states
        
        layer.forward = forward_with_adapters
    
    def _freeze_base_model(self):
        """Freeze base model parameters"""
        for name, param in self.base_model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False
        
        logger.info("Base model parameters frozen")
    
    def _get_adapter_info(self) -> Dict[str, Any]:
        """Get information about adapters"""
        total_params = sum(p.numel() for p in self.parameters())
        adapter_params = sum(p.numel() for n, p in self.named_parameters() if "adapter" in n)
        base_params = total_params - adapter_params
        
        return {
            "total_params": total_params,
            "base_params": base_params,
            "total_adapter_params": adapter_params,
            "adapter_percentage": (adapter_params / total_params) * 100,
            "adapter_config": self.adapter_config.to_dict()
        }
    
    def forward(self, **kwargs):
        """Forward pass through the model"""
        return self.base_model(**kwargs)
    
    def save_adapters(self, save_directory: str):
        """Save only adapter parameters"""
        import os
        import torch
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save adapter state dict
        adapter_state_dict = {}
        for name, param in self.named_parameters():
            if "adapter" in name:
                adapter_state_dict[name] = param
        
        torch.save(adapter_state_dict, os.path.join(save_directory, "adapter_model.bin"))
        
        # Save adapter config
        import json
        with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
            json.dump(self.adapter_config.to_dict(), f, indent=2)
        
        logger.info(f"Adapters saved to {save_directory}")
    
    def load_adapters(self, load_directory: str):
        """Load adapter parameters"""
        import os
        import torch
        import json
        
        # Load adapter config
        config_path = os.path.join(load_directory, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                adapter_config_dict = json.load(f)
            self.adapter_config = AdapterConfig.from_dict(adapter_config_dict)
        
        # Load adapter state dict
        adapter_path = os.path.join(load_directory, "adapter_model.bin")
        if os.path.exists(adapter_path):
            adapter_state_dict = torch.load(adapter_path, map_location="cpu")
            
            # Load only adapter parameters
            model_state_dict = self.state_dict()
            for name, param in adapter_state_dict.items():
                if name in model_state_dict:
                    model_state_dict[name].copy_(param)
            
            logger.info(f"Adapters loaded from {load_directory}")
        else:
            raise FileNotFoundError(f"Adapter model not found at {adapter_path}")
    
    def get_adapter_parameters(self):
        """Get adapter parameters for optimization"""
        return [p for n, p in self.named_parameters() if "adapter" in n and p.requires_grad]
    
    def get_base_parameters(self):
        """Get base model parameters"""
        return [p for n, p in self.named_parameters() if "adapter" not in n and p.requires_grad]
    
    def print_adapter_info(self):
        """Print detailed adapter information"""
        info = self.adapter_info
        print(f"Adapter Model Information:")
        print(f"  Total parameters: {info['total_params']:,}")
        print(f"  Base parameters: {info['base_params']:,}")
        print(f"  Adapter parameters: {info['total_adapter_params']:,}")
        print(f"  Adapter percentage: {info['adapter_percentage']:.2f}%")
        print(f"  Adapter config: {info['adapter_config']}")


def add_adapters_to_model(
    model: PreTrainedModel,
    adapter_config: AdapterConfig
) -> AdapterModel:
    """
    Convenience function to add adapters to an existing model
    
    Args:
        model: Pre-trained model
        adapter_config: Adapter configuration
        
    Returns:
        Model with adapters added
    """
    # Create a dummy model config
    model_config = ModelConfig(
        model_name_or_path=model.config.name_or_path or "unknown",
        num_labels=getattr(model.config, 'num_labels', 2)
    )
    
    return AdapterModel(model_config, adapter_config, model)
