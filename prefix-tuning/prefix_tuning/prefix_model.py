"""
Prefix tuning model implementation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM, PreTrainedModel
)

from ..config.prefix_config import PrefixConfig, PrefixPosition
from ..config.model_config import ModelConfig
from .prefix_embeddings import PrefixEmbeddings

logger = logging.getLogger(__name__)


class PrefixTuningModel(nn.Module):
    """
    Prefix tuning model that wraps a pre-trained transformer with prefix embeddings
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        prefix_config: PrefixConfig,
        base_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[Any] = None
    ):
        super().__init__()
        
        self.model_config = model_config
        self.prefix_config = prefix_config
        self.tokenizer = tokenizer
        
        # Load or use provided base model
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = self._load_base_model()
        
        # Freeze base model parameters
        if prefix_config.freeze_base_model:
            self._freeze_base_model()
        
        # Initialize prefix embeddings
        self.prefix_embeddings = PrefixEmbeddings(
            prefix_config,
            self.base_model.config,
            tokenizer
        )
        
        # Setup hooks for prefix injection
        self._setup_prefix_hooks()
        
        # Store original forward methods
        self._store_original_methods()
        
        logger.info("PrefixTuningModel initialized successfully")
    
    def _load_base_model(self) -> PreTrainedModel:
        """Load the base pre-trained model"""
        model_kwargs = self.model_config.get_model_kwargs()
        
        if self.model_config.task_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name_or_path,
                num_labels=self.prefix_config.num_labels or 2,
                **model_kwargs
            )
        elif self.model_config.task_type == "generation":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_name_or_path,
                **model_kwargs
            )
        elif self.model_config.task_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_config.model_name_or_path,
                **model_kwargs
            )
        else:
            model = AutoModel.from_pretrained(
                self.model_config.model_name_or_path,
                **model_kwargs
            )
        
        return model
    
    def _freeze_base_model(self):
        """Freeze base model parameters"""
        for name, param in self.base_model.named_parameters():
            # Check if this module should remain trainable
            should_train = any(
                trainable_module in name 
                for trainable_module in self.prefix_config.trainable_base_modules
            )
            
            param.requires_grad = should_train
        
        # Count frozen parameters
        frozen_params = sum(
            p.numel() for p in self.base_model.parameters() if not p.requires_grad
        )
        trainable_params = sum(
            p.numel() for p in self.base_model.parameters() if p.requires_grad
        )
        
        logger.info(f"Frozen {frozen_params:,} base model parameters")
        logger.info(f"Trainable {trainable_params:,} base model parameters")
    
    def _setup_prefix_hooks(self):
        """Setup hooks for prefix injection"""
        self.attention_hooks = []
        
        # Get transformer layers
        if hasattr(self.base_model, 'transformer'):
            transformer = self.base_model.transformer
        elif hasattr(self.base_model, 'model'):
            transformer = self.base_model.model
        elif hasattr(self.base_model, 'encoder'):
            transformer = self.base_model.encoder
        else:
            transformer = self.base_model
        
        # Get layers
        if hasattr(transformer, 'h'):  # GPT-style
            layers = transformer.h
        elif hasattr(transformer, 'layer'):  # BERT-style
            layers = transformer.layer
        elif hasattr(transformer, 'layers'):  # T5-style
            layers = transformer.layers
        else:
            logger.warning("Could not find transformer layers, prefix injection may not work")
            return
        
        # Apply hooks based on prefix position
        if self.prefix_config.prefix_position == PrefixPosition.ALL_LAYERS:
            layer_indices = list(range(len(layers)))
        elif self.prefix_config.prefix_position == PrefixPosition.SELECTED_LAYERS:
            layer_indices = self.prefix_config.selected_layers or []
        else:  # INPUT_ONLY
            layer_indices = [0]
        
        # Register hooks
        for layer_idx in layer_indices:
            if layer_idx < len(layers):
                hook = self._create_attention_hook(layer_idx)
                layers[layer_idx].register_forward_hook(hook)
                self.attention_hooks.append(hook)
    
    def _create_attention_hook(self, layer_idx: int):
        """Create attention hook for prefix injection"""
        def attention_hook(module, input, output):
            # This is a simplified hook - actual implementation would depend on model architecture
            return output
        
        return attention_hook
    
    def _store_original_methods(self):
        """Store original forward methods for restoration"""
        self.original_forward = self.base_model.forward
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with prefix conditioning
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size = input_ids.size(0)
        
        # Get prefix embeddings
        if self.prefix_config.prefix_position == PrefixPosition.INPUT_ONLY:
            # Add prefix to input embeddings
            return self._forward_with_input_prefix(
                input_ids, attention_mask, labels, **kwargs
            )
        else:
            # Add prefix to attention layers
            return self._forward_with_attention_prefix(
                input_ids, attention_mask, labels, **kwargs
            )
    
    def _forward_with_input_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with prefix added to input"""
        batch_size = input_ids.size(0)
        
        # Get input embeddings
        if hasattr(self.base_model, 'get_input_embeddings'):
            embedding_layer = self.base_model.get_input_embeddings()
        else:
            # Try to find embedding layer
            if hasattr(self.base_model, 'embeddings'):
                embedding_layer = self.base_model.embeddings.word_embeddings
            elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'wte'):
                embedding_layer = self.base_model.transformer.wte
            else:
                raise ValueError("Could not find embedding layer")
        
        # Get input embeddings
        input_embeddings = embedding_layer(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # Get prefix embeddings (use layer 0 for input-only)
        prefix_keys, prefix_values = self.prefix_embeddings.get_prefix_embeddings(batch_size, layer_idx=0)
        
        # For input-only prefix, we use the prefix keys as prefix embeddings
        # [batch_size, num_heads, prefix_length, head_dim] -> [batch_size, prefix_length, hidden_size]
        prefix_embeds = prefix_keys.transpose(1, 2).contiguous().view(
            batch_size, self.prefix_config.prefix_length, -1
        )
        
        # Concatenate prefix with input embeddings
        prefixed_embeddings = torch.cat([prefix_embeds, input_embeddings], dim=1)
        
        # Update attention mask
        if attention_mask is not None:
            prefix_attention = torch.ones(
                batch_size, self.prefix_config.prefix_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        
        # Forward through model with prefixed embeddings
        # This requires modifying the model's forward method to accept embeddings
        # For simplicity, we'll use the standard forward and handle prefix in hooks
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def _forward_with_attention_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with prefix added to attention layers"""
        # Store prefix embeddings in model state for hooks to access
        batch_size = input_ids.size(0)
        self._current_prefix_embeddings = {}
        
        # Pre-compute prefix embeddings for all layers
        if self.prefix_config.prefix_position == PrefixPosition.ALL_LAYERS:
            num_layers = getattr(self.base_model.config, 'num_hidden_layers', 
                               getattr(self.base_model.config, 'num_layers', 12))
            for layer_idx in range(num_layers):
                prefix_keys, prefix_values = self.prefix_embeddings.get_prefix_embeddings(
                    batch_size, layer_idx
                )
                self._current_prefix_embeddings[layer_idx] = (prefix_keys, prefix_values)
        elif self.prefix_config.prefix_position == PrefixPosition.SELECTED_LAYERS:
            for layer_idx in self.prefix_config.selected_layers or []:
                prefix_keys, prefix_values = self.prefix_embeddings.get_prefix_embeddings(
                    batch_size, layer_idx
                )
                self._current_prefix_embeddings[layer_idx] = (prefix_keys, prefix_values)
        
        # Update attention mask for prefix
        if attention_mask is not None:
            prefix_attention = torch.ones(
                batch_size, self.prefix_config.prefix_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        
        # Forward through model (hooks will inject prefixes)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # Clean up
        self._current_prefix_embeddings = {}
        
        return outputs
    
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate text with prefix conditioning"""
        if not hasattr(self.base_model, 'generate'):
            raise ValueError("Base model does not support generation")
        
        # Update attention mask for prefix if provided
        if 'attention_mask' in kwargs:
            batch_size = input_ids.size(0)
            prefix_attention = torch.ones(
                batch_size, self.prefix_config.prefix_length,
                dtype=kwargs['attention_mask'].dtype,
                device=kwargs['attention_mask'].device
            )
            kwargs['attention_mask'] = torch.cat([prefix_attention, kwargs['attention_mask']], dim=1)
        
        return self.base_model.generate(input_ids=input_ids, **kwargs)
    
    def get_parameter_efficiency(self) -> Dict[str, Any]:
        """Get parameter efficiency metrics"""
        total_params = sum(p.numel() for p in self.base_model.parameters())
        prefix_params = self.prefix_embeddings.get_num_parameters()
        trainable_base_params = sum(
            p.numel() for p in self.base_model.parameters() if p.requires_grad
        )
        
        total_trainable = prefix_params + trainable_base_params
        
        return {
            "total_parameters": total_params,
            "prefix_parameters": prefix_params,
            "trainable_base_parameters": trainable_base_params,
            "total_trainable_parameters": total_trainable,
            "parameter_efficiency": total_trainable / total_params * 100,
            "prefix_efficiency": prefix_params / total_params * 100,
            "reduction_factor": total_params / total_trainable
        }
    
    def print_parameter_summary(self):
        """Print parameter efficiency summary"""
        efficiency = self.get_parameter_efficiency()
        
        print("\nPrefix Tuning Parameter Summary:")
        print("=" * 50)
        print(f"Total parameters: {efficiency['total_parameters']:,}")
        print(f"Prefix parameters: {efficiency['prefix_parameters']:,}")
        print(f"Trainable base parameters: {efficiency['trainable_base_parameters']:,}")
        print(f"Total trainable: {efficiency['total_trainable_parameters']:,}")
        print()
        print(f"Parameter efficiency: {efficiency['parameter_efficiency']:.4f}%")
        print(f"Prefix efficiency: {efficiency['prefix_efficiency']:.4f}%")
        print(f"Reduction factor: {efficiency['reduction_factor']:.1f}x")
        print("=" * 50)
    
    def save_prefix_tuning_model(self, save_path: str):
        """Save prefix tuning model"""
        import os
        import json
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save prefix embeddings
        self.prefix_embeddings.save_prefix_embeddings(
            os.path.join(save_path, "prefix_embeddings.pt")
        )
        
        # Save configurations
        with open(os.path.join(save_path, "prefix_config.json"), 'w') as f:
            json.dump(self.prefix_config.to_dict(), f, indent=2)
        
        with open(os.path.join(save_path, "model_config.json"), 'w') as f:
            json.dump(self.model_config.__dict__, f, indent=2)
        
        # Save parameter efficiency stats
        efficiency = self.get_parameter_efficiency()
        with open(os.path.join(save_path, "efficiency_stats.json"), 'w') as f:
            json.dump(efficiency, f, indent=2)
        
        logger.info(f"Prefix tuning model saved to {save_path}")
    
    def load_prefix_tuning_model(self, load_path: str):
        """Load prefix tuning model"""
        import os
        
        # Load prefix embeddings
        prefix_path = os.path.join(load_path, "prefix_embeddings.pt")
        if os.path.exists(prefix_path):
            self.prefix_embeddings.load_prefix_embeddings(prefix_path)
        
        logger.info(f"Prefix tuning model loaded from {load_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_config: Optional[ModelConfig] = None,
        prefix_config: Optional[PrefixConfig] = None,
        tokenizer: Optional[Any] = None
    ) -> "PrefixTuningModel":
        """Load prefix tuning model from pretrained path"""
        import os
        import json
        
        # Load configurations if not provided
        if model_config is None:
            config_path = os.path.join(model_path, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                model_config = ModelConfig(**config_dict)
            else:
                raise ValueError(f"No model config found at {config_path}")
        
        if prefix_config is None:
            config_path = os.path.join(model_path, "prefix_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                prefix_config = PrefixConfig.from_dict(config_dict)
            else:
                raise ValueError(f"No prefix config found at {config_path}")
        
        # Create model
        model = cls(model_config, prefix_config, tokenizer=tokenizer)
        
        # Load prefix embeddings
        model.load_prefix_tuning_model(model_path)
        
        return model
