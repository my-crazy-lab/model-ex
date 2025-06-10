"""
Fusion model that combines base model with adapter fusion
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
import logging
from transformers import AutoModel, AutoModelForSequenceClassification

from ..config.model_config import ModelConfig
from ..config.fusion_config import FusionConfig
from .adapter_manager import AdapterManager
from .fusion_layer import (
    FusionLayer, AttentionFusion, WeightedFusion, 
    GatingFusion, HierarchicalFusion
)

logger = logging.getLogger(__name__)


class FusionModel(nn.Module):
    """
    Model that combines a base transformer with adapter fusion
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        fusion_config: FusionConfig,
        adapter_manager: Optional[AdapterManager] = None
    ):
        super().__init__()
        
        self.model_config = model_config
        self.fusion_config = fusion_config
        
        # Load base model
        self.base_model = self._load_base_model()
        
        # Get hidden size from base model
        self.hidden_size = self._get_hidden_size()
        
        # Setup adapter manager
        if adapter_manager is not None:
            self.adapter_manager = adapter_manager
        else:
            # Create empty adapter manager
            self.adapter_manager = AdapterManager(
                hidden_size=self.hidden_size,
                adapter_configs={},
                freeze_adapters=fusion_config.freeze_adapters_during_fusion
            )
        
        # Create fusion layer
        self.fusion_layer = self._create_fusion_layer()
        
        # Freeze base model if specified
        if hasattr(fusion_config, 'freeze_base_model') and fusion_config.freeze_base_model:
            self._freeze_base_model()
        
        logger.info(f"Created FusionModel with {len(self.adapter_manager.adapter_names)} adapters")
    
    def _load_base_model(self):
        """Load the base transformer model"""
        model_kwargs = self.model_config.get_model_kwargs()
        
        if self.model_config.task_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name_or_path,
                num_labels=self.model_config.num_labels,
                **model_kwargs
            )
        else:
            model = AutoModel.from_pretrained(
                self.model_config.model_name_or_path,
                **model_kwargs
            )
        
        return model
    
    def _get_hidden_size(self) -> int:
        """Get hidden size from base model"""
        if hasattr(self.base_model.config, 'hidden_size'):
            return self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, 'd_model'):
            return self.base_model.config.d_model
        else:
            raise ValueError("Could not determine hidden size from model config")
    
    def _create_fusion_layer(self) -> FusionLayer:
        """Create fusion layer based on configuration"""
        num_adapters = len(self.adapter_manager.adapter_names)
        
        if num_adapters == 0:
            logger.warning("No adapters found, creating dummy fusion layer")
            num_adapters = 1
        
        if self.fusion_config.fusion_method.value == "attention":
            return AttentionFusion(
                hidden_size=self.hidden_size,
                num_adapters=num_adapters,
                num_attention_heads=self.fusion_config.num_attention_heads,
                dropout=self.fusion_config.fusion_dropout,
                temperature=self.fusion_config.attention_temperature,
                use_bias=self.fusion_config.use_attention_bias
            )
        
        elif self.fusion_config.fusion_method.value == "weighted":
            return WeightedFusion(
                hidden_size=self.hidden_size,
                num_adapters=num_adapters,
                dropout=self.fusion_config.fusion_dropout,
                learnable_weights=self.fusion_config.learnable_weights,
                weight_initialization=self.fusion_config.weight_initialization,
                weight_constraint=self.fusion_config.weight_constraint
            )
        
        elif self.fusion_config.fusion_method.value == "gating":
            return GatingFusion(
                hidden_size=self.hidden_size,
                num_adapters=num_adapters,
                dropout=self.fusion_config.fusion_dropout,
                gate_activation=self.fusion_config.gate_activation,
                gate_hidden_size=self.fusion_config.gate_hidden_size,
                use_bias=self.fusion_config.gate_bias
            )
        
        elif self.fusion_config.fusion_method.value == "hierarchical":
            return HierarchicalFusion(
                hidden_size=self.hidden_size,
                num_adapters=num_adapters,
                fusion_layers=self.fusion_config.fusion_layers,
                dropout=self.fusion_config.fusion_dropout,
                layer_fusion_method=self.fusion_config.layer_fusion_method
            )
        
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_config.fusion_method}")
    
    def _freeze_base_model(self):
        """Freeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Frozen base model parameters")
    
    def _unfreeze_base_model(self):
        """Unfreeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = True
        logger.info("Unfrozen base model parameters")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        adapter_names: Optional[List[str]] = None,
        return_adapter_outputs: bool = False,
        **kwargs
    ):
        """
        Forward pass through fusion model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Labels for training
            adapter_names: List of adapters to use
            return_adapter_outputs: Whether to return individual adapter outputs
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with fused adapter representations
        """
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get hidden states (usually from last layer)
        if hasattr(base_outputs, 'hidden_states'):
            hidden_states = base_outputs.hidden_states[-1]
        elif hasattr(base_outputs, 'last_hidden_state'):
            hidden_states = base_outputs.last_hidden_state
        else:
            raise ValueError("Could not extract hidden states from base model outputs")
        
        # Get adapter outputs
        adapter_outputs_dict = self.adapter_manager(
            hidden_states=hidden_states,
            adapter_names=adapter_names
        )
        
        # Convert to list for fusion
        if adapter_names is None:
            adapter_names = self.adapter_manager.adapter_names
        
        adapter_outputs = [adapter_outputs_dict[name] for name in adapter_names if name in adapter_outputs_dict]
        
        if not adapter_outputs:
            # No adapters available, return base model outputs
            logger.warning("No adapter outputs available, returning base model outputs")
            if labels is not None:
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                    **kwargs
                )
            else:
                return base_outputs
        
        # Apply fusion
        if self.fusion_config.fusion_method.value == "hierarchical":
            # For hierarchical fusion, we need layer-wise outputs
            layer_adapter_outputs = {}
            for layer_idx in self.fusion_config.fusion_layers:
                if layer_idx < len(base_outputs.hidden_states):
                    layer_hidden_states = base_outputs.hidden_states[layer_idx]
                    layer_outputs_dict = self.adapter_manager(
                        hidden_states=layer_hidden_states,
                        adapter_names=adapter_names
                    )
                    layer_adapter_outputs[layer_idx] = [
                        layer_outputs_dict[name] for name in adapter_names 
                        if name in layer_outputs_dict
                    ]
            
            fused_output = self.fusion_layer(
                layer_adapter_outputs,
                attention_mask=attention_mask
            )
        else:
            fused_output = self.fusion_layer(
                adapter_outputs,
                attention_mask=attention_mask
            )
        
        # Create output object
        if self.model_config.task_type == "classification":
            # For classification, we need to pass through classifier
            if hasattr(self.base_model, 'classifier'):
                # Use pooled representation for classification
                if hasattr(self.base_model, 'pooler') and self.base_model.pooler is not None:
                    pooled_output = self.base_model.pooler(fused_output)
                else:
                    # Use [CLS] token representation
                    pooled_output = fused_output[:, 0]
                
                logits = self.base_model.classifier(pooled_output)
            else:
                # Model doesn't have separate classifier
                logits = fused_output
            
            # Calculate loss if labels provided
            loss = None
            if labels is not None:
                if self.model_config.problem_type == "regression":
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                elif self.model_config.problem_type == "single_label_classification":
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.model_config.num_labels), labels.view(-1))
                elif self.model_config.problem_type == "multi_label_classification":
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            
            # Create output similar to transformers models
            from transformers.modeling_outputs import SequenceClassifierOutput
            output = SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=base_outputs.hidden_states if hasattr(base_outputs, 'hidden_states') else None,
                attentions=base_outputs.attentions if hasattr(base_outputs, 'attentions') else None
            )
        else:
            # For other tasks, return fused hidden states
            output = type(base_outputs)(
                last_hidden_state=fused_output,
                hidden_states=base_outputs.hidden_states if hasattr(base_outputs, 'hidden_states') else None,
                attentions=base_outputs.attentions if hasattr(base_outputs, 'attentions') else None
            )
        
        # Add adapter outputs if requested
        if return_adapter_outputs:
            output.adapter_outputs = adapter_outputs_dict
        
        return output
    
    def add_adapter_from_path(self, adapter_name: str, adapter_path: str):
        """Add adapter from saved path"""
        self.adapter_manager.load_adapter_from_path(adapter_name, adapter_path)
        
        # Recreate fusion layer with new number of adapters
        self.fusion_layer = self._create_fusion_layer()
        
        logger.info(f"Added adapter '{adapter_name}' and recreated fusion layer")
    
    def remove_adapter(self, adapter_name: str):
        """Remove adapter"""
        self.adapter_manager.remove_adapter(adapter_name)
        
        # Recreate fusion layer with new number of adapters
        self.fusion_layer = self._create_fusion_layer()
        
        logger.info(f"Removed adapter '{adapter_name}' and recreated fusion layer")
    
    def get_fusion_parameters(self):
        """Get fusion layer parameters"""
        return list(self.fusion_layer.parameters())
    
    def get_adapter_parameters(self):
        """Get adapter parameters"""
        return [p for adapter in self.adapter_manager.adapters.values() for p in adapter.parameters()]
    
    def freeze_adapters(self):
        """Freeze all adapters"""
        self.adapter_manager.freeze_all_adapters()
    
    def unfreeze_adapters(self):
        """Unfreeze all adapters"""
        self.adapter_manager.unfreeze_all_adapters()
    
    def save_fusion(self, save_path: str):
        """Save fusion layer"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save fusion layer
        torch.save(self.fusion_layer.state_dict(), os.path.join(save_path, "fusion_layer.bin"))
        
        # Save fusion config
        import json
        with open(os.path.join(save_path, "fusion_config.json"), 'w') as f:
            json.dump(self.fusion_config.to_dict(), f, indent=2)
        
        logger.info(f"Saved fusion layer to {save_path}")
    
    def load_fusion(self, load_path: str):
        """Load fusion layer"""
        import os
        
        # Load fusion layer weights
        fusion_weights_path = os.path.join(load_path, "fusion_layer.bin")
        if os.path.exists(fusion_weights_path):
            self.fusion_layer.load_state_dict(torch.load(fusion_weights_path, map_location="cpu"))
            logger.info(f"Loaded fusion layer from {load_path}")
        else:
            raise ValueError(f"No fusion weights found at {fusion_weights_path}")
    
    def print_model_info(self):
        """Print model information"""
        print("Fusion Model Information:")
        print(f"  Base model: {self.model_config.model_name_or_path}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Fusion method: {self.fusion_config.fusion_method.value}")
        print(f"  Task type: {self.model_config.task_type}")
        
        # Print adapter info
        self.adapter_manager.print_adapter_info()
        
        # Print fusion layer info
        fusion_params = sum(p.numel() for p in self.fusion_layer.parameters())
        print(f"  Fusion layer parameters: {fusion_params:,}")
        
        # Print total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable percentage: {trainable_params/total_params*100:.2f}%")
