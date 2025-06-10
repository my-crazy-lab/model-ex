"""
Core adapter layer implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class AdapterLayer(nn.Module):
    """Base class for adapter layers"""
    
    def __init__(
        self,
        input_size: int,
        adapter_size: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_residual: bool = True,
        use_layer_norm: bool = False,
        init_range: float = 1e-3,
        scaling: float = 1.0
    ):
        super().__init__()
        self.input_size = input_size
        self.adapter_size = adapter_size
        self.dropout = dropout
        self.activation = activation
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.scaling = scaling
        
        # Activation function
        self.activation_fn = self._get_activation_fn(activation)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Layer normalization (optional)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_size)
        
        # Initialize weights
        self._init_weights(init_range)
    
    def _get_activation_fn(self, activation: str):
        """Get activation function"""
        activation_map = {
            "relu": F.relu,
            "gelu": F.gelu,
            "swish": F.silu,
            "silu": F.silu,
            "tanh": torch.tanh,
        }
        
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        
        return activation_map[activation]
    
    def _init_weights(self, init_range: float):
        """Initialize adapter weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=init_range)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError


class BottleneckAdapter(AdapterLayer):
    """
    Standard bottleneck adapter implementation
    
    Architecture:
    Input → Down-projection → Activation → Dropout → Up-projection → Scaling → Residual
    """
    
    def __init__(
        self,
        input_size: int,
        adapter_size: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_residual: bool = True,
        use_layer_norm: bool = False,
        init_range: float = 1e-3,
        scaling: float = 1.0
    ):
        super().__init__(
            input_size, adapter_size, dropout, activation,
            use_residual, use_layer_norm, init_range, scaling
        )
        
        # Down-projection (input_size -> adapter_size)
        self.down_project = nn.Linear(input_size, adapter_size, bias=True)
        
        # Up-projection (adapter_size -> input_size)
        self.up_project = nn.Linear(adapter_size, input_size, bias=True)
        
        # Initialize up-projection to zero for stable training
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bottleneck adapter
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of same shape as input
        """
        # Store original input for residual connection
        residual = hidden_states
        
        # Optional layer normalization
        if self.use_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        
        # Down-projection
        hidden_states = self.down_project(hidden_states)
        
        # Activation
        hidden_states = self.activation_fn(hidden_states)
        
        # Dropout
        hidden_states = self.dropout_layer(hidden_states)
        
        # Up-projection
        hidden_states = self.up_project(hidden_states)
        
        # Scaling
        hidden_states = hidden_states * self.scaling
        
        # Residual connection
        if self.use_residual:
            hidden_states = hidden_states + residual
        
        return hidden_states
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get adapter information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "type": "bottleneck",
            "input_size": self.input_size,
            "adapter_size": self.adapter_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "compression_ratio": self.input_size / self.adapter_size,
            "activation": self.activation,
            "dropout": self.dropout,
            "scaling": self.scaling
        }


class ParallelAdapter(AdapterLayer):
    """
    Parallel adapter implementation
    
    Architecture:
    Input → [Original Path, Adapter Path] → Combine → Output
    """
    
    def __init__(
        self,
        input_size: int,
        adapter_size: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_residual: bool = True,
        use_layer_norm: bool = False,
        init_range: float = 1e-3,
        scaling: float = 1.0,
        combination_method: str = "add"  # add, concat, weighted
    ):
        super().__init__(
            input_size, adapter_size, dropout, activation,
            use_residual, use_layer_norm, init_range, scaling
        )
        
        self.combination_method = combination_method
        
        # Adapter path
        self.adapter_linear = nn.Linear(input_size, adapter_size, bias=True)
        self.adapter_output = nn.Linear(adapter_size, input_size, bias=True)
        
        # Combination weights (for weighted combination)
        if combination_method == "weighted":
            self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # Initialize adapter output to zero
        nn.init.zeros_(self.adapter_output.weight)
        nn.init.zeros_(self.adapter_output.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through parallel adapter"""
        # Original path
        original_output = hidden_states
        
        # Adapter path
        adapter_hidden = self.adapter_linear(hidden_states)
        adapter_hidden = self.activation_fn(adapter_hidden)
        adapter_hidden = self.dropout_layer(adapter_hidden)
        adapter_output = self.adapter_output(adapter_hidden)
        adapter_output = adapter_output * self.scaling
        
        # Combine paths
        if self.combination_method == "add":
            output = original_output + adapter_output
        elif self.combination_method == "weighted":
            weights = F.softmax(self.combination_weights, dim=0)
            output = weights[0] * original_output + weights[1] * adapter_output
        else:
            raise ValueError(f"Unsupported combination method: {self.combination_method}")
        
        return output


class MultiAdapter(nn.Module):
    """
    Multiple adapters that can be used together
    """
    
    def __init__(
        self,
        input_size: int,
        adapter_configs: Dict[str, Dict[str, Any]],
        fusion_method: str = "attention"  # attention, average, weighted
    ):
        super().__init__()
        self.input_size = input_size
        self.fusion_method = fusion_method
        
        # Create individual adapters
        self.adapters = nn.ModuleDict()
        for name, config in adapter_configs.items():
            adapter_type = config.get("type", "bottleneck")
            if adapter_type == "bottleneck":
                self.adapters[name] = BottleneckAdapter(input_size, **config)
            elif adapter_type == "parallel":
                self.adapters[name] = ParallelAdapter(input_size, **config)
            else:
                raise ValueError(f"Unsupported adapter type: {adapter_type}")
        
        # Fusion mechanism
        if fusion_method == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=input_size,
                num_heads=8,
                batch_first=True
            )
        elif fusion_method == "weighted":
            num_adapters = len(adapter_configs)
            self.fusion_weights = nn.Parameter(torch.ones(num_adapters) / num_adapters)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        adapter_names: Optional[list] = None
    ) -> torch.Tensor:
        """Forward pass through multiple adapters"""
        if adapter_names is None:
            adapter_names = list(self.adapters.keys())
        
        # Get outputs from selected adapters
        adapter_outputs = []
        for name in adapter_names:
            if name in self.adapters:
                output = self.adapters[name](hidden_states)
                adapter_outputs.append(output)
        
        if not adapter_outputs:
            return hidden_states
        
        # Fuse adapter outputs
        if self.fusion_method == "average":
            fused_output = torch.stack(adapter_outputs).mean(dim=0)
        elif self.fusion_method == "weighted":
            weights = F.softmax(self.fusion_weights[:len(adapter_outputs)], dim=0)
            fused_output = sum(w * output for w, output in zip(weights, adapter_outputs))
        elif self.fusion_method == "attention":
            # Stack adapter outputs for attention
            stacked_outputs = torch.stack(adapter_outputs, dim=1)  # (batch, num_adapters, seq_len, hidden)
            batch_size, num_adapters, seq_len, hidden_size = stacked_outputs.shape
            
            # Reshape for attention
            stacked_outputs = stacked_outputs.view(batch_size * seq_len, num_adapters, hidden_size)
            
            # Apply attention
            attended_output, _ = self.fusion_attention(
                stacked_outputs, stacked_outputs, stacked_outputs
            )
            
            # Take the first output (query result)
            fused_output = attended_output[:, 0, :].view(batch_size, seq_len, hidden_size)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        return fused_output
    
    def get_adapter_names(self) -> list:
        """Get list of available adapter names"""
        return list(self.adapters.keys())
    
    def add_adapter(self, name: str, adapter: AdapterLayer):
        """Add a new adapter"""
        self.adapters[name] = adapter
    
    def remove_adapter(self, name: str):
        """Remove an adapter"""
        if name in self.adapters:
            del self.adapters[name]
