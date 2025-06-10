"""
Basic adapter layer implementations for Adapter Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from abc import ABC, abstractmethod


class AdapterLayer(nn.Module, ABC):
    """Abstract base class for adapter layers"""
    
    def __init__(
        self,
        input_size: int,
        adapter_size: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        self.input_size = input_size
        self.adapter_size = adapter_size
        self.dropout = dropout
        self.activation_name = activation
        
        # Set activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "swish" or activation == "silu":
            self.activation = F.silu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    @abstractmethod
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapter"""
        pass


class BottleneckAdapter(AdapterLayer):
    """
    Bottleneck adapter implementation
    
    Architecture: Input → Down → Activation → Up → Residual → Output
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
        super().__init__(input_size, adapter_size, dropout, activation)
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.scaling = scaling
        
        # Down-projection
        self.down_project = nn.Linear(input_size, adapter_size)
        
        # Up-projection
        self.up_project = nn.Linear(adapter_size, input_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Optional layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_size)
        
        # Initialize weights
        self._init_weights(init_range)
    
    def _init_weights(self, init_range: float):
        """Initialize adapter weights"""
        # Initialize down projection normally
        nn.init.normal_(self.down_project.weight, std=init_range)
        nn.init.zeros_(self.down_project.bias)
        
        # Initialize up projection to zero (important for stable training)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bottleneck adapter
        
        Args:
            hidden_states: [batch_size, seq_len, input_size]
        Returns:
            output: [batch_size, seq_len, input_size]
        """
        # Store residual
        residual = hidden_states
        
        # Down-projection
        hidden_states = self.down_project(hidden_states)
        
        # Activation
        hidden_states = self.activation(hidden_states)
        
        # Dropout
        hidden_states = self.dropout_layer(hidden_states)
        
        # Up-projection
        hidden_states = self.up_project(hidden_states)
        
        # Apply scaling
        if self.scaling != 1.0:
            hidden_states = hidden_states * self.scaling
        
        # Residual connection
        if self.use_residual:
            output = residual + hidden_states
        else:
            output = hidden_states
        
        # Layer normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        return output


class ParallelAdapter(AdapterLayer):
    """
    Parallel adapter implementation
    
    Processes input in parallel with the main transformer layer
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
        super().__init__(input_size, adapter_size, dropout, activation)
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.scaling = scaling
        
        # Parallel processing layers
        self.adapter_dense = nn.Linear(input_size, adapter_size)
        self.adapter_output = nn.Linear(adapter_size, input_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Optional layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_size)
        
        # Initialize weights
        self._init_weights(init_range)
    
    def _init_weights(self, init_range: float):
        """Initialize adapter weights"""
        # Initialize all layers normally
        for layer in [self.adapter_dense, self.adapter_output]:
            nn.init.normal_(layer.weight, std=init_range)
            nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through parallel adapter
        
        Args:
            hidden_states: [batch_size, seq_len, input_size]
        Returns:
            output: [batch_size, seq_len, input_size]
        """
        # Store residual
        residual = hidden_states
        
        # Parallel processing
        adapter_output = self.adapter_dense(hidden_states)
        adapter_output = self.activation(adapter_output)
        adapter_output = self.dropout_layer(adapter_output)
        adapter_output = self.adapter_output(adapter_output)
        
        # Apply scaling
        if self.scaling != 1.0:
            adapter_output = adapter_output * self.scaling
        
        # Residual connection
        if self.use_residual:
            output = residual + adapter_output
        else:
            output = adapter_output
        
        # Layer normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        return output


class SequentialAdapter(AdapterLayer):
    """
    Sequential adapter with multiple bottleneck layers
    """
    
    def __init__(
        self,
        input_size: int,
        adapter_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        use_residual: bool = True,
        use_layer_norm: bool = False,
        init_range: float = 1e-3,
        scaling: float = 1.0
    ):
        super().__init__(input_size, adapter_size, dropout, activation)
        
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.scaling = scaling
        
        # Create sequential layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                # First layer: input_size -> adapter_size
                layers.append(nn.Linear(input_size, adapter_size))
            elif i == num_layers - 1:
                # Last layer: adapter_size -> input_size
                layers.append(nn.Linear(adapter_size, input_size))
            else:
                # Middle layers: adapter_size -> adapter_size
                layers.append(nn.Linear(adapter_size, adapter_size))
            
            # Add activation and dropout (except for last layer)
            if i < num_layers - 1:
                layers.append(nn.ReLU() if activation == "relu" else nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.adapter_layers = nn.Sequential(*layers)
        
        # Optional layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_size)
        
        # Initialize weights
        self._init_weights(init_range)
    
    def _init_weights(self, init_range: float):
        """Initialize adapter weights"""
        for module in self.adapter_layers:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_range)
                nn.init.zeros_(module.bias)
        
        # Initialize last layer to zero for stable training
        if len(self.adapter_layers) > 0:
            last_linear = None
            for module in reversed(self.adapter_layers):
                if isinstance(module, nn.Linear):
                    last_linear = module
                    break
            
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through sequential adapter
        
        Args:
            hidden_states: [batch_size, seq_len, input_size]
        Returns:
            output: [batch_size, seq_len, input_size]
        """
        # Store residual
        residual = hidden_states
        
        # Sequential processing
        adapter_output = self.adapter_layers(hidden_states)
        
        # Apply scaling
        if self.scaling != 1.0:
            adapter_output = adapter_output * self.scaling
        
        # Residual connection
        if self.use_residual:
            output = residual + adapter_output
        else:
            output = adapter_output
        
        # Layer normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        return output
