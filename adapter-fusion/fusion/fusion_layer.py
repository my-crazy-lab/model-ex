"""
Core fusion layer implementations for Adapter Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math
from abc import ABC, abstractmethod


class FusionLayer(nn.Module, ABC):
    """Abstract base class for fusion layers"""
    
    def __init__(
        self,
        hidden_size: int,
        num_adapters: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_adapters = num_adapters
        self.dropout = nn.Dropout(dropout)
    
    @abstractmethod
    def forward(
        self,
        adapter_outputs: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        adapter_masks: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Fuse adapter outputs"""
        pass


class AttentionFusion(FusionLayer):
    """
    Attention-based fusion mechanism
    
    Uses multi-head attention to learn how to combine adapter outputs
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_adapters: int,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_bias: bool = True
    ):
        super().__init__(hidden_size, num_adapters, dropout)
        
        self.num_attention_heads = num_attention_heads
        self.temperature = temperature
        self.head_dim = hidden_size // num_attention_heads
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})")
        
        # Query, Key, Value projections for fusion attention
        self.query = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.key = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize fusion weights"""
        for module in [self.query, self.key, self.value, self.output]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        adapter_outputs: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        adapter_masks: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through attention fusion
        
        Args:
            adapter_outputs: List of adapter outputs [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            adapter_masks: Optional masks for each adapter
            
        Returns:
            Fused output [batch_size, seq_len, hidden_size]
        """
        if not adapter_outputs:
            raise ValueError("adapter_outputs cannot be empty")
        
        batch_size, seq_len, hidden_size = adapter_outputs[0].shape
        
        # Stack adapter outputs: [batch_size, seq_len, num_adapters, hidden_size]
        stacked_outputs = torch.stack(adapter_outputs, dim=2)
        
        # Reshape for multi-head attention: [batch_size * seq_len, num_adapters, hidden_size]
        stacked_outputs = stacked_outputs.view(-1, self.num_adapters, hidden_size)
        
        # Compute queries, keys, values
        queries = self.query(stacked_outputs)  # [batch_size * seq_len, num_adapters, hidden_size]
        keys = self.key(stacked_outputs)
        values = self.value(stacked_outputs)
        
        # Reshape for multi-head attention
        queries = queries.view(-1, self.num_adapters, self.num_attention_heads, self.head_dim)
        keys = keys.view(-1, self.num_adapters, self.num_attention_heads, self.head_dim)
        values = values.view(-1, self.num_adapters, self.num_attention_heads, self.head_dim)
        
        # Transpose for attention computation: [batch_size * seq_len, num_heads, num_adapters, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / (math.sqrt(self.head_dim) * self.temperature)
        
        # Apply adapter masks if provided
        if adapter_masks is not None:
            # Create combined mask for all adapters
            combined_mask = torch.stack(adapter_masks, dim=1)  # [batch_size, num_adapters, seq_len]
            combined_mask = combined_mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, num_adapters, seq_len, 1]
            combined_mask = combined_mask.expand(-1, self.num_attention_heads, -1, -1, self.num_adapters)
            combined_mask = combined_mask.reshape(-1, self.num_attention_heads, self.num_adapters, self.num_adapters)
            
            attention_scores = attention_scores.masked_fill(combined_mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)
        
        # Reshape back: [batch_size * seq_len, num_heads, num_adapters, head_dim]
        attended_values = attended_values.transpose(1, 2)
        attended_values = attended_values.contiguous().view(-1, self.num_adapters, hidden_size)
        
        # Average across adapters (or use first adapter as query)
        fused_output = attended_values.mean(dim=1)  # [batch_size * seq_len, hidden_size]
        
        # Apply output projection
        fused_output = self.output(fused_output)
        
        # Reshape back to original dimensions
        fused_output = fused_output.view(batch_size, seq_len, hidden_size)
        
        # Apply layer normalization and residual connection
        # Use first adapter output as residual
        residual = adapter_outputs[0]
        fused_output = self.layer_norm(fused_output + residual)
        
        return fused_output


class WeightedFusion(FusionLayer):
    """
    Weighted fusion mechanism
    
    Learns weights to combine adapter outputs
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_adapters: int,
        dropout: float = 0.1,
        learnable_weights: bool = True,
        weight_initialization: str = "uniform",
        weight_constraint: str = "softmax"
    ):
        super().__init__(hidden_size, num_adapters, dropout)
        
        self.learnable_weights = learnable_weights
        self.weight_constraint = weight_constraint
        
        if learnable_weights:
            # Initialize fusion weights
            if weight_initialization == "uniform":
                weights = torch.ones(num_adapters) / num_adapters
            elif weight_initialization == "normal":
                weights = torch.randn(num_adapters)
            elif weight_initialization == "xavier":
                weights = torch.empty(num_adapters)
                nn.init.xavier_uniform_(weights.unsqueeze(0))
                weights = weights.squeeze(0)
            else:
                raise ValueError(f"Unknown weight_initialization: {weight_initialization}")
            
            self.fusion_weights = nn.Parameter(weights)
        else:
            # Fixed uniform weights
            self.register_buffer('fusion_weights', torch.ones(num_adapters) / num_adapters)
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        adapter_outputs: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        adapter_masks: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through weighted fusion
        
        Args:
            adapter_outputs: List of adapter outputs
            attention_mask: Not used in weighted fusion
            adapter_masks: Optional masks for each adapter
            
        Returns:
            Fused output
        """
        if not adapter_outputs:
            raise ValueError("adapter_outputs cannot be empty")
        
        if len(adapter_outputs) != self.num_adapters:
            raise ValueError(f"Expected {self.num_adapters} adapters, got {len(adapter_outputs)}")
        
        # Apply weight constraint
        if self.weight_constraint == "softmax":
            weights = F.softmax(self.fusion_weights, dim=0)
        elif self.weight_constraint == "sigmoid":
            weights = torch.sigmoid(self.fusion_weights)
            weights = weights / weights.sum()  # Normalize
        else:
            weights = self.fusion_weights
        
        # Apply adapter masks if provided
        if adapter_masks is not None:
            # Modify weights based on adapter availability
            adapter_availability = torch.stack([mask.float().mean() for mask in adapter_masks])
            weights = weights * adapter_availability
            weights = weights / weights.sum()  # Renormalize
        
        # Weighted combination
        fused_output = torch.zeros_like(adapter_outputs[0])
        for i, (adapter_output, weight) in enumerate(zip(adapter_outputs, weights)):
            fused_output += weight * adapter_output
        
        # Apply dropout and layer normalization
        fused_output = self.dropout(fused_output)
        fused_output = self.layer_norm(fused_output)
        
        return fused_output


class GatingFusion(FusionLayer):
    """
    Gating fusion mechanism
    
    Uses a gating network to dynamically select adapters
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_adapters: int,
        dropout: float = 0.1,
        gate_activation: str = "sigmoid",
        gate_hidden_size: Optional[int] = None,
        use_bias: bool = True
    ):
        super().__init__(hidden_size, num_adapters, dropout)
        
        self.gate_activation = gate_activation
        gate_hidden_size = gate_hidden_size or hidden_size // 2
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden_size, bias=use_bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, num_adapters, bias=use_bias)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gating network weights"""
        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        adapter_outputs: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        adapter_masks: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through gating fusion
        
        Args:
            adapter_outputs: List of adapter outputs
            attention_mask: Attention mask for input
            adapter_masks: Optional masks for each adapter
            
        Returns:
            Fused output
        """
        if not adapter_outputs:
            raise ValueError("adapter_outputs cannot be empty")
        
        batch_size, seq_len, hidden_size = adapter_outputs[0].shape
        
        # Use first adapter output as input to gating network
        gate_input = adapter_outputs[0]
        
        # Compute gates: [batch_size, seq_len, num_adapters]
        gates = self.gate_network(gate_input)
        
        # Apply gate activation
        if self.gate_activation == "sigmoid":
            gates = torch.sigmoid(gates)
        elif self.gate_activation == "tanh":
            gates = torch.tanh(gates)
        elif self.gate_activation == "softmax":
            gates = F.softmax(gates, dim=-1)
        else:
            raise ValueError(f"Unknown gate_activation: {self.gate_activation}")
        
        # Apply adapter masks if provided
        if adapter_masks is not None:
            for i, mask in enumerate(adapter_masks):
                gates[:, :, i] = gates[:, :, i] * mask.float()
        
        # Normalize gates if not using softmax
        if self.gate_activation != "softmax":
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply gates to adapter outputs
        fused_output = torch.zeros_like(adapter_outputs[0])
        for i, adapter_output in enumerate(adapter_outputs):
            gate_weight = gates[:, :, i:i+1]  # [batch_size, seq_len, 1]
            fused_output += gate_weight * adapter_output
        
        # Apply dropout and layer normalization
        fused_output = self.dropout(fused_output)
        fused_output = self.layer_norm(fused_output)
        
        return fused_output


class HierarchicalFusion(FusionLayer):
    """
    Hierarchical fusion mechanism
    
    Fuses adapters at multiple levels in the transformer
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_adapters: int,
        fusion_layers: List[int],
        dropout: float = 0.1,
        layer_fusion_method: str = "attention"
    ):
        super().__init__(hidden_size, num_adapters, dropout)
        
        self.fusion_layers = fusion_layers
        self.layer_fusion_method = layer_fusion_method
        
        # Create fusion layers for each specified layer
        self.layer_fusions = nn.ModuleDict()
        for layer_idx in fusion_layers:
            if layer_fusion_method == "attention":
                fusion_layer = AttentionFusion(hidden_size, num_adapters, dropout=dropout)
            elif layer_fusion_method == "weighted":
                fusion_layer = WeightedFusion(hidden_size, num_adapters, dropout=dropout)
            elif layer_fusion_method == "gating":
                fusion_layer = GatingFusion(hidden_size, num_adapters, dropout=dropout)
            else:
                raise ValueError(f"Unknown layer_fusion_method: {layer_fusion_method}")
            
            self.layer_fusions[str(layer_idx)] = fusion_layer
        
        # Final fusion across layers
        self.final_fusion = WeightedFusion(hidden_size, len(fusion_layers), dropout=dropout)
    
    def forward(
        self,
        layer_adapter_outputs: Dict[int, List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        adapter_masks: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical fusion
        
        Args:
            layer_adapter_outputs: Dict mapping layer indices to adapter outputs
            attention_mask: Attention mask
            adapter_masks: Optional masks for each adapter
            
        Returns:
            Fused output
        """
        layer_fused_outputs = []
        
        # Fuse adapters at each layer
        for layer_idx in self.fusion_layers:
            if layer_idx in layer_adapter_outputs:
                adapter_outputs = layer_adapter_outputs[layer_idx]
                fusion_layer = self.layer_fusions[str(layer_idx)]
                
                fused_output = fusion_layer(
                    adapter_outputs,
                    attention_mask=attention_mask,
                    adapter_masks=adapter_masks
                )
                layer_fused_outputs.append(fused_output)
        
        if not layer_fused_outputs:
            raise ValueError("No valid layer outputs found")
        
        # Final fusion across layers
        final_output = self.final_fusion(layer_fused_outputs)
        
        return final_output
