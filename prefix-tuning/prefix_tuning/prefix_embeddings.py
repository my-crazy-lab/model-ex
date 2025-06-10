"""
Prefix embeddings implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
import logging

from ..config.prefix_config import PrefixConfig, PrefixInitMethod, ReparameterizationType

logger = logging.getLogger(__name__)


class PrefixReparameterization(nn.Module):
    """Reparameterization network for prefix embeddings"""
    
    def __init__(
        self,
        config: PrefixConfig,
        input_size: int,
        output_size: int
    ):
        super().__init__()
        
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        
        if config.reparameterization_type == ReparameterizationType.MLP:
            self.reparameterization = self._build_mlp()
        elif config.reparameterization_type == ReparameterizationType.LSTM:
            self.reparameterization = self._build_lstm()
        elif config.reparameterization_type == ReparameterizationType.TRANSFORMER:
            self.reparameterization = self._build_transformer()
        else:
            raise ValueError(f"Unsupported reparameterization type: {config.reparameterization_type}")
    
    def _build_mlp(self) -> nn.Module:
        """Build MLP reparameterization network"""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_size, self.config.reparameterization_hidden_size))
        layers.append(self._get_activation())
        
        # Hidden layers
        for _ in range(self.config.reparameterization_num_layers - 1):
            layers.append(nn.Linear(
                self.config.reparameterization_hidden_size,
                self.config.reparameterization_hidden_size
            ))
            layers.append(self._get_activation())
        
        # Output layer
        layers.append(nn.Linear(self.config.reparameterization_hidden_size, self.output_size))
        
        return nn.Sequential(*layers)
    
    def _build_lstm(self) -> nn.Module:
        """Build LSTM reparameterization network"""
        return nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.config.reparameterization_hidden_size,
            num_layers=self.config.reparameterization_num_layers,
            batch_first=True,
            dropout=self.config.prefix_dropout if self.config.reparameterization_num_layers > 1 else 0
        )
    
    def _build_transformer(self) -> nn.Module:
        """Build Transformer reparameterization network"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.reparameterization_hidden_size,
            nhead=8,
            dim_feedforward=self.config.reparameterization_hidden_size * 4,
            dropout=self.config.prefix_dropout,
            activation=self.config.reparameterization_activation,
            batch_first=True
        )
        
        return nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.reparameterization_num_layers
        )
    
    def _get_activation(self) -> nn.Module:
        """Get activation function"""
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "swish": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU()
        }
        
        return activation_map.get(
            self.config.reparameterization_activation.lower(),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through reparameterization network"""
        if self.config.reparameterization_type == ReparameterizationType.LSTM:
            # LSTM returns (output, (hidden, cell))
            output, _ = self.reparameterization(x)
            return output
        else:
            return self.reparameterization(x)


class PrefixEmbeddings(nn.Module):
    """Prefix embeddings for prefix tuning"""
    
    def __init__(
        self,
        config: PrefixConfig,
        model_config: Any,
        tokenizer: Optional[Any] = None
    ):
        super().__init__()
        
        self.config = config
        self.model_config = model_config
        self.tokenizer = tokenizer
        
        # Get dimensions
        self.prefix_dims = config.get_prefix_dimensions(model_config)
        self.prefix_length = self.prefix_dims["prefix_length"]
        self.hidden_size = self.prefix_dims["hidden_size"]
        self.num_heads = self.prefix_dims["num_heads"]
        self.head_dim = self.prefix_dims["head_dim"]
        
        # Number of layers to apply prefixes
        self.num_layers = getattr(model_config, 'num_hidden_layers', model_config.num_layers)
        
        # Initialize prefix parameters
        self._initialize_prefix_parameters()
        
        # Setup reparameterization if enabled
        if config.reparameterization:
            self._setup_reparameterization()
        
        # Setup projection if enabled
        if config.prefix_projection:
            self.prefix_projection = nn.Linear(
                self.hidden_size,
                config.prefix_projection_hidden_size
            )
        
        logger.info(f"PrefixEmbeddings initialized with {self.get_num_parameters():,} parameters")
    
    def _initialize_prefix_parameters(self):
        """Initialize prefix parameters"""
        if self.config.different_prefix_per_layer:
            # Different prefix for each layer
            if self.config.reparameterization:
                # Initialize smaller parameters for reparameterization
                param_size = self.config.reparameterization_hidden_size
            else:
                # Direct prefix parameters
                param_size = self.hidden_size
            
            self.prefix_keys = nn.Parameter(
                torch.randn(self.num_layers, self.prefix_length, param_size)
            )
            self.prefix_values = nn.Parameter(
                torch.randn(self.num_layers, self.prefix_length, param_size)
            )
        else:
            # Shared prefix across layers
            if self.config.reparameterization:
                param_size = self.config.reparameterization_hidden_size
            else:
                param_size = self.hidden_size
            
            self.prefix_keys = nn.Parameter(
                torch.randn(self.prefix_length, param_size)
            )
            self.prefix_values = nn.Parameter(
                torch.randn(self.prefix_length, param_size)
            )
        
        # Initialize based on method
        self._apply_initialization()
    
    def _apply_initialization(self):
        """Apply initialization method to prefix parameters"""
        if self.config.init_method == PrefixInitMethod.RANDOM:
            # Already initialized as random
            pass
        elif self.config.init_method == PrefixInitMethod.NORMAL:
            nn.init.normal_(self.prefix_keys, std=self.config.init_std)
            nn.init.normal_(self.prefix_values, std=self.config.init_std)
        elif self.config.init_method == PrefixInitMethod.UNIFORM:
            nn.init.uniform_(self.prefix_keys, -self.config.init_range, self.config.init_range)
            nn.init.uniform_(self.prefix_values, -self.config.init_range, self.config.init_range)
        elif self.config.init_method == PrefixInitMethod.XAVIER:
            nn.init.xavier_uniform_(self.prefix_keys)
            nn.init.xavier_uniform_(self.prefix_values)
        elif self.config.init_method == PrefixInitMethod.KAIMING:
            nn.init.kaiming_uniform_(self.prefix_keys)
            nn.init.kaiming_uniform_(self.prefix_values)
        elif self.config.init_method == PrefixInitMethod.FROM_VOCAB:
            self._initialize_from_vocab()
        elif self.config.init_method == PrefixInitMethod.FROM_TEXT:
            self._initialize_from_text()
    
    def _initialize_from_vocab(self):
        """Initialize from vocabulary embeddings"""
        if not hasattr(self.model_config, 'vocab_size'):
            logger.warning("Model config doesn't have vocab_size, falling back to random initialization")
            return
        
        # Sample random vocabulary indices
        vocab_indices = torch.randint(
            0, min(self.config.init_vocab_size, self.model_config.vocab_size),
            (self.prefix_length,)
        )
        
        # This would require access to the model's embedding layer
        # For now, use random initialization
        logger.info("Vocabulary-based initialization not fully implemented, using random")
    
    def _initialize_from_text(self):
        """Initialize from text embeddings"""
        if not self.tokenizer or not self.config.init_text:
            logger.warning("Tokenizer or init_text not provided, falling back to random initialization")
            return
        
        # Tokenize initialization text
        tokens = self.tokenizer(
            self.config.init_text,
            return_tensors="pt",
            max_length=self.prefix_length,
            truncation=True,
            padding="max_length"
        )
        
        # This would require access to the model's embedding layer
        # For now, use random initialization
        logger.info("Text-based initialization not fully implemented, using random")
    
    def _setup_reparameterization(self):
        """Setup reparameterization networks"""
        if self.config.different_prefix_per_layer:
            # Separate reparameterization for each layer
            self.key_reparameterization = nn.ModuleList([
                PrefixReparameterization(
                    self.config,
                    self.config.reparameterization_hidden_size,
                    self.hidden_size
                ) for _ in range(self.num_layers)
            ])
            self.value_reparameterization = nn.ModuleList([
                PrefixReparameterization(
                    self.config,
                    self.config.reparameterization_hidden_size,
                    self.hidden_size
                ) for _ in range(self.num_layers)
            ])
        else:
            # Shared reparameterization
            self.key_reparameterization = PrefixReparameterization(
                self.config,
                self.config.reparameterization_hidden_size,
                self.hidden_size
            )
            self.value_reparameterization = PrefixReparameterization(
                self.config,
                self.config.reparameterization_hidden_size,
                self.hidden_size
            )
    
    def get_prefix_embeddings(
        self,
        batch_size: int,
        layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prefix key and value embeddings
        
        Args:
            batch_size: Batch size
            layer_idx: Layer index (required if different_prefix_per_layer=True)
            
        Returns:
            Tuple of (prefix_keys, prefix_values)
        """
        device = self.prefix_keys.device
        
        if self.config.different_prefix_per_layer:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when different_prefix_per_layer=True")
            
            # Get layer-specific prefix parameters
            prefix_keys = self.prefix_keys[layer_idx]  # [prefix_length, hidden_size]
            prefix_values = self.prefix_values[layer_idx]
            
            # Apply reparameterization if enabled
            if self.config.reparameterization:
                prefix_keys = self.key_reparameterization[layer_idx](prefix_keys.unsqueeze(0)).squeeze(0)
                prefix_values = self.value_reparameterization[layer_idx](prefix_values.unsqueeze(0)).squeeze(0)
        else:
            # Shared prefix across layers
            prefix_keys = self.prefix_keys  # [prefix_length, hidden_size]
            prefix_values = self.prefix_values
            
            # Apply reparameterization if enabled
            if self.config.reparameterization:
                prefix_keys = self.key_reparameterization(prefix_keys.unsqueeze(0)).squeeze(0)
                prefix_values = self.value_reparameterization(prefix_values.unsqueeze(0)).squeeze(0)
        
        # Apply projection if enabled
        if self.config.prefix_projection:
            prefix_keys = self.prefix_projection(prefix_keys)
            prefix_values = self.prefix_projection(prefix_values)
        
        # Expand for batch size
        prefix_keys = prefix_keys.unsqueeze(0).expand(batch_size, -1, -1)
        prefix_values = prefix_values.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Reshape for multi-head attention
        # [batch_size, prefix_length, hidden_size] -> [batch_size, prefix_length, num_heads, head_dim]
        prefix_keys = prefix_keys.view(batch_size, self.prefix_length, self.num_heads, self.head_dim)
        prefix_values = prefix_values.view(batch_size, self.prefix_length, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, prefix_length, head_dim]
        prefix_keys = prefix_keys.transpose(1, 2)
        prefix_values = prefix_values.transpose(1, 2)
        
        return prefix_keys, prefix_values
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_efficiency(self, total_model_params: int) -> Dict[str, float]:
        """Get parameter efficiency metrics"""
        prefix_params = self.get_num_parameters()
        
        return {
            "prefix_parameters": prefix_params,
            "total_parameters": total_model_params,
            "efficiency": prefix_params / total_model_params * 100,
            "reduction_factor": total_model_params / prefix_params
        }
    
    def save_prefix_embeddings(self, filepath: str):
        """Save prefix embeddings"""
        state_dict = {
            "prefix_keys": self.prefix_keys,
            "prefix_values": self.prefix_values,
            "config": self.config.to_dict()
        }
        
        if self.config.reparameterization:
            if self.config.different_prefix_per_layer:
                state_dict["key_reparameterization"] = [module.state_dict() for module in self.key_reparameterization]
                state_dict["value_reparameterization"] = [module.state_dict() for module in self.value_reparameterization]
            else:
                state_dict["key_reparameterization"] = self.key_reparameterization.state_dict()
                state_dict["value_reparameterization"] = self.value_reparameterization.state_dict()
        
        torch.save(state_dict, filepath)
        logger.info(f"Prefix embeddings saved to {filepath}")
    
    def load_prefix_embeddings(self, filepath: str):
        """Load prefix embeddings"""
        state_dict = torch.load(filepath, map_location="cpu")
        
        self.prefix_keys.data = state_dict["prefix_keys"]
        self.prefix_values.data = state_dict["prefix_values"]
        
        if self.config.reparameterization and "key_reparameterization" in state_dict:
            if self.config.different_prefix_per_layer:
                for i, module_state in enumerate(state_dict["key_reparameterization"]):
                    self.key_reparameterization[i].load_state_dict(module_state)
                for i, module_state in enumerate(state_dict["value_reparameterization"]):
                    self.value_reparameterization[i].load_state_dict(module_state)
            else:
                self.key_reparameterization.load_state_dict(state_dict["key_reparameterization"])
                self.value_reparameterization.load_state_dict(state_dict["value_reparameterization"])
        
        logger.info(f"Prefix embeddings loaded from {filepath}")
