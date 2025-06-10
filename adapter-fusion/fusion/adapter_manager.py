"""
Adapter manager for handling multiple adapters in fusion
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from collections import OrderedDict
import logging

from ..config.adapter_config import AdapterConfig
from ..adapters.adapter_layer import BottleneckAdapter

logger = logging.getLogger(__name__)


class AdapterManager(nn.Module):
    """
    Manages multiple adapters for fusion
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_configs: Dict[str, AdapterConfig],
        freeze_adapters: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.adapter_configs = adapter_configs
        self.freeze_adapters = freeze_adapters
        
        # Create adapters
        self.adapters = nn.ModuleDict()
        self.adapter_names = list(adapter_configs.keys())
        
        for adapter_name, adapter_config in adapter_configs.items():
            adapter = self._create_adapter(adapter_config)
            self.adapters[adapter_name] = adapter
            
            # Freeze adapter if specified
            if freeze_adapters:
                for param in adapter.parameters():
                    param.requires_grad = False
        
        logger.info(f"Created {len(self.adapters)} adapters: {self.adapter_names}")
    
    def _create_adapter(self, adapter_config: AdapterConfig) -> nn.Module:
        """Create an adapter based on configuration"""
        
        adapter_size = adapter_config.get_adapter_size(self.hidden_size)
        
        if adapter_config.adapter_type == "bottleneck":
            return BottleneckAdapter(
                input_size=self.hidden_size,
                adapter_size=adapter_size,
                dropout=adapter_config.adapter_dropout,
                activation=adapter_config.adapter_activation,
                use_residual=adapter_config.use_residual,
                use_layer_norm=adapter_config.use_layer_norm,
                init_range=adapter_config.adapter_init_range,
                scaling=adapter_config.adapter_scaling
            )
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_config.adapter_type}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        adapter_names: Optional[List[str]] = None,
        adapter_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through selected adapters
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            adapter_names: List of adapter names to use (None = all)
            adapter_masks: Optional masks for each adapter
            
        Returns:
            Dict mapping adapter names to their outputs
        """
        if adapter_names is None:
            adapter_names = self.adapter_names
        
        adapter_outputs = {}
        
        for adapter_name in adapter_names:
            if adapter_name not in self.adapters:
                logger.warning(f"Adapter '{adapter_name}' not found, skipping")
                continue
            
            adapter = self.adapters[adapter_name]
            
            # Apply adapter mask if provided
            if adapter_masks and adapter_name in adapter_masks:
                mask = adapter_masks[adapter_name]
                masked_hidden_states = hidden_states * mask.unsqueeze(-1)
                output = adapter(masked_hidden_states)
            else:
                output = adapter(hidden_states)
            
            adapter_outputs[adapter_name] = output
        
        return adapter_outputs
    
    def get_adapter_names(self) -> List[str]:
        """Get list of available adapter names"""
        return self.adapter_names
    
    def get_adapter(self, adapter_name: str) -> nn.Module:
        """Get specific adapter by name"""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        return self.adapters[adapter_name]
    
    def add_adapter(
        self,
        adapter_name: str,
        adapter_config: AdapterConfig,
        adapter_weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Add a new adapter"""
        if adapter_name in self.adapters:
            logger.warning(f"Adapter '{adapter_name}' already exists, replacing")
        
        # Create new adapter
        adapter = self._create_adapter(adapter_config)
        
        # Load weights if provided
        if adapter_weights is not None:
            adapter.load_state_dict(adapter_weights)
        
        # Freeze if specified
        if self.freeze_adapters:
            for param in adapter.parameters():
                param.requires_grad = False
        
        self.adapters[adapter_name] = adapter
        self.adapter_configs[adapter_name] = adapter_config
        
        if adapter_name not in self.adapter_names:
            self.adapter_names.append(adapter_name)
        
        logger.info(f"Added adapter '{adapter_name}'")
    
    def remove_adapter(self, adapter_name: str):
        """Remove an adapter"""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        del self.adapters[adapter_name]
        del self.adapter_configs[adapter_name]
        self.adapter_names.remove(adapter_name)
        
        logger.info(f"Removed adapter '{adapter_name}'")
    
    def freeze_adapter(self, adapter_name: str):
        """Freeze specific adapter"""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        for param in self.adapters[adapter_name].parameters():
            param.requires_grad = False
        
        logger.info(f"Frozen adapter '{adapter_name}'")
    
    def unfreeze_adapter(self, adapter_name: str):
        """Unfreeze specific adapter"""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        for param in self.adapters[adapter_name].parameters():
            param.requires_grad = True
        
        logger.info(f"Unfrozen adapter '{adapter_name}'")
    
    def freeze_all_adapters(self):
        """Freeze all adapters"""
        for adapter_name in self.adapter_names:
            self.freeze_adapter(adapter_name)
        self.freeze_adapters = True
    
    def unfreeze_all_adapters(self):
        """Unfreeze all adapters"""
        for adapter_name in self.adapter_names:
            self.unfreeze_adapter(adapter_name)
        self.freeze_adapters = False
    
    def load_adapter_from_path(
        self,
        adapter_name: str,
        adapter_path: str,
        adapter_config: Optional[AdapterConfig] = None
    ):
        """Load adapter from saved path"""
        
        # Load adapter config if not provided
        if adapter_config is None:
            config_path = os.path.join(adapter_path, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                adapter_config = AdapterConfig.from_dict(config_dict)
            else:
                raise ValueError(f"No adapter config found at {config_path}")
        
        # Load adapter weights
        weights_path = os.path.join(adapter_path, "adapter_model.bin")
        if os.path.exists(weights_path):
            adapter_weights = torch.load(weights_path, map_location="cpu")
        else:
            raise ValueError(f"No adapter weights found at {weights_path}")
        
        # Add adapter
        self.add_adapter(adapter_name, adapter_config, adapter_weights)
        
        logger.info(f"Loaded adapter '{adapter_name}' from {adapter_path}")
    
    def save_adapter(self, adapter_name: str, save_path: str):
        """Save specific adapter"""
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save adapter weights
        adapter = self.adapters[adapter_name]
        torch.save(adapter.state_dict(), os.path.join(save_path, "adapter_model.bin"))
        
        # Save adapter config
        adapter_config = self.adapter_configs[adapter_name]
        with open(os.path.join(save_path, "adapter_config.json"), 'w') as f:
            json.dump(adapter_config.to_dict(), f, indent=2)
        
        logger.info(f"Saved adapter '{adapter_name}' to {save_path}")
    
    def save_all_adapters(self, save_dir: str):
        """Save all adapters"""
        for adapter_name in self.adapter_names:
            adapter_save_path = os.path.join(save_dir, adapter_name)
            self.save_adapter(adapter_name, adapter_save_path)
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about all adapters"""
        info = {
            "num_adapters": len(self.adapters),
            "adapter_names": self.adapter_names,
            "freeze_adapters": self.freeze_adapters,
            "adapters": {}
        }
        
        total_adapter_params = 0
        
        for adapter_name, adapter in self.adapters.items():
            adapter_params = sum(p.numel() for p in adapter.parameters())
            trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
            
            info["adapters"][adapter_name] = {
                "total_params": adapter_params,
                "trainable_params": trainable_params,
                "config": self.adapter_configs[adapter_name].to_dict()
            }
            
            total_adapter_params += adapter_params
        
        info["total_adapter_params"] = total_adapter_params
        
        return info
    
    def print_adapter_info(self):
        """Print adapter information"""
        info = self.get_adapter_info()
        
        print(f"Adapter Manager Information:")
        print(f"  Number of adapters: {info['num_adapters']}")
        print(f"  Adapter names: {info['adapter_names']}")
        print(f"  Freeze adapters: {info['freeze_adapters']}")
        print(f"  Total adapter parameters: {info['total_adapter_params']:,}")
        
        for adapter_name, adapter_info in info["adapters"].items():
            print(f"  {adapter_name}:")
            print(f"    Total params: {adapter_info['total_params']:,}")
            print(f"    Trainable params: {adapter_info['trainable_params']:,}")
    
    @classmethod
    def from_adapter_paths(
        cls,
        hidden_size: int,
        adapter_paths: Dict[str, str],
        freeze_adapters: bool = True
    ) -> "AdapterManager":
        """Create adapter manager from saved adapter paths"""
        
        adapter_configs = {}
        
        # Load adapter configs
        for adapter_name, adapter_path in adapter_paths.items():
            config_path = os.path.join(adapter_path, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                adapter_configs[adapter_name] = AdapterConfig.from_dict(config_dict)
            else:
                # Use default config
                adapter_configs[adapter_name] = AdapterConfig(task_name=adapter_name)
        
        # Create manager
        manager = cls(hidden_size, adapter_configs, freeze_adapters)
        
        # Load adapter weights
        for adapter_name, adapter_path in adapter_paths.items():
            weights_path = os.path.join(adapter_path, "adapter_model.bin")
            if os.path.exists(weights_path):
                adapter_weights = torch.load(weights_path, map_location="cpu")
                manager.adapters[adapter_name].load_state_dict(adapter_weights)
                logger.info(f"Loaded weights for adapter '{adapter_name}'")
        
        return manager
