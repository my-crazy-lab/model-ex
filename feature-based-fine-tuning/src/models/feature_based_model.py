"""
Feature-Based Fine-Tuning Model Implementation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from transformers import (
    AutoModel, AutoTokenizer, AutoImageProcessor,
    BertModel, RobertaModel, DistilBertModel,
    ResNetModel, ViTModel
)
from .classifiers import LinearClassifier, MLPClassifier, AttentionClassifier
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class FeatureBasedModel(nn.Module):
    """
    Feature-based fine-tuning model with frozen backbone and trainable classifier
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        freeze_backbone: bool = True,
        feature_dim: Optional[int] = None,
        task: str = 'classification',
        **kwargs
    ):
        super().__init__()
        
        self.backbone = backbone
        self.classifier = classifier
        self.freeze_backbone = freeze_backbone
        self.feature_dim = feature_dim
        self.task = task
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Track frozen parameters
        self._frozen_params = set()
        self._trainable_params = set()
        self._update_param_tracking()
        
        logger.info(f"Initialized FeatureBasedModel with {self._count_parameters()} parameters")
        logger.info(f"Frozen parameters: {len(self._frozen_params)}")
        logger.info(f"Trainable parameters: {len(self._trainable_params)}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_classes: int,
        task: str = 'classification',
        classifier_type: str = 'linear',
        freeze_backbone: bool = True,
        classifier_config: Optional[Dict] = None,
        **kwargs
    ) -> 'FeatureBasedModel':
        """
        Create feature-based model from pre-trained backbone
        
        Args:
            model_name_or_path: Pre-trained model name or path
            num_classes: Number of output classes
            task: Task type ('classification', 'regression')
            classifier_type: Type of classifier ('linear', 'mlp', 'attention')
            freeze_backbone: Whether to freeze backbone parameters
            classifier_config: Configuration for classifier
            
        Returns:
            FeatureBasedModel instance
        """
        try:
            # Load backbone model
            backbone = AutoModel.from_pretrained(model_name_or_path, **kwargs)
            
            # Get feature dimension
            feature_dim = cls._get_feature_dimension(backbone, model_name_or_path)
            
            # Create classifier
            classifier_config = classifier_config or {}
            classifier = cls._create_classifier(
                classifier_type=classifier_type,
                input_dim=feature_dim,
                num_classes=num_classes,
                task=task,
                **classifier_config
            )
            
            # Create model
            model = cls(
                backbone=backbone,
                classifier=classifier,
                freeze_backbone=freeze_backbone,
                feature_dim=feature_dim,
                task=task
            )
            
            logger.info(f"Created FeatureBasedModel from {model_name_or_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model from {model_name_or_path}: {e}")
            raise
    
    @staticmethod
    def _get_feature_dimension(backbone: nn.Module, model_name: str) -> int:
        """Get feature dimension from backbone model"""
        if hasattr(backbone, 'config'):
            if hasattr(backbone.config, 'hidden_size'):
                return backbone.config.hidden_size
            elif hasattr(backbone.config, 'num_features'):
                return backbone.config.num_features
        
        # Try to infer from model name
        if 'bert-base' in model_name.lower():
            return 768
        elif 'bert-large' in model_name.lower():
            return 1024
        elif 'distilbert' in model_name.lower():
            return 768
        elif 'roberta-base' in model_name.lower():
            return 768
        elif 'roberta-large' in model_name.lower():
            return 1024
        elif 'resnet' in model_name.lower():
            return 2048  # ResNet-50 default
        elif 'vit' in model_name.lower():
            return 768   # ViT-Base default
        else:
            # Default fallback
            return 768
    
    @staticmethod
    def _create_classifier(
        classifier_type: str,
        input_dim: int,
        num_classes: int,
        task: str = 'classification',
        **config
    ) -> nn.Module:
        """Create classifier based on type"""
        if classifier_type == 'linear':
            return LinearClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                task=task,
                **config
            )
        elif classifier_type == 'mlp':
            return MLPClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                task=task,
                **config
            )
        elif classifier_type == 'attention':
            return AttentionClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                task=task,
                **config
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        logger.info("Backbone parameters frozen")
    
    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        logger.info("Backbone parameters unfrozen")
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specific layers by name"""
        frozen_count = 0
        for name, param in self.backbone.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                frozen_count += 1
        
        self._update_param_tracking()
        logger.info(f"Frozen {frozen_count} parameters in layers: {layer_names}")
    
    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers by name"""
        unfrozen_count = 0
        for name, param in self.backbone.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                unfrozen_count += 1
        
        self._update_param_tracking()
        logger.info(f"Unfrozen {unfrozen_count} parameters in layers: {layer_names}")
    
    def _update_param_tracking(self):
        """Update tracking of frozen and trainable parameters"""
        self._frozen_params.clear()
        self._trainable_params.clear()
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                self._trainable_params.add(name)
            else:
                self._frozen_params.add(name)
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count total, trainable, and frozen parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs for text input
            attention_mask: Attention mask for text input
            pixel_values: Pixel values for image input
            labels: Ground truth labels
            
        Returns:
            Dictionary containing logits, loss, and features
        """
        # Prepare inputs for backbone
        backbone_inputs = {}
        if input_ids is not None:
            backbone_inputs['input_ids'] = input_ids
        if attention_mask is not None:
            backbone_inputs['attention_mask'] = attention_mask
        if pixel_values is not None:
            backbone_inputs['pixel_values'] = pixel_values
        
        # Extract features from backbone
        if self.freeze_backbone and self.training:
            # Use no_grad for frozen backbone during training
            with torch.no_grad():
                backbone_outputs = self.backbone(**backbone_inputs)
        else:
            backbone_outputs = self.backbone(**backbone_inputs)
        
        # Get pooled features
        if hasattr(backbone_outputs, 'pooler_output') and backbone_outputs.pooler_output is not None:
            features = backbone_outputs.pooler_output
        elif hasattr(backbone_outputs, 'last_hidden_state'):
            # Use mean pooling for sequence models
            features = backbone_outputs.last_hidden_state.mean(dim=1)
        else:
            # Fallback to first element if it's a tensor
            features = backbone_outputs[0] if isinstance(backbone_outputs, tuple) else backbone_outputs
            if features.dim() > 2:
                features = features.mean(dim=1)  # Pool over sequence dimension
        
        # Classify
        logits = self.classifier(features)
        
        # Prepare outputs
        outputs = {
            'logits': logits,
            'features': features
        }
        
        # Compute loss if labels provided
        if labels is not None:
            if self.task == 'classification':
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
            elif self.task == 'regression':
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.squeeze(), labels.float())
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            outputs['loss'] = loss
        
        return outputs
    
    def extract_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Extract features without classification
        
        Returns:
            Feature tensor
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **kwargs
            )
            return outputs['features']
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters"""
        return [param for param in self.parameters() if param.requires_grad]
    
    def get_frozen_parameters(self) -> List[nn.Parameter]:
        """Get list of frozen parameters"""
        return [param for param in self.parameters() if not param.requires_grad]
    
    def print_parameter_status(self):
        """Print detailed parameter status"""
        param_counts = self._count_parameters()
        
        print(f"Parameter Status:")
        print(f"  Total parameters: {param_counts['total']:,}")
        print(f"  Trainable parameters: {param_counts['trainable']:,}")
        print(f"  Frozen parameters: {param_counts['frozen']:,}")
        print(f"  Trainable ratio: {param_counts['trainable']/param_counts['total']:.2%}")
        
        print(f"\nTrainable parameter groups:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,} parameters")
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save configuration
        config = {
            'feature_dim': self.feature_dim,
            'task': self.task,
            'freeze_backbone': self.freeze_backbone,
            'classifier_type': self.classifier.__class__.__name__,
            'num_classes': getattr(self.classifier, 'num_classes', None)
        }
        
        with open(os.path.join(save_directory, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {save_directory}")


# Example usage and testing
if __name__ == "__main__":
    # Test with BERT
    model = FeatureBasedModel.from_pretrained(
        'bert-base-uncased',
        num_classes=3,
        classifier_type='linear',
        freeze_backbone=True
    )
    
    # Print parameter status
    model.print_parameter_status()
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)
    labels = torch.randint(0, 3, (2,))
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    print("FeatureBasedModel test completed successfully!")
