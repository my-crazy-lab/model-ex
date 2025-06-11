"""
Feature Extractor Wrapper for Pre-trained Models
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from transformers import (
    AutoModel, AutoTokenizer, AutoImageProcessor,
    BertModel, RobertaModel, DistilBertModel, ElectraModel,
    ResNetModel, ViTModel, CLIPModel
)

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """
    Wrapper for pre-trained models to extract features
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        freeze: bool = True,
        pooling_strategy: str = 'cls',
        **kwargs
    ):
        super().__init__()
        
        self.model = model
        self.model_name = model_name
        self.freeze = freeze
        self.pooling_strategy = pooling_strategy
        
        # Freeze parameters if specified
        if freeze:
            self._freeze_parameters()
        
        # Determine model type and feature dimension
        self.model_type = self._determine_model_type()
        self.feature_dim = self._get_feature_dimension()
        
        logger.info(f"Initialized FeatureExtractor: {model_name}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Feature dimension: {self.feature_dim}")
        logger.info(f"Frozen: {freeze}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        freeze: bool = True,
        pooling_strategy: str = 'cls',
        **kwargs
    ) -> 'FeatureExtractor':
        """
        Create feature extractor from pre-trained model
        
        Args:
            model_name_or_path: Pre-trained model name or path
            freeze: Whether to freeze parameters
            pooling_strategy: How to pool sequence features
            
        Returns:
            FeatureExtractor instance
        """
        try:
            # Load pre-trained model
            model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
            
            return cls(
                model=model,
                model_name=model_name_or_path,
                freeze=freeze,
                pooling_strategy=pooling_strategy
            )
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name_or_path}: {e}")
            raise
    
    def _freeze_parameters(self):
        """Freeze all model parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        logger.info("All parameters frozen")
    
    def _unfreeze_parameters(self):
        """Unfreeze all model parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("All parameters unfrozen")
    
    def _determine_model_type(self) -> str:
        """Determine the type of the model"""
        model_name_lower = self.model_name.lower()
        
        if any(name in model_name_lower for name in ['bert', 'roberta', 'distilbert', 'electra']):
            return 'text_encoder'
        elif any(name in model_name_lower for name in ['resnet', 'vit', 'efficientnet']):
            return 'image_encoder'
        elif 'clip' in model_name_lower:
            return 'multimodal'
        else:
            return 'unknown'
    
    def _get_feature_dimension(self) -> int:
        """Get feature dimension from model configuration"""
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            # Try different attribute names
            for attr in ['hidden_size', 'num_features', 'd_model', 'embed_dim']:
                if hasattr(config, attr):
                    return getattr(config, attr)
        
        # Fallback based on model name
        model_name_lower = self.model_name.lower()
        
        if 'bert-base' in model_name_lower or 'distilbert' in model_name_lower:
            return 768
        elif 'bert-large' in model_name_lower:
            return 1024
        elif 'roberta-base' in model_name_lower:
            return 768
        elif 'roberta-large' in model_name_lower:
            return 1024
        elif 'resnet' in model_name_lower:
            return 2048  # ResNet-50 default
        elif 'vit' in model_name_lower:
            return 768   # ViT-Base default
        else:
            # Try to infer from a dummy forward pass
            return self._infer_feature_dimension()
    
    def _infer_feature_dimension(self) -> int:
        """Infer feature dimension by running a dummy forward pass"""
        try:
            # Create dummy input based on model type
            if self.model_type == 'text_encoder':
                dummy_input = {
                    'input_ids': torch.randint(0, 1000, (1, 10)),
                    'attention_mask': torch.ones(1, 10)
                }
            elif self.model_type == 'image_encoder':
                dummy_input = {
                    'pixel_values': torch.randn(1, 3, 224, 224)
                }
            else:
                # Default to text input
                dummy_input = {
                    'input_ids': torch.randint(0, 1000, (1, 10))
                }
            
            # Run forward pass
            with torch.no_grad():
                outputs = self.model(**dummy_input)
                features = self._extract_features_from_outputs(outputs)
                return features.shape[-1]
                
        except Exception as e:
            logger.warning(f"Could not infer feature dimension: {e}")
            return 768  # Default fallback
    
    def _extract_features_from_outputs(self, outputs) -> torch.Tensor:
        """Extract features from model outputs based on pooling strategy"""
        
        # Handle different output types
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            # Use pooler output if available (BERT-style models)
            return outputs.pooler_output
            
        elif hasattr(outputs, 'last_hidden_state'):
            # Use last hidden state with pooling
            hidden_states = outputs.last_hidden_state
            
            if self.pooling_strategy == 'cls':
                # Use CLS token (first token)
                return hidden_states[:, 0, :]
            elif self.pooling_strategy == 'mean':
                # Mean pooling over sequence
                return hidden_states.mean(dim=1)
            elif self.pooling_strategy == 'max':
                # Max pooling over sequence
                return hidden_states.max(dim=1)[0]
            else:
                # Default to CLS token
                return hidden_states[:, 0, :]
                
        elif isinstance(outputs, tuple):
            # Handle tuple outputs
            features = outputs[0]
            if features.dim() > 2:
                # Pool over sequence dimension
                return features.mean(dim=1)
            return features
            
        elif isinstance(outputs, torch.Tensor):
            # Direct tensor output
            if outputs.dim() > 2:
                return outputs.mean(dim=1)
            return outputs
            
        else:
            raise ValueError(f"Unknown output type: {type(outputs)}")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Extract features from inputs
        
        Args:
            input_ids: Token IDs for text input
            attention_mask: Attention mask for text input
            pixel_values: Pixel values for image input
            
        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        # Prepare inputs
        model_inputs = {}
        
        if input_ids is not None:
            model_inputs['input_ids'] = input_ids
        if attention_mask is not None:
            model_inputs['attention_mask'] = attention_mask
        if pixel_values is not None:
            model_inputs['pixel_values'] = pixel_values
        
        # Add any additional kwargs
        model_inputs.update(kwargs)
        
        # Forward pass through model
        if self.freeze and self.training:
            # Use no_grad for frozen model during training
            with torch.no_grad():
                outputs = self.model(**model_inputs)
        else:
            outputs = self.model(**model_inputs)
        
        # Extract features
        features = self._extract_features_from_outputs(outputs)
        
        return features
    
    def extract_features_batch(
        self,
        dataloader,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features for entire dataset
        
        Args:
            dataloader: DataLoader for the dataset
            device: Device to run on
            
        Returns:
            Tuple of (features, labels)
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Extract features
                features = self.forward(**{k: v for k, v in batch.items() if k != 'labels'})
                
                all_features.append(features.cpu())
                
                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu())
        
        # Concatenate all features
        features_tensor = torch.cat(all_features, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0) if all_labels else None
        
        return features_tensor, labels_tensor
    
    def freeze(self):
        """Freeze all parameters"""
        self.freeze = True
        self._freeze_parameters()
    
    def unfreeze(self):
        """Unfreeze all parameters"""
        self.freeze = False
        self._unfreeze_parameters()
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    
    def print_info(self):
        """Print feature extractor information"""
        param_counts = self.get_parameter_count()
        
        print(f"FeatureExtractor Info:")
        print(f"  Model: {self.model_name}")
        print(f"  Type: {self.model_type}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Pooling strategy: {self.pooling_strategy}")
        print(f"  Frozen: {self.freeze}")
        print(f"  Parameters:")
        print(f"    Total: {param_counts['total']:,}")
        print(f"    Trainable: {param_counts['trainable']:,}")
        print(f"    Frozen: {param_counts['frozen']:,}")


# Specialized feature extractors for different modalities

class TextFeatureExtractor(FeatureExtractor):
    """Specialized feature extractor for text models"""
    
    def __init__(self, model, model_name, **kwargs):
        super().__init__(model, model_name, **kwargs)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def encode_texts(
        self,
        texts: List[str],
        max_length: int = 128,
        batch_size: int = 32
    ) -> torch.Tensor:
        """Encode list of texts to features"""
        
        all_features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Extract features
            with torch.no_grad():
                features = self.forward(**inputs)
                all_features.append(features)
        
        return torch.cat(all_features, dim=0)


class ImageFeatureExtractor(FeatureExtractor):
    """Specialized feature extractor for image models"""
    
    def __init__(self, model, model_name, **kwargs):
        super().__init__(model, model_name, **kwargs)
        
        # Initialize image processor
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        except:
            self.processor = None
    
    def encode_images(
        self,
        images: List,
        batch_size: int = 32
    ) -> torch.Tensor:
        """Encode list of images to features"""
        
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Process images
            if self.processor:
                inputs = self.processor(batch_images, return_tensors='pt')
            else:
                # Assume images are already tensors
                inputs = {'pixel_values': torch.stack(batch_images)}
            
            # Extract features
            with torch.no_grad():
                features = self.forward(**inputs)
                all_features.append(features)
        
        return torch.cat(all_features, dim=0)


# Example usage and testing
if __name__ == "__main__":
    # Test text feature extractor
    text_extractor = FeatureExtractor.from_pretrained(
        'bert-base-uncased',
        freeze=True,
        pooling_strategy='cls'
    )
    
    text_extractor.print_info()
    
    # Test with dummy input
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)
    
    features = text_extractor(input_ids=input_ids, attention_mask=attention_mask)
    print(f"Text features shape: {features.shape}")
    
    # Test image feature extractor
    try:
        image_extractor = FeatureExtractor.from_pretrained(
            'microsoft/resnet-50',
            freeze=True
        )
        
        image_extractor.print_info()
        
        # Test with dummy input
        pixel_values = torch.randn(2, 3, 224, 224)
        features = image_extractor(pixel_values=pixel_values)
        print(f"Image features shape: {features.shape}")
        
    except Exception as e:
        print(f"Image model test failed: {e}")
    
    print("FeatureExtractor test completed successfully!")
