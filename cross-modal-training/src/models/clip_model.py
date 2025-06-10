"""
CLIP (Contrastive Language-Image Pre-training) Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from transformers import (
    CLIPModel as HFCLIPModel,
    CLIPProcessor,
    CLIPTextModel,
    CLIPVisionModel,
    CLIPConfig
)
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CLIPModel(nn.Module):
    """
    Enhanced CLIP model with additional functionality for cross-modal training
    """
    
    def __init__(
        self,
        config: Optional[CLIPConfig] = None,
        text_model_name: str = "openai/clip-vit-base-patch32",
        vision_model_name: str = "openai/clip-vit-base-patch32",
        projection_dim: int = 512,
        temperature: float = 0.07,
        **kwargs
    ):
        super().__init__()
        
        self.config = config or CLIPConfig.from_pretrained(text_model_name)
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Load pre-trained components
        self.text_model = CLIPTextModel.from_pretrained(text_model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        
        # Projection layers
        self.text_projection = nn.Linear(
            self.text_model.config.hidden_size,
            projection_dim,
            bias=False
        )
        self.visual_projection = nn.Linear(
            self.vision_model.config.hidden_size,
            projection_dim,
            bias=False
        )
        
        # Initialize projections
        self._initialize_projections()
        
        # Processor for input preprocessing
        self.processor = CLIPProcessor.from_pretrained(text_model_name)
        
        logger.info(f"Initialized CLIP model with projection dim: {projection_dim}")
    
    def _initialize_projections(self):
        """Initialize projection layers"""
        nn.init.normal_(self.text_projection.weight, std=0.02)
        nn.init.normal_(self.visual_projection.weight, std=0.02)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs
    ) -> 'CLIPModel':
        """Load pre-trained CLIP model"""
        try:
            # Try loading from Hugging Face
            hf_model = HFCLIPModel.from_pretrained(model_name_or_path)
            
            # Create our model instance
            model = cls(
                config=hf_model.config,
                text_model_name=model_name_or_path,
                vision_model_name=model_name_or_path,
                **kwargs
            )
            
            # Copy weights from pre-trained model
            model.text_model.load_state_dict(hf_model.text_model.state_dict())
            model.vision_model.load_state_dict(hf_model.vision_model.state_dict())
            model.text_projection.load_state_dict(hf_model.text_projection.state_dict())
            model.visual_projection.load_state_dict(hf_model.visual_projection.state_dict())
            
            logger.info(f"Loaded pre-trained CLIP model from {model_name_or_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load pre-trained model: {e}")
            raise
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text inputs to embeddings
        
        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask for text
            normalize: Whether to normalize embeddings
            
        Returns:
            Text embeddings
        """
        # Get text features
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use pooled output (CLS token)
        text_features = text_outputs.pooler_output
        
        # Project to shared space
        text_embeds = self.text_projection(text_features)
        
        # Normalize if requested
        if normalize:
            text_embeds = F.normalize(text_embeds, dim=-1)
        
        return text_embeds
    
    def encode_image(
        self,
        pixel_values: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode image inputs to embeddings
        
        Args:
            pixel_values: Image pixel values
            normalize: Whether to normalize embeddings
            
        Returns:
            Image embeddings
        """
        # Get image features
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        
        # Use pooled output
        image_features = vision_outputs.pooler_output
        
        # Project to shared space
        image_embeds = self.visual_projection(image_features)
        
        # Normalize if requested
        if normalize:
            image_embeds = F.normalize(image_embeds, dim=-1)
        
        return image_embeds
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLIP model
        
        Args:
            input_ids: Text token IDs
            pixel_values: Image pixel values
            attention_mask: Text attention mask
            return_loss: Whether to compute contrastive loss
            
        Returns:
            Dictionary containing embeddings and optionally loss
        """
        outputs = {}
        
        # Encode text if provided
        if input_ids is not None:
            text_embeds = self.encode_text(input_ids, attention_mask)
            outputs['text_embeds'] = text_embeds
        
        # Encode images if provided
        if pixel_values is not None:
            image_embeds = self.encode_image(pixel_values)
            outputs['image_embeds'] = image_embeds
        
        # Compute similarity and loss if both modalities present
        if 'text_embeds' in outputs and 'image_embeds' in outputs:
            # Compute similarity matrices
            logits_per_text = text_embeds @ image_embeds.T / self.temperature
            logits_per_image = image_embeds @ text_embeds.T / self.temperature
            
            outputs['logits_per_text'] = logits_per_text
            outputs['logits_per_image'] = logits_per_image
            
            # Compute contrastive loss if requested
            if return_loss:
                batch_size = text_embeds.shape[0]
                labels = torch.arange(batch_size, device=text_embeds.device)
                
                text_loss = F.cross_entropy(logits_per_text, labels)
                image_loss = F.cross_entropy(logits_per_image, labels)
                
                loss = (text_loss + image_loss) / 2
                outputs['loss'] = loss
                outputs['text_loss'] = text_loss
                outputs['image_loss'] = image_loss
        
        return outputs
    
    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        """
        Get text features from raw text strings
        
        Args:
            texts: List of text strings
            
        Returns:
            Text embeddings tensor
        """
        # Process texts
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        
        # Move to model device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
        
        # Encode
        with torch.no_grad():
            text_embeds = self.encode_text(**inputs)
        
        return text_embeds
    
    def get_image_features(self, images: List[Union[Image.Image, str]]) -> torch.Tensor:
        """
        Get image features from PIL images or image paths
        
        Args:
            images: List of PIL images or image paths
            
        Returns:
            Image embeddings tensor
        """
        # Load images if paths provided
        if isinstance(images[0], str):
            images = [Image.open(img_path).convert('RGB') for img_path in images]
        
        # Process images
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # Move to model device
        device = next(self.parameters()).device
        pixel_values = inputs['pixel_values'].to(device)
        
        # Encode
        with torch.no_grad():
            image_embeds = self.encode_image(pixel_values)
        
        return image_embeds
    
    def compute_similarity(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute similarity between text and image embeddings
        
        Args:
            text_embeds: Text embeddings
            image_embeds: Image embeddings
            temperature: Temperature for scaling (uses model default if None)
            
        Returns:
            Similarity matrix
        """
        temp = temperature or self.temperature
        return text_embeds @ image_embeds.T / temp
    
    def zero_shot_classify(
        self,
        images: List[Union[Image.Image, str]],
        candidate_labels: List[str],
        hypothesis_template: str = "a photo of a {}"
    ) -> List[Dict[str, float]]:
        """
        Perform zero-shot image classification
        
        Args:
            images: List of images to classify
            candidate_labels: List of candidate class labels
            hypothesis_template: Template for creating text hypotheses
            
        Returns:
            List of classification results
        """
        # Create text hypotheses
        text_hypotheses = [hypothesis_template.format(label) for label in candidate_labels]
        
        # Get embeddings
        image_embeds = self.get_image_features(images)
        text_embeds = self.get_text_features(text_hypotheses)
        
        # Compute similarities
        similarities = self.compute_similarity(image_embeds, text_embeds)
        
        # Convert to probabilities
        probs = F.softmax(similarities, dim=-1)
        
        # Format results
        results = []
        for i, image_probs in enumerate(probs):
            result = {
                label: float(prob)
                for label, prob in zip(candidate_labels, image_probs)
            }
            results.append(result)
        
        return results
    
    def text_to_image_retrieval(
        self,
        query_text: str,
        candidate_images: List[Union[Image.Image, str]],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Retrieve most similar images for a text query
        
        Args:
            query_text: Text query
            candidate_images: List of candidate images
            top_k: Number of top results to return
            
        Returns:
            List of (image_index, similarity_score) tuples
        """
        # Get embeddings
        text_embeds = self.get_text_features([query_text])
        image_embeds = self.get_image_features(candidate_images)
        
        # Compute similarities
        similarities = self.compute_similarity(text_embeds, image_embeds)
        similarities = similarities.squeeze(0)  # Remove batch dimension
        
        # Get top-k results
        top_k = min(top_k, len(candidate_images))
        top_indices = torch.topk(similarities, top_k).indices
        top_scores = similarities[top_indices]
        
        # Format results
        results = [
            (int(idx), float(score))
            for idx, score in zip(top_indices, top_scores)
        ]
        
        return results
    
    def image_to_text_retrieval(
        self,
        query_image: Union[Image.Image, str],
        candidate_texts: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Retrieve most similar texts for an image query
        
        Args:
            query_image: Query image
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (text_index, similarity_score) tuples
        """
        # Get embeddings
        image_embeds = self.get_image_features([query_image])
        text_embeds = self.get_text_features(candidate_texts)
        
        # Compute similarities
        similarities = self.compute_similarity(image_embeds, text_embeds)
        similarities = similarities.squeeze(0)  # Remove batch dimension
        
        # Get top-k results
        top_k = min(top_k, len(candidate_texts))
        top_indices = torch.topk(similarities, top_k).indices
        top_scores = similarities[top_indices]
        
        # Format results
        results = [
            (int(idx), float(score))
            for idx, score in zip(top_indices, top_scores)
        ]
        
        return results
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save processor
        self.processor.save_pretrained(save_directory)
        
        logger.info(f"Model saved to {save_directory}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Test zero-shot classification
    from PIL import Image
    import requests
    
    # Load test image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Classify
    labels = ["cat", "dog", "car", "airplane"]
    results = model.zero_shot_classify([image], labels)
    
    print("Zero-shot classification results:")
    for label, score in results[0].items():
        print(f"{label}: {score:.3f}")
    
    # Test text-to-image retrieval
    query = "a cat sitting on a couch"
    retrieval_results = model.text_to_image_retrieval(query, [image])
    
    print(f"\nText-to-image retrieval for '{query}':")
    for idx, score in retrieval_results:
        print(f"Image {idx}: {score:.3f}")
    
    print("CLIP model test completed successfully!")
