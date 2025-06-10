"""
Comprehensive Image Captioning Example using Cross-Modal Training

This example demonstrates how to use cross-modal models for image captioning,
including training, inference, and evaluation.
"""

import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import requests
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.clip_model import CLIPModel
from data.multimodal_dataset import COCOCaptionsDataset, create_multimodal_dataloader
from training.cross_modal_trainer import CrossModalTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageCaptioningModel(nn.Module):
    """
    Image captioning model using BLIP architecture
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        max_length: int = 50,
        num_beams: int = 5,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.num_beams = num_beams
        
        # Load pre-trained BLIP model
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        logger.info(f"Loaded image captioning model: {model_name}")
    
    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        """Forward pass for training"""
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate_caption(
        self,
        image: Image.Image,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate caption for a single image
        
        Args:
            image: PIL Image
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            
        Returns:
            Generated caption string
        """
        max_length = max_length or self.max_length
        num_beams = num_beams or self.num_beams
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.model.device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                **kwargs
            )
        
        # Decode caption
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption
    
    def generate_captions_batch(
        self,
        images: List[Image.Image],
        **kwargs
    ) -> List[str]:
        """
        Generate captions for a batch of images
        
        Args:
            images: List of PIL Images
            **kwargs: Generation parameters
            
        Returns:
            List of generated captions
        """
        captions = []
        for image in images:
            caption = self.generate_caption(image, **kwargs)
            captions.append(caption)
        
        return captions


class ImageCaptioningPipeline:
    """
    Complete pipeline for image captioning
    """
    
    def __init__(
        self,
        model: Optional[ImageCaptioningModel] = None,
        model_name: str = "Salesforce/blip-image-captioning-base"
    ):
        self.model = model or ImageCaptioningModel(model_name)
        self.model.eval()
        
        logger.info("Image captioning pipeline initialized")
    
    @classmethod
    def from_pretrained(cls, model_name: str) -> 'ImageCaptioningPipeline':
        """Load pre-trained pipeline"""
        model = ImageCaptioningModel(model_name)
        return cls(model)
    
    def __call__(
        self,
        image_input: Any,
        **kwargs
    ) -> str:
        """
        Generate caption for image input
        
        Args:
            image_input: Image path, URL, or PIL Image
            **kwargs: Generation parameters
            
        Returns:
            Generated caption
        """
        # Load image
        image = self._load_image(image_input)
        
        # Generate caption
        caption = self.model.generate_caption(image, **kwargs)
        
        return caption
    
    def _load_image(self, image_input: Any) -> Image.Image:
        """Load image from various input types"""
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                # URL
                response = requests.get(image_input, stream=True)
                image = Image.open(response.raw).convert('RGB')
            else:
                # File path
                image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        return image
    
    def caption_multiple_images(
        self,
        image_inputs: List[Any],
        **kwargs
    ) -> List[str]:
        """Caption multiple images"""
        captions = []
        for image_input in image_inputs:
            try:
                caption = self(image_input, **kwargs)
                captions.append(caption)
            except Exception as e:
                logger.error(f"Failed to caption image {image_input}: {e}")
                captions.append("")
        
        return captions
    
    def visualize_captions(
        self,
        image_inputs: List[Any],
        captions: Optional[List[str]] = None,
        figsize: tuple = (15, 10),
        cols: int = 3
    ):
        """
        Visualize images with their captions
        
        Args:
            image_inputs: List of image inputs
            captions: List of captions (generated if None)
            figsize: Figure size
            cols: Number of columns in the grid
        """
        if captions is None:
            captions = self.caption_multiple_images(image_inputs)
        
        num_images = len(image_inputs)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (image_input, caption) in enumerate(zip(image_inputs, captions)):
            row = i // cols
            col = i % cols
            
            # Load and display image
            image = self._load_image(image_input)
            axes[row, col].imshow(image)
            axes[row, col].set_title(caption, fontsize=10, wrap=True)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()


class CaptioningEvaluator:
    """
    Evaluator for image captioning models
    """
    
    def __init__(self):
        # Import evaluation metrics
        try:
            from pycocotools.coco import COCO
            from pycocoevalcap.eval import COCOEvalCap
            self.coco_available = True
        except ImportError:
            logger.warning("COCO evaluation tools not available")
            self.coco_available = False
        
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.translate.meteor_score import meteor_score
            nltk.download('wordnet', quiet=True)
            self.nltk_available = True
        except ImportError:
            logger.warning("NLTK not available for evaluation")
            self.nltk_available = False
    
    def evaluate_captions(
        self,
        generated_captions: List[str],
        reference_captions: List[List[str]],
        metrics: List[str] = ['bleu', 'meteor', 'rouge']
    ) -> Dict[str, float]:
        """
        Evaluate generated captions against references
        
        Args:
            generated_captions: List of generated captions
            reference_captions: List of reference caption lists
            metrics: Metrics to compute
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        if 'bleu' in metrics and self.nltk_available:
            bleu_scores = self._compute_bleu_scores(generated_captions, reference_captions)
            results.update(bleu_scores)
        
        if 'meteor' in metrics and self.nltk_available:
            meteor_score = self._compute_meteor_score(generated_captions, reference_captions)
            results['meteor'] = meteor_score
        
        if 'rouge' in metrics:
            rouge_scores = self._compute_rouge_scores(generated_captions, reference_captions)
            results.update(rouge_scores)
        
        return results
    
    def _compute_bleu_scores(
        self,
        generated_captions: List[str],
        reference_captions: List[List[str]]
    ) -> Dict[str, float]:
        """Compute BLEU scores"""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smoothing = SmoothingFunction().method1
        bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
        
        for gen_cap, ref_caps in zip(generated_captions, reference_captions):
            gen_tokens = gen_cap.lower().split()
            ref_tokens_list = [ref.lower().split() for ref in ref_caps]
            
            # Compute BLEU scores for different n-grams
            for n in range(1, 5):
                weights = [1.0/n] * n + [0.0] * (4-n)
                score = sentence_bleu(
                    ref_tokens_list,
                    gen_tokens,
                    weights=weights,
                    smoothing_function=smoothing
                )
                bleu_scores[f'bleu_{n}'].append(score)
        
        # Average scores
        return {k: np.mean(v) for k, v in bleu_scores.items()}
    
    def _compute_meteor_score(
        self,
        generated_captions: List[str],
        reference_captions: List[List[str]]
    ) -> float:
        """Compute METEOR score"""
        from nltk.translate.meteor_score import meteor_score
        
        scores = []
        for gen_cap, ref_caps in zip(generated_captions, reference_captions):
            # METEOR expects a single reference, so we use the first one
            ref_cap = ref_caps[0] if ref_caps else ""
            score = meteor_score([ref_cap.lower().split()], gen_cap.lower().split())
            scores.append(score)
        
        return np.mean(scores)
    
    def _compute_rouge_scores(
        self,
        generated_captions: List[str],
        reference_captions: List[List[str]]
    ) -> Dict[str, float]:
        """Compute ROUGE scores"""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for gen_cap, ref_caps in zip(generated_captions, reference_captions):
                # Use the first reference for ROUGE
                ref_cap = ref_caps[0] if ref_caps else ""
                scores = scorer.score(ref_cap, gen_cap)
                
                for metric in rouge_scores:
                    rouge_scores[metric].append(scores[metric].fmeasure)
            
            return {k: np.mean(v) for k, v in rouge_scores.items()}
            
        except ImportError:
            logger.warning("rouge-score package not available")
            return {}


def main():
    """
    Main function to demonstrate image captioning
    """
    logger.info("Starting image captioning example")
    
    # Initialize captioning pipeline
    pipeline = ImageCaptioningPipeline.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Example images (using URLs for demonstration)
    example_images = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # Cats on couch
        "http://images.cocodataset.org/val2017/000000397133.jpg",  # Baseball player
        "http://images.cocodataset.org/val2017/000000037777.jpg",  # Traffic scene
        "http://images.cocodataset.org/val2017/000000252219.jpg",  # Kitchen scene
    ]
    
    print("\n" + "="*60)
    print("IMAGE CAPTIONING EXAMPLES")
    print("="*60)
    
    # Generate captions for example images
    for i, image_url in enumerate(example_images):
        try:
            print(f"\nImage {i+1}: {image_url}")
            
            # Generate caption
            caption = pipeline(image_url)
            print(f"Generated Caption: {caption}")
            
            # Generate multiple captions with different parameters
            captions_diverse = []
            for temp in [0.7, 1.0, 1.3]:
                cap = pipeline(
                    image_url,
                    do_sample=True,
                    temperature=temp,
                    max_length=30
                )
                captions_diverse.append(cap)
            
            print(f"Diverse Captions:")
            for j, cap in enumerate(captions_diverse):
                print(f"  {j+1}. {cap}")
            
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")
    
    # Visualize results
    print(f"\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    try:
        # Generate captions for visualization
        captions = pipeline.caption_multiple_images(example_images)
        
        # Display images with captions
        pipeline.visualize_captions(example_images, captions)
        
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Evaluation example
    print(f"\n" + "="*60)
    print("EVALUATION EXAMPLE")
    print("="*60)
    
    # Example evaluation
    evaluator = CaptioningEvaluator()
    
    # Mock data for evaluation
    generated_captions = [
        "a cat sitting on a couch",
        "a baseball player swinging a bat",
        "cars driving on a busy street",
        "a kitchen with white cabinets"
    ]
    
    reference_captions = [
        ["two cats sitting on a couch", "cats relaxing on furniture"],
        ["baseball player at bat", "man playing baseball"],
        ["traffic on city street", "busy urban road with cars"],
        ["modern kitchen interior", "white kitchen with appliances"]
    ]
    
    if evaluator.nltk_available:
        metrics = evaluator.evaluate_captions(
            generated_captions,
            reference_captions,
            metrics=['bleu', 'meteor']
        )
        
        print("Evaluation Metrics:")
        for metric, score in metrics.items():
            print(f"  {metric.upper()}: {score:.4f}")
    else:
        print("Evaluation metrics not available (missing dependencies)")
    
    # Training example (simplified)
    print(f"\n" + "="*60)
    print("TRAINING EXAMPLE")
    print("="*60)
    
    print("Note: This is a simplified training example.")
    print("For full training, you would need:")
    print("1. Large-scale dataset (COCO, Flickr30k)")
    print("2. Proper data preprocessing")
    print("3. Distributed training setup")
    print("4. Comprehensive evaluation")
    
    logger.info("Image captioning example completed")


if __name__ == "__main__":
    main()
