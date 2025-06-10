"""
Multi-Modal Dataset Implementation for Cross-Modal Training
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, AutoTokenizer
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """
    Generic multi-modal dataset supporting text-image, text-audio, and text-video pairs
    """
    
    def __init__(
        self,
        data_path: str,
        modalities: List[str] = ['text', 'image'],
        processor: Optional[Any] = None,
        max_text_length: int = 77,
        image_size: Tuple[int, int] = (224, 224),
        split: str = 'train',
        transform: Optional[Any] = None,
        **kwargs
    ):
        self.data_path = Path(data_path)
        self.modalities = modalities
        self.processor = processor
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.split = split
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
        # Initialize processor if not provided
        if self.processor is None:
            self._initialize_processor()
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
        logger.info(f"Modalities: {modalities}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from various formats"""
        data_file = self.data_path / f"{self.split}.json"
        
        if data_file.exists():
            return self._load_json_data(data_file)
        
        # Try CSV format
        csv_file = self.data_path / f"{self.split}.csv"
        if csv_file.exists():
            return self._load_csv_data(csv_file)
        
        # Try directory structure
        return self._load_directory_data()
    
    def _load_json_data(self, data_file: Path) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Handle different JSON structures
            if 'annotations' in data:
                return data['annotations']
            elif 'data' in data:
                return data['data']
            else:
                return [data]
        
        return data
    
    def _load_csv_data(self, csv_file: Path) -> List[Dict[str, Any]]:
        """Load data from CSV file"""
        df = pd.read_csv(csv_file)
        return df.to_dict('records')
    
    def _load_directory_data(self) -> List[Dict[str, Any]]:
        """Load data from directory structure"""
        data = []
        
        # Look for image-text pairs
        if 'text' in self.modalities and 'image' in self.modalities:
            image_dir = self.data_path / 'images' / self.split
            text_dir = self.data_path / 'texts' / self.split
            
            if image_dir.exists() and text_dir.exists():
                for img_file in image_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        text_file = text_dir / f"{img_file.stem}.txt"
                        if text_file.exists():
                            with open(text_file, 'r') as f:
                                text = f.read().strip()
                            
                            data.append({
                                'image_path': str(img_file),
                                'text': text,
                                'id': img_file.stem
                            })
        
        return data
    
    def _initialize_processor(self):
        """Initialize default processor"""
        if 'text' in self.modalities and 'image' in self.modalities:
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif 'text' in self.modalities:
            self.processor = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        sample = self.data[idx].copy()
        processed_sample = {}
        
        # Process text
        if 'text' in self.modalities and 'text' in sample:
            processed_sample.update(self._process_text(sample['text']))
        
        # Process image
        if 'image' in self.modalities:
            if 'image_path' in sample:
                processed_sample.update(self._process_image(sample['image_path']))
            elif 'image' in sample:
                processed_sample.update(self._process_image(sample['image']))
        
        # Process audio
        if 'audio' in self.modalities and 'audio_path' in sample:
            processed_sample.update(self._process_audio(sample['audio_path']))
        
        # Add metadata
        processed_sample['sample_id'] = sample.get('id', idx)
        
        return processed_sample
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text input"""
        if hasattr(self.processor, 'tokenizer'):
            # CLIP processor
            inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length
            )
        else:
            # Regular tokenizer
            inputs = self.processor(
                text,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length
            )
        
        # Remove batch dimension
        return {k: v.squeeze(0) for k, v in inputs.items()}
    
    def _process_image(self, image_path: Union[str, Image.Image]) -> Dict[str, torch.Tensor]:
        """Process image input"""
        # Load image if path provided
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Apply custom transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Process with processor
        if hasattr(self.processor, 'image_processor'):
            # CLIP processor
            inputs = self.processor.image_processor(
                image,
                return_tensors="pt"
            )
        else:
            # Custom processing
            inputs = self._default_image_processing(image)
        
        # Remove batch dimension
        return {k: v.squeeze(0) for k, v in inputs.items()}
    
    def _default_image_processing(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Default image processing"""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        pixel_values = transform(image)
        return {'pixel_values': pixel_values}
    
    def _process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """Process audio input"""
        import librosa
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        return {'audio_values': audio_tensor}


class COCOCaptionsDataset(MultiModalDataset):
    """
    COCO Captions dataset for image captioning and retrieval
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        year: str = '2017',
        **kwargs
    ):
        self.data_root = Path(data_root)
        self.year = year
        
        # Set paths
        self.image_dir = self.data_root / f"{split}{year}"
        self.annotation_file = self.data_root / "annotations" / f"captions_{split}{year}.json"
        
        super().__init__(
            data_path=str(self.data_root),
            modalities=['text', 'image'],
            split=split,
            **kwargs
        )
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load COCO annotations"""
        with open(self.annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image id to filename mapping
        id_to_filename = {
            img['id']: img['file_name']
            for img in coco_data['images']
        }
        
        # Process annotations
        data = []
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in id_to_filename:
                data.append({
                    'id': ann['id'],
                    'image_id': image_id,
                    'image_path': str(self.image_dir / id_to_filename[image_id]),
                    'text': ann['caption'],
                    'caption': ann['caption']
                })
        
        return data


class Flickr30kDataset(MultiModalDataset):
    """
    Flickr30k dataset for image captioning and retrieval
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        **kwargs
    ):
        self.data_root = Path(data_root)
        
        super().__init__(
            data_path=str(self.data_root),
            modalities=['text', 'image'],
            split=split,
            **kwargs
        )
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load Flickr30k data"""
        # Load split file
        split_file = self.data_root / f"{self.split}.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                image_files = [line.strip() for line in f]
        else:
            # Use all images in directory
            image_dir = self.data_root / "images"
            image_files = [f.name for f in image_dir.glob("*.jpg")]
        
        # Load captions
        caption_file = self.data_root / "results.csv"
        if caption_file.exists():
            captions_df = pd.read_csv(caption_file, sep='|')
            captions_df.columns = ['image', 'caption_number', 'caption']
        else:
            raise FileNotFoundError(f"Caption file not found: {caption_file}")
        
        # Process data
        data = []
        for img_file in image_files:
            img_captions = captions_df[captions_df['image'] == img_file]
            
            for _, row in img_captions.iterrows():
                data.append({
                    'id': f"{img_file}_{row['caption_number']}",
                    'image_path': str(self.data_root / "images" / img_file),
                    'text': row['caption'].strip(),
                    'caption': row['caption'].strip()
                })
        
        return data


class VQADataset(MultiModalDataset):
    """
    Visual Question Answering dataset
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        version: str = 'v2',
        **kwargs
    ):
        self.data_root = Path(data_root)
        self.version = version
        
        super().__init__(
            data_path=str(self.data_root),
            modalities=['text', 'image'],
            split=split,
            **kwargs
        )
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load VQA data"""
        # Load questions
        question_file = self.data_root / f"{self.version}_{self.split}_questions.json"
        with open(question_file, 'r') as f:
            questions_data = json.load(f)
        
        # Load annotations (for train/val)
        annotations_data = None
        if self.split in ['train', 'val']:
            annotation_file = self.data_root / f"{self.version}_{self.split}_annotations.json"
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    annotations_data = json.load(f)
        
        # Create question id to annotation mapping
        id_to_annotation = {}
        if annotations_data:
            id_to_annotation = {
                ann['question_id']: ann
                for ann in annotations_data['annotations']
            }
        
        # Process questions
        data = []
        for question in questions_data['questions']:
            question_id = question['question_id']
            image_id = question['image_id']
            
            # Format image path (assuming COCO format)
            image_filename = f"COCO_{self.split}2014_{image_id:012d}.jpg"
            image_path = self.data_root / "images" / image_filename
            
            sample = {
                'id': question_id,
                'image_id': image_id,
                'image_path': str(image_path),
                'question': question['question'],
                'text': question['question']  # For compatibility
            }
            
            # Add answer if available
            if question_id in id_to_annotation:
                ann = id_to_annotation[question_id]
                sample['answer'] = ann['multiple_choice_answer']
                sample['answers'] = [ans['answer'] for ans in ann['answers']]
            
            data.append(sample)
        
        return data


def create_multimodal_dataloader(
    dataset: MultiModalDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Optional[Any] = None,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for multi-modal dataset
    """
    if collate_fn is None:
        collate_fn = multimodal_collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for multi-modal batches
    """
    collated = {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        values = [sample[key] for sample in batch if key in sample]
        
        if len(values) == 0:
            continue
        
        # Handle different data types
        if isinstance(values[0], torch.Tensor):
            # Stack tensors
            collated[key] = torch.stack(values)
        elif isinstance(values[0], str):
            # Keep strings as list
            collated[key] = values
        elif isinstance(values[0], (int, float)):
            # Convert to tensor
            collated[key] = torch.tensor(values)
        else:
            # Keep as list for other types
            collated[key] = values
    
    return collated


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data
    import tempfile
    import os
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy data
        data_dir = Path(temp_dir)
        
        # Create images directory
        images_dir = data_dir / "images" / "train"
        images_dir.mkdir(parents=True)
        
        # Create texts directory
        texts_dir = data_dir / "texts" / "train"
        texts_dir.mkdir(parents=True)
        
        # Create dummy image and text files
        for i in range(5):
            # Create dummy image (just touch the file)
            img_file = images_dir / f"image_{i}.jpg"
            img_file.touch()
            
            # Create dummy text
            text_file = texts_dir / f"image_{i}.txt"
            with open(text_file, 'w') as f:
                f.write(f"This is a description of image {i}")
        
        # Test dataset
        dataset = MultiModalDataset(
            data_path=str(data_dir),
            modalities=['text', 'image'],
            split='train'
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test dataloader
        dataloader = create_multimodal_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        
        # Test batch
        for batch in dataloader:
            print("Batch keys:", batch.keys())
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.shape}")
                else:
                    print(f"{key}: {type(value)} (length: {len(value)})")
            break
        
        print("Multi-modal dataset test completed successfully!")
