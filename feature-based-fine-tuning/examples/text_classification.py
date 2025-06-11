"""
Comprehensive Text Classification Example using Feature-Based Fine-Tuning

This example demonstrates how to use feature-based fine-tuning for text classification,
comparing it with full fine-tuning and showing the efficiency gains.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import time
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.feature_based_model import FeatureBasedModel
from training.trainer import FeatureBasedTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    """
    Dataset for text classification tasks
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TextClassificationPipeline:
    """
    Complete pipeline for text classification with feature-based fine-tuning
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 3,
        classifier_type: str = 'linear',
        freeze_backbone: bool = True,
        max_length: int = 128
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.classifier_type = classifier_type
        self.freeze_backbone = freeze_backbone
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = FeatureBasedModel.from_pretrained(
            model_name,
            num_classes=num_classes,
            classifier_type=classifier_type,
            freeze_backbone=freeze_backbone
        )
        
        # Training components
        self.trainer = None
        self.training_history = []
        
        logger.info(f"Initialized TextClassificationPipeline with {model_name}")
        self.model.print_parameter_status()
    
    def prepare_data(
        self,
        train_texts: List[str],
        train_labels: List[int],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[int]] = None,
        batch_size: int = 16
    ):
        """Prepare datasets and dataloaders"""
        
        # Create train dataset
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Create eval dataset if provided
        self.eval_dataloader = None
        if eval_texts is not None and eval_labels is not None:
            eval_dataset = TextClassificationDataset(
                eval_texts, eval_labels, self.tokenizer, self.max_length
            )
            
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
        
        logger.info(f"Prepared data: {len(train_dataset)} train samples")
        if self.eval_dataloader:
            logger.info(f"Prepared data: {len(eval_dataset)} eval samples")
    
    def train(
        self,
        num_epochs: int = 5,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        save_dir: Optional[str] = None,
        **trainer_kwargs
    ):
        """Train the model"""
        
        # Training configuration
        config = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'max_grad_norm': 1.0,
            **trainer_kwargs
        }
        
        # Initialize trainer
        self.trainer = FeatureBasedTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            config=config
        )
        
        # Start training
        start_time = time.time()
        
        self.trainer.train(
            num_epochs=num_epochs,
            save_dir=save_dir,
            early_stopping_patience=3
        )
        
        training_time = time.time() - start_time
        self.training_history = self.trainer.training_history
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'training_time': training_time,
            'best_metric': self.trainer.best_eval_metric,
            'history': self.training_history
        }
    
    def predict(self, texts: List[str], batch_size: int = 16) -> Dict[str, Any]:
        """Make predictions on new texts"""
        
        # Create dataset
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        dataset = TextClassificationDataset(
            texts, dummy_labels, self.tokenizer, self.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Make predictions
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Get predictions
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return {
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
    
    def evaluate_detailed(
        self,
        test_texts: List[str],
        test_labels: List[int],
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detailed evaluation with metrics and visualizations"""
        
        # Make predictions
        results = self.predict(test_texts)
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
        precision = precision_score(test_labels, predictions, average='weighted')
        recall = recall_score(test_labels, predictions, average='weighted')
        
        # Classification report
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(self.num_classes)]
        
        report = classification_report(
            test_labels, predictions,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def plot_results(self, evaluation_results: Dict[str, Any], class_names: Optional[List[str]] = None):
        """Plot evaluation results"""
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Training history
        if self.training_history:
            epochs = [h['epoch'] for h in self.training_history]
            train_losses = [h.get('train_loss', 0) for h in self.training_history]
            eval_losses = [h.get('eval_loss', 0) for h in self.training_history if 'eval_loss' in h]
            
            axes[0, 0].plot(epochs, train_losses, label='Train Loss', marker='o')
            if eval_losses:
                eval_epochs = [h['epoch'] for h in self.training_history if 'eval_loss' in h]
                axes[0, 0].plot(eval_epochs, eval_losses, label='Eval Loss', marker='s')
            
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 2. Accuracy over epochs
        if self.training_history:
            train_accs = [h.get('train_accuracy', 0) for h in self.training_history if 'train_accuracy' in h]
            eval_accs = [h.get('eval_accuracy', 0) for h in self.training_history if 'eval_accuracy' in h]
            
            if train_accs:
                train_acc_epochs = [h['epoch'] for h in self.training_history if 'train_accuracy' in h]
                axes[0, 1].plot(train_acc_epochs, train_accs, label='Train Accuracy', marker='o')
            
            if eval_accs:
                eval_acc_epochs = [h['epoch'] for h in self.training_history if 'eval_accuracy' in h]
                axes[0, 1].plot(eval_acc_epochs, eval_accs, label='Eval Accuracy', marker='s')
            
            axes[0, 1].set_title('Training Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 3. Confusion Matrix
        cm = evaluation_results['confusion_matrix']
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(len(cm))]
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Metrics bar plot
        metrics = {
            'Accuracy': evaluation_results['accuracy'],
            'F1': evaluation_results['f1'],
            'Precision': evaluation_results['precision'],
            'Recall': evaluation_results['recall']
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = axes[1, 1].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom'
            )
        
        plt.tight_layout()
        plt.show()


def create_sample_data(num_samples: int = 1000) -> Dict[str, List]:
    """Create sample text classification data"""
    
    # Sample texts for different classes
    positive_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Amazing performance by the actors. Highly recommended!",
        "One of the best films I've ever seen. Brilliant storytelling.",
        "Excellent cinematography and outstanding direction.",
        "A masterpiece that will be remembered for years to come."
    ]
    
    negative_texts = [
        "This movie was terrible. Complete waste of time.",
        "Poor acting and boring plot. Very disappointed.",
        "One of the worst films ever made. Avoid at all costs.",
        "Terrible direction and awful screenplay.",
        "Completely boring and predictable. Not worth watching."
    ]
    
    neutral_texts = [
        "The movie was okay. Nothing special but watchable.",
        "Average film with some good moments and some bad ones.",
        "It's an alright movie. Not great, not terrible.",
        "Decent acting but the plot could have been better.",
        "Watchable but forgettable. Standard Hollywood fare."
    ]
    
    # Generate data
    texts = []
    labels = []
    
    for i in range(num_samples):
        if i % 3 == 0:
            texts.append(np.random.choice(positive_texts))
            labels.append(2)  # Positive
        elif i % 3 == 1:
            texts.append(np.random.choice(negative_texts))
            labels.append(0)  # Negative
        else:
            texts.append(np.random.choice(neutral_texts))
            labels.append(1)  # Neutral
    
    return {
        'texts': texts,
        'labels': labels,
        'class_names': ['Negative', 'Neutral', 'Positive']
    }


def main():
    """
    Main function to demonstrate feature-based fine-tuning for text classification
    """
    logger.info("Starting text classification example")
    
    # Create sample data
    data = create_sample_data(num_samples=1000)
    texts = data['texts']
    labels = data['labels']
    class_names = data['class_names']
    
    # Split data
    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    test_texts = texts[split_idx:]
    test_labels = labels[split_idx:]
    
    # Further split train into train/eval
    eval_split_idx = int(0.8 * len(train_texts))
    eval_texts = train_texts[eval_split_idx:]
    eval_labels = train_labels[eval_split_idx:]
    train_texts = train_texts[:eval_split_idx]
    train_labels = train_labels[:eval_split_idx]
    
    print(f"Data splits:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Eval: {len(eval_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    
    # Initialize pipeline
    pipeline = TextClassificationPipeline(
        model_name='distilbert-base-uncased',  # Smaller model for faster training
        num_classes=3,
        classifier_type='linear',
        freeze_backbone=True,
        max_length=128
    )
    
    # Prepare data
    pipeline.prepare_data(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        batch_size=16
    )
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING FEATURE-BASED MODEL")
    print("="*60)
    
    training_results = pipeline.train(
        num_epochs=5,
        learning_rate=1e-3,
        weight_decay=0.01,
        save_dir="./checkpoints/feature_based"
    )
    
    print(f"Training completed in {training_results['training_time']:.2f} seconds")
    print(f"Best evaluation metric: {training_results['best_metric']:.4f}")
    
    # Evaluate model
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    evaluation_results = pipeline.evaluate_detailed(
        test_texts=test_texts,
        test_labels=test_labels,
        class_names=class_names
    )
    
    print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Test F1 Score: {evaluation_results['f1']:.4f}")
    print(f"Test Precision: {evaluation_results['precision']:.4f}")
    print(f"Test Recall: {evaluation_results['recall']:.4f}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    report = evaluation_results['classification_report']
    for class_name in class_names:
        metrics = report[class_name]
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")
        print(f"  Support: {metrics['support']}")
    
    # Plot results
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    pipeline.plot_results(evaluation_results, class_names)
    
    # Test predictions on new examples
    print("\n" + "="*60)
    print("PREDICTION EXAMPLES")
    print("="*60)
    
    test_examples = [
        "This movie is absolutely amazing! Best film ever!",
        "Terrible movie, waste of time and money.",
        "The movie was okay, nothing special.",
        "Outstanding performance by all actors!",
        "Boring and predictable plot."
    ]
    
    prediction_results = pipeline.predict(test_examples)
    predictions = prediction_results['predictions']
    probabilities = prediction_results['probabilities']
    
    for i, (text, pred, probs) in enumerate(zip(test_examples, predictions, probabilities)):
        predicted_class = class_names[pred]
        confidence = probs[pred]
        
        print(f"\nExample {i+1}:")
        print(f"Text: {text}")
        print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
        print(f"All probabilities: {dict(zip(class_names, probs))}")
    
    logger.info("Text classification example completed successfully")


if __name__ == "__main__":
    main()
