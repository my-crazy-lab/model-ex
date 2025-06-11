"""
Comprehensive Text Classification Example using Full Fine-Tuning

This example demonstrates how to use full fine-tuning for text classification,
showcasing advanced training techniques and comprehensive evaluation.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
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

from models.full_model import FullFinetuningModel
from training.trainer import FullFinetuningTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    """
    Dataset for text classification tasks with advanced preprocessing
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
        label_smoothing: float = 0.0
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_smoothing = label_smoothing
        self.num_classes = len(set(labels))
    
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
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            smooth_label = torch.full((self.num_classes,), self.label_smoothing / (self.num_classes - 1))
            smooth_label[label] = 1.0 - self.label_smoothing
            label_tensor = smooth_label
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }


class AdvancedTextClassificationPipeline:
    """
    Advanced pipeline for text classification with full fine-tuning
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 3,
        max_length: int = 128,
        use_gradient_checkpointing: bool = False
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model with full fine-tuning
        self.model = FullFinetuningModel.from_pretrained(
            model_name,
            task='classification',
            num_classes=num_classes,
            freeze_backbone=False,  # Full fine-tuning
            gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Training components
        self.trainer = None
        self.training_history = []
        
        logger.info(f"Initialized AdvancedTextClassificationPipeline with {model_name}")
        self.model.print_parameter_status()
    
    def prepare_data(
        self,
        train_texts: List[str],
        train_labels: List[int],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[int]] = None,
        batch_size: int = 16,
        label_smoothing: float = 0.0
    ):
        """Prepare datasets and dataloaders with advanced features"""
        
        # Create train dataset
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, 
            self.max_length, label_smoothing
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
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
                num_workers=2,
                pin_memory=True
            )
        
        logger.info(f"Prepared data: {len(train_dataset)} train samples")
        if self.eval_dataloader:
            logger.info(f"Prepared data: {len(eval_dataset)} eval samples")
    
    def train(
        self,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_dir: Optional[str] = None,
        use_mixed_precision: bool = False,
        **trainer_kwargs
    ):
        """Train the model with advanced techniques"""
        
        # Calculate training steps
        total_steps = len(self.train_dataloader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        
        # Training configuration
        config = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'optimizer': 'adamw',
            'scheduler': 'linear_warmup',
            'warmup_steps': warmup_steps,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'max_grad_norm': max_grad_norm,
            'mixed_precision': 'fp16' if use_mixed_precision else 'no',
            'logging_steps': 50,
            'eval_steps': 200,
            'save_steps': 500,
            **trainer_kwargs
        }
        
        # Initialize trainer
        self.trainer = FullFinetuningTrainer(
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
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 16,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
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
                if return_probabilities:
                    all_probabilities.extend(probabilities.cpu().numpy())
        
        results = {'predictions': all_predictions}
        if return_probabilities:
            results['probabilities'] = all_probabilities
        
        return results
    
    def evaluate_detailed(
        self,
        test_texts: List[str],
        test_labels: List[int],
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detailed evaluation with comprehensive metrics"""
        
        # Make predictions
        results = self.predict(test_texts, return_probabilities=True)
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, average_precision_score
        )
        
        accuracy = accuracy_score(test_labels, predictions)
        f1_macro = f1_score(test_labels, predictions, average='macro')
        f1_weighted = f1_score(test_labels, predictions, average='weighted')
        precision = precision_score(test_labels, predictions, average='weighted')
        recall = recall_score(test_labels, predictions, average='weighted')
        
        # Multi-class AUC
        try:
            if len(set(test_labels)) > 2:
                auc = roc_auc_score(test_labels, probabilities, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(test_labels, [p[1] for p in probabilities])
        except:
            auc = 0.0
        
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
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if class_name in report:
                per_class_metrics[class_name] = report[class_name]
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def plot_comprehensive_results(
        self,
        evaluation_results: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plot comprehensive evaluation results"""
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Training history - Loss
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
        
        # 2. Training history - Accuracy
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
        
        # 3. Learning Rate Schedule
        if self.training_history:
            learning_rates = [h.get('learning_rate', 0) for h in self.training_history]
            axes[0, 2].plot(epochs, learning_rates, marker='o')
            axes[0, 2].set_title('Learning Rate Schedule')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].grid(True)
        
        # 4. Confusion Matrix
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
        
        # 5. Performance Metrics
        metrics = {
            'Accuracy': evaluation_results['accuracy'],
            'F1 (Macro)': evaluation_results['f1_macro'],
            'F1 (Weighted)': evaluation_results['f1_weighted'],
            'Precision': evaluation_results['precision'],
            'Recall': evaluation_results['recall'],
            'AUC': evaluation_results['auc']
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = axes[1, 1].bar(metric_names, metric_values, 
                             color=['skyblue', 'lightgreen', 'lightcoral', 
                                   'lightyellow', 'lightpink', 'lightgray'])
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom'
            )
        
        # 6. Per-Class F1 Scores
        per_class_f1 = []
        for class_name in class_names:
            if class_name in evaluation_results['per_class_metrics']:
                per_class_f1.append(evaluation_results['per_class_metrics'][class_name]['f1-score'])
            else:
                per_class_f1.append(0.0)
        
        bars = axes[1, 2].bar(class_names, per_class_f1, color='lightsteelblue')
        axes[1, 2].set_title('Per-Class F1 Scores')
        axes[1, 2].set_ylabel('F1 Score')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, per_class_f1):
            axes[1, 2].text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        
        plt.show()


def create_advanced_sample_data(num_samples: int = 2000) -> Dict[str, List]:
    """Create advanced sample text classification data"""
    
    # More diverse sample texts for different classes
    positive_texts = [
        "This movie is absolutely fantastic! The acting is superb and the plot is engaging.",
        "Amazing cinematography and outstanding direction. A true masterpiece!",
        "Brilliant performance by all actors. Highly recommended for everyone.",
        "Excellent storytelling with beautiful visuals. One of the best films ever.",
        "Outstanding movie with great character development and emotional depth.",
        "Incredible film that exceeded all my expectations. Simply amazing!",
        "Wonderful acting and a compelling storyline. Truly exceptional work.",
        "Magnificent production with stellar performances. A must-watch film!"
    ]
    
    negative_texts = [
        "This movie was terrible. Poor acting and a boring, predictable plot.",
        "Awful film with terrible direction and weak character development.",
        "Complete waste of time. The story makes no sense and acting is horrible.",
        "Disappointing movie with poor production quality and bad screenplay.",
        "Terrible film that fails on every level. Avoid at all costs.",
        "Boring and poorly executed. One of the worst movies I've ever seen.",
        "Awful acting and a confusing plot. Very disappointing experience.",
        "Poor quality film with no redeeming qualities. Completely unwatchable."
    ]
    
    neutral_texts = [
        "The movie was okay. Some good moments but overall just average.",
        "Decent film with acceptable acting. Nothing special but watchable.",
        "Average movie that's neither great nor terrible. Just okay.",
        "The film has its moments but overall it's pretty standard.",
        "Watchable but forgettable. Standard Hollywood entertainment.",
        "Okay movie with some interesting parts but nothing outstanding.",
        "Average production with decent acting. Not bad but not great either.",
        "The film is fine. Some good scenes but overall just mediocre."
    ]
    
    # Generate balanced data with some noise
    texts = []
    labels = []
    
    samples_per_class = num_samples // 3
    
    for i in range(samples_per_class):
        # Positive samples
        base_text = np.random.choice(positive_texts)
        # Add some variation
        if np.random.random() < 0.3:
            base_text += " " + np.random.choice([
                "Really enjoyed it.", "Great experience.", "Loved every minute.",
                "Fantastic work.", "Brilliant movie."
            ])
        texts.append(base_text)
        labels.append(2)  # Positive
        
        # Negative samples
        base_text = np.random.choice(negative_texts)
        if np.random.random() < 0.3:
            base_text += " " + np.random.choice([
                "Very disappointed.", "Not recommended.", "Waste of money.",
                "Terrible experience.", "Completely awful."
            ])
        texts.append(base_text)
        labels.append(0)  # Negative
        
        # Neutral samples
        base_text = np.random.choice(neutral_texts)
        if np.random.random() < 0.3:
            base_text += " " + np.random.choice([
                "Could be better.", "It's alright.", "Nothing special.",
                "Pretty standard.", "Just okay."
            ])
        texts.append(base_text)
        labels.append(1)  # Neutral
    
    # Shuffle the data
    combined = list(zip(texts, labels))
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return {
        'texts': list(texts),
        'labels': list(labels),
        'class_names': ['Negative', 'Neutral', 'Positive']
    }


def main():
    """
    Main function to demonstrate full fine-tuning for text classification
    """
    logger.info("Starting advanced text classification example with full fine-tuning")
    
    # Create sample data
    data = create_advanced_sample_data(num_samples=2000)
    texts = data['texts']
    labels = data['labels']
    class_names = data['class_names']
    
    # Split data
    split_idx = int(0.7 * len(texts))
    eval_split_idx = int(0.85 * len(texts))
    
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    eval_texts = texts[split_idx:eval_split_idx]
    eval_labels = labels[split_idx:eval_split_idx]
    test_texts = texts[eval_split_idx:]
    test_labels = labels[eval_split_idx:]
    
    print(f"Data splits:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Eval: {len(eval_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    
    # Initialize pipeline
    pipeline = AdvancedTextClassificationPipeline(
        model_name='bert-base-uncased',
        num_classes=3,
        max_length=128,
        use_gradient_checkpointing=True  # For memory efficiency
    )
    
    # Prepare data
    pipeline.prepare_data(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        batch_size=16,
        label_smoothing=0.1  # Advanced technique
    )
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING FULL FINE-TUNING MODEL")
    print("="*60)
    
    training_results = pipeline.train(
        num_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        save_dir="./checkpoints/full_finetuning",
        use_mixed_precision=True,  # FP16 for efficiency
        use_wandb=False  # Set to True if you have wandb
    )
    
    print(f"Training completed in {training_results['training_time']:.2f} seconds")
    print(f"Best evaluation metric: {training_results['best_metric']:.4f}")
    
    # Evaluate model
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION")
    print("="*60)
    
    evaluation_results = pipeline.evaluate_detailed(
        test_texts=test_texts,
        test_labels=test_labels,
        class_names=class_names
    )
    
    print(f"Test Results:")
    print(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"  F1 (Macro): {evaluation_results['f1_macro']:.4f}")
    print(f"  F1 (Weighted): {evaluation_results['f1_weighted']:.4f}")
    print(f"  Precision: {evaluation_results['precision']:.4f}")
    print(f"  Recall: {evaluation_results['recall']:.4f}")
    print(f"  AUC: {evaluation_results['auc']:.4f}")
    
    # Print per-class metrics
    print("\nPer-Class Metrics:")
    for class_name, metrics in evaluation_results['per_class_metrics'].items():
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")
        print(f"  Support: {metrics['support']}")
    
    # Plot comprehensive results
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    pipeline.plot_comprehensive_results(
        evaluation_results,
        class_names,
        save_path="full_finetuning_results.png"
    )
    
    # Test predictions on new examples
    print("\n" + "="*60)
    print("PREDICTION EXAMPLES")
    print("="*60)
    
    test_examples = [
        "This movie is absolutely incredible! Best film I've ever seen!",
        "Terrible movie, complete waste of time and money.",
        "The movie was okay, nothing special but watchable.",
        "Outstanding cinematography and brilliant acting throughout!",
        "Boring and predictable plot with poor character development.",
        "Decent film with some good moments and some weak parts."
    ]
    
    prediction_results = pipeline.predict(test_examples, return_probabilities=True)
    predictions = prediction_results['predictions']
    probabilities = prediction_results['probabilities']
    
    for i, (text, pred, probs) in enumerate(zip(test_examples, predictions, probabilities)):
        predicted_class = class_names[pred]
        confidence = probs[pred]
        
        print(f"\nExample {i+1}:")
        print(f"Text: {text}")
        print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
        print(f"All probabilities:")
        for j, (class_name, prob) in enumerate(zip(class_names, probs)):
            print(f"  {class_name}: {prob:.3f}")
    
    logger.info("Advanced text classification example completed successfully")


if __name__ == "__main__":
    main()
