"""
Feature-Based Fine-Tuning Trainer Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
from tqdm import tqdm
import logging
import time
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class FeatureBasedTrainer:
    """
    Trainer for feature-based fine-tuning with frozen backbone
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        if scheduler is None:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = scheduler
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_eval_metric = 0.0
        self.training_history = []
        
        # Setup logging
        self.use_wandb = self.config.get('use_wandb', False)
        self.use_tensorboard = self.config.get('use_tensorboard', False)
        
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.get('wandb_project', 'feature-based-fine-tuning'),
                    config=self.config
                )
            except ImportError:
                logger.warning("wandb not available")
                self.use_wandb = False
        
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(
                    log_dir=self.config.get('tensorboard_dir', 'runs/feature_based')
                )
            except ImportError:
                logger.warning("tensorboard not available")
                self.use_tensorboard = False
        
        logger.info("FeatureBasedTrainer initialized")
        self._log_model_info()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for trainable parameters only"""
        # Get only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer_type = self.config.get('optimizer', 'adamw')
        learning_rate = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(trainable_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            num_epochs = self.config.get('num_epochs', 10)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            step_size = self.config.get('step_size', 5)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=3, factor=0.5
            )
        else:
            return None
    
    def _log_model_info(self):
        """Log model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info(f"Model parameters:")
        logger.info(f"  Total: {total_params:,}")
        logger.info(f"  Trainable: {trainable_params:,} ({trainable_params/total_params:.2%})")
        logger.info(f"  Frozen: {frozen_params:,} ({frozen_params/total_params:.2%})")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
            leave=False
        )
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.config.get('max_grad_norm', 1.0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Collect predictions for metrics
            if 'labels' in batch:
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb/tensorboard
            if self.global_step % self.config.get('log_steps', 100) == 0:
                self._log_metrics({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/global_step': self.global_step
                })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = {'train_loss': avg_loss}
        
        if all_predictions and all_labels:
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            metrics.update({
                'train_accuracy': accuracy,
                'train_f1': f1
            })
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model"""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_logits = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                    num_batches += 1
                
                # Collect predictions
                if 'labels' in batch:
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                    all_logits.extend(outputs['logits'].cpu().numpy())
        
        # Compute metrics
        metrics = {}
        
        if num_batches > 0:
            metrics['eval_loss'] = total_loss / num_batches
        
        if all_predictions and all_labels:
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            
            metrics.update({
                'eval_accuracy': accuracy,
                'eval_f1': f1,
                'eval_precision': precision,
                'eval_recall': recall
            })
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None,
        save_every: int = 1,
        eval_every: int = 1,
        early_stopping_patience: int = 5,
        metric_for_best_model: str = 'eval_accuracy'
    ):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            eval_metrics = {}
            if epoch % eval_every == 0:
                eval_metrics = self.evaluate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if metric_for_best_model in eval_metrics:
                        self.scheduler.step(eval_metrics[metric_for_best_model])
                else:
                    self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **eval_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            
            self.training_history.append(epoch_metrics)
            
            # Log metrics
            logger.info(f"Epoch {epoch}: {epoch_metrics}")
            self._log_metrics(epoch_metrics)
            
            # Save checkpoint
            if save_dir and epoch % save_every == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch}")
            
            # Early stopping and best model saving
            if metric_for_best_model in eval_metrics:
                current_metric = eval_metrics[metric_for_best_model]
                
                if current_metric > self.best_eval_metric:
                    self.best_eval_metric = current_metric
                    patience_counter = 0
                    
                    # Save best model
                    if save_dir:
                        self.save_checkpoint(save_path / "best_model")
                        logger.info(f"New best model saved with {metric_for_best_model}: {current_metric:.4f}")
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping after {epoch} epochs")
                        break
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Best {metric_for_best_model}: {self.best_eval_metric:.4f}")
        
        # Save final model
        if save_dir:
            self.save_checkpoint(save_path / "final_model")
        
        # Close logging
        if self.use_tensorboard:
            self.tb_writer.close()
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to wandb/tensorboard"""
        if self.use_wandb:
            import wandb
            wandb.log(metrics)
        
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, self.global_step)
    
    def save_checkpoint(self, save_path: Path):
        """Save model checkpoint"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
        
        # Save training state
        training_state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_eval_metric': self.best_eval_metric,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        if self.scheduler:
            training_state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(training_state, save_path / "training_state.pt")
        
        # Save config
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        # Load model
        if (checkpoint_path / "pytorch_model.bin").exists():
            state_dict = torch.load(
                checkpoint_path / "pytorch_model.bin",
                map_location=self.device
            )
            self.model.load_state_dict(state_dict)
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=self.device)
            
            self.current_epoch = training_state['epoch']
            self.global_step = training_state['global_step']
            self.best_eval_metric = training_state['best_eval_metric']
            self.training_history = training_state['training_history']
            
            self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in training_state:
                self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if not self.training_history:
            logger.warning("No training history to plot")
            return
        
        # Extract metrics
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h.get('train_loss', 0) for h in self.training_history]
        eval_losses = [h.get('eval_loss', 0) for h in self.training_history if 'eval_loss' in h]
        train_accuracies = [h.get('train_accuracy', 0) for h in self.training_history if 'train_accuracy' in h]
        eval_accuracies = [h.get('eval_accuracy', 0) for h in self.training_history if 'eval_accuracy' in h]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plots
        axes[0, 0].plot(epochs, train_losses, label='Train Loss')
        if eval_losses:
            eval_epochs = [h['epoch'] for h in self.training_history if 'eval_loss' in h]
            axes[0, 0].plot(eval_epochs, eval_losses, label='Eval Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plots
        if train_accuracies:
            train_acc_epochs = [h['epoch'] for h in self.training_history if 'train_accuracy' in h]
            axes[0, 1].plot(train_acc_epochs, train_accuracies, label='Train Accuracy')
        if eval_accuracies:
            eval_acc_epochs = [h['epoch'] for h in self.training_history if 'eval_accuracy' in h]
            axes[0, 1].plot(eval_acc_epochs, eval_accuracies, label='Eval Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        learning_rates = [h.get('learning_rate', 0) for h in self.training_history]
        axes[1, 0].plot(epochs, learning_rates)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # F1 scores
        train_f1s = [h.get('train_f1', 0) for h in self.training_history if 'train_f1' in h]
        eval_f1s = [h.get('eval_f1', 0) for h in self.training_history if 'eval_f1' in h]
        
        if train_f1s:
            train_f1_epochs = [h['epoch'] for h in self.training_history if 'train_f1' in h]
            axes[1, 1].plot(train_f1_epochs, train_f1s, label='Train F1')
        if eval_f1s:
            eval_f1_epochs = [h['epoch'] for h in self.training_history if 'eval_f1' in h]
            axes[1, 1].plot(eval_f1_epochs, eval_f1s, label='Eval F1')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    from ..models.feature_based_model import FeatureBasedModel
    
    # Create dummy model
    model = FeatureBasedModel.from_pretrained(
        'bert-base-uncased',
        num_classes=3,
        freeze_backbone=True
    )
    
    # Create dummy data
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 1000, (128,)),
                'attention_mask': torch.ones(128),
                'labels': torch.randint(0, 3, ())
            }
    
    from torch.utils.data import DataLoader
    
    train_dataset = DummyDataset(100)
    eval_dataset = DummyDataset(20)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)
    
    # Initialize trainer
    config = {
        'learning_rate': 1e-3,
        'num_epochs': 3,
        'optimizer': 'adamw',
        'scheduler': 'cosine'
    }
    
    trainer = FeatureBasedTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config
    )
    
    # Test training
    print("Testing trainer...")
    trainer.train(num_epochs=2)
    
    print("FeatureBasedTrainer test completed successfully!")
