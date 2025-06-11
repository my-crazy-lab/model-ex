"""
Full Fine-Tuning Trainer Implementation
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
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


class FullFinetuningTrainer:
    """
    Comprehensive trainer for full fine-tuning with advanced features
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        accelerator: Optional[Accelerator] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or {}
        
        # Setup accelerator for distributed training
        self.accelerator = accelerator or Accelerator(
            mixed_precision=self.config.get('mixed_precision', 'no'),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1)
        )
        self.device = device or self.accelerator.device
        
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
        
        # Prepare for distributed training
        if self.accelerator:
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.eval_dataloader,
                self.scheduler
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.eval_dataloader,
                self.scheduler
            )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_eval_metric = 0.0
        self.training_history = []
        
        # Advanced features
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.warmup_steps = self.config.get('warmup_steps', 0)
        self.logging_steps = self.config.get('logging_steps', 100)
        self.eval_steps = self.config.get('eval_steps', 500)
        self.save_steps = self.config.get('save_steps', 1000)
        
        # Setup logging
        self.use_wandb = self.config.get('use_wandb', False)
        self.use_tensorboard = self.config.get('use_tensorboard', False)
        
        if self.use_wandb and self.accelerator.is_main_process:
            try:
                import wandb
                wandb.init(
                    project=self.config.get('wandb_project', 'full-fine-tuning'),
                    config=self.config
                )
            except ImportError:
                logger.warning("wandb not available")
                self.use_wandb = False
        
        if self.use_tensorboard and self.accelerator.is_main_process:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(
                    log_dir=self.config.get('tensorboard_dir', 'runs/full_finetuning')
                )
            except ImportError:
                logger.warning("tensorboard not available")
                self.use_tensorboard = False
        
        logger.info("FullFinetuningTrainer initialized")
        self._log_model_info()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with advanced configurations"""
        
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.get('weight_decay', 0.01),
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer_type = self.config.get('optimizer', 'adamw')
        learning_rate = self.config.get('learning_rate', 2e-5)
        
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=self.config.get('adam_betas', (0.9, 0.999)),
                eps=self.config.get('adam_epsilon', 1e-8)
            )
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=self.config.get('adam_betas', (0.9, 0.999)),
                eps=self.config.get('adam_epsilon', 1e-8)
            )
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(
                optimizer_grouped_parameters,
                lr=learning_rate,
                momentum=self.config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'linear_warmup')
        
        if scheduler_type == 'linear_warmup':
            num_training_steps = len(self.train_dataloader) * self.config.get('num_epochs', 3)
            num_warmup_steps = self.warmup_steps or int(0.1 * num_training_steps)
            
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'cosine_warmup':
            num_training_steps = len(self.train_dataloader) * self.config.get('num_epochs', 3)
            num_warmup_steps = self.warmup_steps or int(0.1 * num_training_steps)
            
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 3)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 1),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.config.get('patience', 3),
                factor=self.config.get('factor', 0.5)
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
        
        # Log training configuration
        logger.info(f"Training configuration:")
        logger.info(f"  Learning rate: {self.config.get('learning_rate', 2e-5)}")
        logger.info(f"  Batch size: {self.train_dataloader.batch_size}")
        logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"  Max gradient norm: {self.max_grad_norm}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")
    
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
            disable=not self.accelerator.is_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.accelerator:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.accelerator:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    self._log_metrics({
                        'train/loss': loss.item() * self.gradient_accumulation_steps,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/global_step': self.global_step
                    })
                
                # Evaluation
                if self.eval_dataloader and self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self._log_metrics({f"eval/{k}": v for k, v in eval_metrics.items()})
                    self.model.train()  # Return to training mode
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    save_dir = self.config.get('output_dir', './checkpoints')
                    self.save_checkpoint(Path(save_dir) / f"checkpoint-{self.global_step}")
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Collect predictions for metrics
            if 'labels' in batch and 'logits' in outputs:
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
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
            for batch in tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_main_process
            ):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                    num_batches += 1
                
                # Collect predictions
                if 'labels' in batch and 'logits' in outputs:
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
        logger.info(f"Starting full fine-tuning for {num_epochs} epochs")
        
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
            
            # Update scheduler (for epoch-based schedulers)
            if self.scheduler and not isinstance(self.scheduler, (
                type(get_linear_schedule_with_warmup(self.optimizer, 0, 1)),
                type(get_cosine_schedule_with_warmup(self.optimizer, 0, 1))
            )):
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
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch}: {epoch_metrics}")
                self._log_metrics({f"epoch/{k}": v for k, v in epoch_metrics.items()})
            
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
        logger.info(f"Full fine-tuning completed in {total_time:.2f} seconds")
        logger.info(f"Best {metric_for_best_model}: {self.best_eval_metric:.4f}")
        
        # Save final model
        if save_dir:
            self.save_checkpoint(save_path / "final_model")
        
        # Close logging
        if self.use_tensorboard:
            self.tb_writer.close()
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to wandb/tensorboard"""
        if self.use_wandb and self.accelerator.is_main_process:
            import wandb
            wandb.log(metrics)
        
        if self.use_tensorboard and self.accelerator.is_main_process:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, self.global_step)
    
    def save_checkpoint(self, save_path: Path):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            # Unwrap model for saving
            model_to_save = self.accelerator.unwrap_model(self.model)
            model_to_save.save_pretrained(save_path)
        else:
            torch.save(
                self.accelerator.unwrap_model(self.model).state_dict(),
                save_path / "pytorch_model.bin"
            )
        
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


# Example usage and testing
if __name__ == "__main__":
    from ..models.full_model import FullFinetuningModel
    
    # Create dummy model
    model = FullFinetuningModel.from_pretrained(
        'bert-base-uncased',
        task='classification',
        num_classes=3,
        freeze_backbone=False
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
        'learning_rate': 2e-5,
        'num_epochs': 2,
        'optimizer': 'adamw',
        'scheduler': 'linear_warmup',
        'gradient_accumulation_steps': 2,
        'max_grad_norm': 1.0,
        'warmup_steps': 10
    }
    
    trainer = FullFinetuningTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config
    )
    
    # Test training
    print("Testing full fine-tuning trainer...")
    trainer.train(num_epochs=1)
    
    print("FullFinetuningTrainer test completed successfully!")
