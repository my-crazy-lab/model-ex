"""
Cross-Modal Training Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
from tqdm import tqdm
import logging
import wandb
from pathlib import Path
import json
import time
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class CrossModalTrainer:
    """
    Comprehensive trainer for cross-modal models
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        accelerator: Optional[Accelerator] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        
        # Setup accelerator for distributed training
        self.accelerator = accelerator or Accelerator()
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
        
        # Setup loss function
        if loss_fn is None:
            self.loss_fn = self._create_loss_function()
        else:
            self.loss_fn = loss_fn
        
        # Prepare for distributed training
        if self.accelerator:
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.val_dataloader,
                self.scheduler
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.val_dataloader,
                self.scheduler
            )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Setup logging
        self.use_wandb = self.config.get('use_wandb', False)
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=self.config.get('wandb_project', 'cross-modal-training'),
                config=self.config
            )
        
        logger.info("CrossModalTrainer initialized")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        # Separate parameters for different learning rates
        text_params = []
        vision_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'text' in name.lower():
                text_params.append(param)
            elif 'vision' in name.lower() or 'visual' in name.lower():
                vision_params.append(param)
            else:
                other_params.append(param)
        
        # Different learning rates for different components
        param_groups = [
            {'params': text_params, 'lr': lr * 0.1},  # Lower LR for pre-trained text
            {'params': vision_params, 'lr': lr * 0.1},  # Lower LR for pre-trained vision
            {'params': other_params, 'lr': lr}  # Full LR for new parameters
        ]
        
        optimizer_type = self.config.get('optimizer', 'adamw')
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(param_groups, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            return optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'linear')
        
        if scheduler_type == 'linear':
            num_training_steps = len(self.train_dataloader) * self.config.get('num_epochs', 10)
            num_warmup_steps = int(0.1 * num_training_steps)
            
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 10)
            )
        else:
            return None
    
    def _create_loss_function(self) -> Callable:
        """Create loss function"""
        loss_type = self.config.get('loss_type', 'contrastive')
        
        if loss_type == 'contrastive':
            return self._contrastive_loss
        elif loss_type == 'triplet':
            margin = self.config.get('triplet_margin', 0.2)
            return nn.TripletMarginLoss(margin=margin)
        else:
            return nn.CrossEntropyLoss()
    
    def _contrastive_loss(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """Compute contrastive loss (CLIP-style)"""
        # Normalize embeddings
        text_embeds = nn.functional.normalize(text_embeds, dim=-1)
        image_embeds = nn.functional.normalize(image_embeds, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(text_embeds, image_embeds.T) / temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = text_embeds.shape[0]
        labels = torch.arange(batch_size, device=text_embeds.device)
        
        # Compute loss in both directions
        text_loss = nn.functional.cross_entropy(logits, labels)
        image_loss = nn.functional.cross_entropy(logits.T, labels)
        
        return (text_loss + image_loss) / 2
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
            disable=not self.accelerator.is_main_process
        )
        
        for batch in progress_bar:
            # Forward pass
            outputs = self.model(**batch)
            
            # Compute loss
            if 'loss' in outputs:
                loss = outputs['loss']
            else:
                # Compute custom loss
                if 'text_embeds' in outputs and 'image_embeds' in outputs:
                    loss = self._contrastive_loss(
                        outputs['text_embeds'],
                        outputs['image_embeds']
                    )
                else:
                    raise ValueError("Cannot compute loss from model outputs")
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.config.get('max_grad_norm', 0) > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
            
            # Optimizer step
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if self.use_wandb and self.accelerator.is_main_process:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/global_step': self.global_step
                })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(
                self.val_dataloader,
                desc="Validation",
                disable=not self.accelerator.is_main_process
            ):
                # Forward pass
                outputs = self.model(**batch)
                
                # Compute loss
                if 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    if 'text_embeds' in outputs and 'image_embeds' in outputs:
                        loss = self._contrastive_loss(
                            outputs['text_embeds'],
                            outputs['image_embeds']
                        )
                    else:
                        continue
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss}
    
    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None,
        save_every: int = 1,
        eval_every: int = 1,
        early_stopping_patience: int = 5
    ):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = {}
            if epoch % eval_every == 0:
                val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_history.append(epoch_metrics)
            
            # Log metrics
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch}: {epoch_metrics}")
                
                if self.use_wandb:
                    wandb.log({
                        **{f"epoch/{k}": v for k, v in epoch_metrics.items()},
                        'epoch': epoch
                    })
            
            # Save checkpoint
            if save_dir and epoch % save_every == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch}")
            
            # Early stopping
            if 'val_loss' in val_metrics:
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    
                    # Save best model
                    if save_dir:
                        self.save_checkpoint(save_path / "best_model")
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping after {epoch} epochs")
                        break
        
        logger.info("Training completed")
        
        # Save final model
        if save_dir:
            self.save_checkpoint(save_path / "final_model")
    
    def save_checkpoint(self, save_path: Path):
        """Save model checkpoint"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
        
        # Save training state
        training_state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
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
        # Load model state
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
            self.best_val_loss = training_state['best_val_loss']
            self.training_history = training_state['training_history']
            
            self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in training_state:
                self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def evaluate_retrieval(
        self,
        test_dataloader: DataLoader,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """Evaluate cross-modal retrieval performance"""
        self.model.eval()
        
        all_text_embeds = []
        all_image_embeds = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Computing embeddings"):
                outputs = self.model(**batch)
                
                if 'text_embeds' in outputs:
                    all_text_embeds.append(outputs['text_embeds'].cpu())
                if 'image_embeds' in outputs:
                    all_image_embeds.append(outputs['image_embeds'].cpu())
        
        # Concatenate all embeddings
        text_embeds = torch.cat(all_text_embeds, dim=0)
        image_embeds = torch.cat(all_image_embeds, dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(text_embeds, image_embeds.T)
        
        # Evaluate text-to-image retrieval
        t2i_metrics = self._compute_retrieval_metrics(
            similarity_matrix, k_values, "text_to_image"
        )
        
        # Evaluate image-to-text retrieval
        i2t_metrics = self._compute_retrieval_metrics(
            similarity_matrix.T, k_values, "image_to_text"
        )
        
        return {**t2i_metrics, **i2t_metrics}
    
    def _compute_retrieval_metrics(
        self,
        similarity_matrix: torch.Tensor,
        k_values: List[int],
        prefix: str
    ) -> Dict[str, float]:
        """Compute retrieval metrics"""
        num_queries = similarity_matrix.shape[0]
        
        # Get rankings
        rankings = torch.argsort(similarity_matrix, dim=1, descending=True)
        
        # Compute recall@k
        metrics = {}
        for k in k_values:
            # Check if correct item is in top-k
            correct_in_topk = (rankings[:, :k] == torch.arange(num_queries).unsqueeze(1)).any(dim=1)
            recall_at_k = correct_in_topk.float().mean().item()
            metrics[f"{prefix}_recall_at_{k}"] = recall_at_k
        
        # Compute mean rank
        correct_ranks = (rankings == torch.arange(num_queries).unsqueeze(1)).nonzero()[:, 1]
        mean_rank = correct_ranks.float().mean().item() + 1  # 1-indexed
        metrics[f"{prefix}_mean_rank"] = mean_rank
        
        # Compute median rank
        median_rank = correct_ranks.float().median().item() + 1  # 1-indexed
        metrics[f"{prefix}_median_rank"] = median_rank
        
        return metrics


# Example usage and testing
if __name__ == "__main__":
    from ..models.clip_model import CLIPModel
    from ..data.multimodal_dataset import MultiModalDataset, create_multimodal_dataloader
    
    # Initialize model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create dummy dataset
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy data (simplified for testing)
        dataset = MultiModalDataset(
            data_path=temp_dir,
            modalities=['text', 'image']
        )
        
        # Create dataloader
        dataloader = create_multimodal_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        # Initialize trainer
        config = {
            'learning_rate': 1e-5,
            'num_epochs': 2,
            'use_wandb': False
        }
        
        trainer = CrossModalTrainer(
            model=model,
            train_dataloader=dataloader,
            config=config
        )
        
        # Test training (just one step)
        print("Testing trainer initialization...")
        print("CrossModalTrainer test completed successfully!")
