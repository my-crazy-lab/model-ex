"""
Main distillation class
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os
import json
from tqdm import tqdm

from ..config.distillation_config import DistillationConfig, DistillationType
from .teacher_model import TeacherModel
from .student_model import StudentModel
from .losses import DistillationLoss

logger = logging.getLogger(__name__)


class Distiller:
    """
    Main distillation class that orchestrates knowledge transfer from teacher to student
    """
    
    def __init__(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        config: DistillationConfig,
        tokenizer: Optional[Any] = None
    ):
        self.teacher = teacher
        self.student = student
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize loss function
        self.loss_fn = DistillationLoss(config)
        
        # Training state
        self.current_step = 0
        self.current_stage = 0  # For progressive distillation
        self.current_temperature = config.temperature
        
        # Statistics tracking
        self.training_stats = {
            "total_steps": 0,
            "total_loss": 0.0,
            "distillation_loss": 0.0,
            "task_loss": 0.0,
            "feature_loss": 0.0,
            "attention_loss": 0.0
        }
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher.to(self.device)
        self.student.to(self.device)
        
        logger.info(f"Distiller initialized with {config.distillation_type.value} distillation")
        self._print_model_comparison()
    
    def _print_model_comparison(self):
        """Print comparison between teacher and student models"""
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        
        compression_ratio = teacher_params / student_params
        size_reduction = (1 - student_params / teacher_params) * 100
        
        logger.info("\n" + "="*60)
        logger.info("📊 MODEL COMPARISON")
        logger.info("="*60)
        logger.info(f"Teacher parameters: {teacher_params:,}")
        logger.info(f"Student parameters: {student_params:,}")
        logger.info(f"Compression ratio: {compression_ratio:.1f}x")
        logger.info(f"Size reduction: {size_reduction:.1f}%")
        logger.info("="*60)
    
    def distill(
        self,
        train_dataloader,
        eval_dataloader=None,
        num_epochs: int = 3,
        save_dir: str = "./distilled_model"
    ) -> Dict[str, Any]:
        """
        Main distillation training loop
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save the distilled model
            
        Returns:
            Training statistics
        """
        logger.info("🎓 Starting knowledge distillation...")
        
        # Setup optimizer
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer, len(train_dataloader) * num_epochs)
        
        # Training loop
        self.student.train()
        self.teacher.eval()  # Teacher is always in eval mode
        
        total_steps = len(train_dataloader) * num_epochs
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            epoch_stats = self._train_epoch(
                train_dataloader, optimizer, scheduler, epoch
            )
            
            # Evaluation
            if eval_dataloader is not None:
                eval_stats = self._evaluate(eval_dataloader)
                logger.info(f"Eval accuracy: {eval_stats.get('accuracy', 0):.4f}")
            
            # Progressive distillation stage update
            if self.config.distillation_type == DistillationType.PROGRESSIVE:
                self._update_progressive_stage(epoch, num_epochs)
            
            # Save checkpoint
            if (epoch + 1) % max(1, num_epochs // 3) == 0:
                self._save_checkpoint(save_dir, epoch)
        
        # Final save
        self.save_student_model(save_dir)
        
        logger.info("✅ Distillation completed!")
        return self.training_stats
    
    def _train_epoch(
        self,
        dataloader,
        optimizer,
        scheduler,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        
        epoch_losses = {
            "total_loss": 0.0,
            "distillation_loss": 0.0,
            "task_loss": 0.0,
            "feature_loss": 0.0,
            "attention_loss": 0.0
        }
        
        num_batches = len(dataloader)
        
        with tqdm(dataloader, desc=f"Training Epoch {epoch + 1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                losses = self._forward_step(batch)
                
                # Backward pass
                optimizer.zero_grad()
                losses['total_loss'].backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), 
                        self.config.max_grad_norm
                    )
                
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                
                # Update statistics
                self._update_training_stats(losses)
                
                # Update epoch losses
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'temp': f"{self.current_temperature:.2f}"
                })
                
                self.current_step += 1
                
                # Adaptive temperature update
                if self.config.adaptive_temperature:
                    self.current_temperature = self.loss_fn.adaptive_temperature_update(
                        batch.get('student_logits'),
                        batch.get('teacher_logits'),
                        self.current_temperature
                    )
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward step for distillation"""
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                output_hidden_states=True,
                output_attentions=True
            )
        
        # Student forward pass
        student_outputs = self.student(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Compute losses based on distillation type
        if self.config.distillation_type == DistillationType.PROGRESSIVE:
            losses = self.loss_fn.compute_progressive_loss(
                student_outputs,
                teacher_outputs,
                batch.get('labels'),
                self.current_stage,
                self.current_step
            )
        else:
            losses = self.loss_fn.compute_combined_loss(
                student_outputs,
                teacher_outputs,
                batch.get('labels'),
                self.current_temperature,
                self.current_step
            )
        
        return losses
    
    def _setup_optimizer(self):
        """Setup optimizer for student model"""
        return torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _setup_scheduler(self, optimizer, total_steps: int):
        """Setup learning rate scheduler"""
        from transformers import get_scheduler
        
        return get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
    
    def _update_training_stats(self, losses: Dict[str, torch.Tensor]):
        """Update training statistics"""
        self.training_stats["total_steps"] += 1
        
        for key, value in losses.items():
            if key in self.training_stats and isinstance(value, torch.Tensor):
                self.training_stats[key] += value.item()
    
    def _update_progressive_stage(self, epoch: int, total_epochs: int):
        """Update progressive distillation stage"""
        stage_length = total_epochs / self.config.progressive_stages
        new_stage = min(int(epoch / stage_length), self.config.progressive_stages - 1)
        
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            logger.info(f"🔄 Progressive distillation stage: {self.current_stage + 1}/{self.config.progressive_stages}")
    
    def _evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate student model"""
        self.student.eval()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Student forward pass
                outputs = self.student(**batch)
                
                # Compute accuracy
                if 'labels' in batch:
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct = (predictions == batch['labels']).sum().item()
                    total_correct += correct
                    total_samples += batch['labels'].size(0)
                
                # Compute loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    total_loss += outputs.loss.item()
        
        self.student.train()
        
        results = {
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
            "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        }
        
        return results
    
    def _save_checkpoint(self, save_dir: str, epoch: int):
        """Save training checkpoint"""
        checkpoint_dir = os.path.join(save_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save student model
        self.student.save_pretrained(checkpoint_dir)
        
        # Save training state
        checkpoint_state = {
            "epoch": epoch,
            "step": self.current_step,
            "stage": self.current_stage,
            "temperature": self.current_temperature,
            "training_stats": self.training_stats,
            "config": self.config.to_dict()
        }
        
        with open(os.path.join(checkpoint_dir, "training_state.json"), 'w') as f:
            json.dump(checkpoint_state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def save_student_model(self, save_dir: str):
        """Save the final distilled student model"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save student model
        self.student.save_pretrained(save_dir)
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)
        
        # Save distillation config
        with open(os.path.join(save_dir, "distillation_config.json"), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save training statistics
        final_stats = self.get_training_statistics()
        with open(os.path.join(save_dir, "training_stats.json"), 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info(f"Distilled model saved to {save_dir}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = self.training_stats.copy()
        
        # Average losses
        if stats["total_steps"] > 0:
            for key in ["total_loss", "distillation_loss", "task_loss", "feature_loss", "attention_loss"]:
                if key in stats:
                    stats[f"avg_{key}"] = stats[key] / stats["total_steps"]
        
        # Model comparison
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        
        stats.update({
            "teacher_parameters": teacher_params,
            "student_parameters": student_params,
            "compression_ratio": teacher_params / student_params,
            "size_reduction_percent": (1 - student_params / teacher_params) * 100,
            "final_temperature": self.current_temperature,
            "final_stage": self.current_stage
        })
        
        return stats
    
    def compare_models(self, test_dataloader) -> Dict[str, Any]:
        """Compare teacher and student performance"""
        logger.info("🔍 Comparing teacher and student models...")
        
        # Evaluate teacher
        teacher_stats = self._evaluate_model(self.teacher, test_dataloader, "Teacher")
        
        # Evaluate student
        student_stats = self._evaluate_model(self.student, test_dataloader, "Student")
        
        # Compute performance retention
        performance_retention = (student_stats["accuracy"] / teacher_stats["accuracy"]) * 100
        
        comparison = {
            "teacher": teacher_stats,
            "student": student_stats,
            "performance_retention": performance_retention,
            "accuracy_drop": teacher_stats["accuracy"] - student_stats["accuracy"],
            "compression_ratio": self.get_training_statistics()["compression_ratio"]
        }
        
        logger.info(f"Performance retention: {performance_retention:.1f}%")
        logger.info(f"Accuracy drop: {comparison['accuracy_drop']:.4f}")
        
        return comparison
    
    def _evaluate_model(self, model, dataloader, model_name: str) -> Dict[str, float]:
        """Evaluate a specific model"""
        model.eval()
        
        total_correct = 0
        total_samples = 0
        total_time = 0.0
        
        import time
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {model_name}"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                start_time = time.time()
                outputs = model(**batch)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                
                if 'labels' in batch:
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct = (predictions == batch['labels']).sum().item()
                    total_correct += correct
                    total_samples += batch['labels'].size(0)
        
        return {
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
            "inference_time": total_time / len(dataloader),
            "samples_per_second": total_samples / total_time if total_time > 0 else 0.0
        }
