"""
Main multitask model implementation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os
import json
from transformers import (
    AutoModel, AutoConfig, AutoTokenizer,
    PreTrainedModel, PreTrainedTokenizer
)

from ..config.multitask_config import MultitaskConfig, ArchitectureType
from .task_heads import TaskHeads
from .loss_manager import MultitaskLossManager

logger = logging.getLogger(__name__)


class MultitaskModel(nn.Module):
    """
    Main multitask model that combines shared backbone with task-specific heads
    """
    
    def __init__(
        self,
        config: MultitaskConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        super().__init__()
        
        self.config = config
        self.tokenizer = tokenizer
        
        # Load model configuration
        self.model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        
        # Load shared backbone
        self.backbone = self._load_backbone()
        
        # Initialize task heads
        self.task_heads = TaskHeads(
            hidden_size=self.model_config.hidden_size,
            task_configs=config.tasks,
            vocab_size=getattr(self.model_config, 'vocab_size', None),
            use_task_embeddings=config.use_task_embeddings,
            task_embedding_dim=config.task_embedding_dim
        )
        
        # Initialize loss manager
        self.loss_manager = MultitaskLossManager(config)
        
        # Training state
        self.current_task = None
        self.training_step = 0
        
        logger.info("MultitaskModel initialized successfully")
        self._print_model_summary()
    
    def _load_backbone(self) -> PreTrainedModel:
        """Load the shared backbone model"""
        if self.config.architecture_type == ArchitectureType.SHARED_BOTTOM:
            # Standard shared encoder
            backbone = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                config=self.model_config
            )
        else:
            # For other architectures, start with base model
            backbone = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                config=self.model_config
            )
            
            # TODO: Implement other architectures (MOE, Cross-stitch, etc.)
            logger.warning(f"Architecture {self.config.architecture_type} not fully implemented, using shared bottom")
        
        return backbone
    
    def _print_model_summary(self):
        """Print model summary"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = self.task_heads.get_num_parameters()
        total_head_params = sum(head_params.values())
        total_params = backbone_params + total_head_params
        
        logger.info("\n" + "="*60)
        logger.info("🎯 MULTITASK MODEL SUMMARY")
        logger.info("="*60)
        logger.info(f"Backbone parameters: {backbone_params:,}")
        logger.info(f"Task head parameters: {total_head_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Number of tasks: {len(self.config.tasks)}")
        logger.info(f"Tasks: {', '.join(self.config.get_task_names())}")
        
        for task_name, param_count in head_params.items():
            logger.info(f"  {task_name}: {param_count:,} parameters")
        
        logger.info("="*60)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
        task_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through multitask model
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            task_name: Name of the current task
            task_ids: Task IDs for task embeddings [batch_size]
            labels: Ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing outputs and losses
        """
        if task_name is None:
            task_name = self.current_task
        
        if task_name is None:
            raise ValueError("Task name must be specified")
        
        # Forward through shared backbone
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            **kwargs
        )
        
        # Get hidden states
        hidden_states = backbone_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Create task IDs if not provided and task embeddings are used
        if self.config.use_task_embeddings and task_ids is None:
            task_id = self.task_heads.get_task_id(task_name)
            task_ids = torch.full(
                (input_ids.size(0),), 
                task_id, 
                dtype=torch.long, 
                device=input_ids.device
            )
        
        # Forward through task-specific head
        task_outputs = self.task_heads(
            hidden_states=hidden_states,
            task_name=task_name,
            attention_mask=attention_mask,
            task_ids=task_ids
        )
        
        # Prepare outputs
        outputs = {
            "task_name": task_name,
            "hidden_states": hidden_states,
            "backbone_outputs": backbone_outputs
        }
        
        # Handle different task output formats
        if isinstance(task_outputs, dict):
            outputs.update(task_outputs)
        else:
            outputs["logits"] = task_outputs
        
        # Compute loss if labels provided
        if labels is not None:
            task_config = self.config.tasks[task_name]
            loss = self.loss_manager.compute_task_loss(
                task_outputs=task_outputs,
                labels=labels,
                task_config=task_config,
                task_name=task_name
            )
            outputs["loss"] = loss
        
        return outputs
    
    def set_current_task(self, task_name: str):
        """Set the current task for training"""
        if task_name not in self.config.tasks:
            raise ValueError(f"Unknown task: {task_name}")
        
        self.current_task = task_name
        logger.debug(f"Current task set to: {task_name}")
    
    def add_task(self, task_name: str, task_config: Any):
        """Add a new task to the model"""
        # Add to config
        self.config.tasks[task_name] = task_config
        
        # Add task head
        self.task_heads.add_task_head(
            task_name=task_name,
            task_config=task_config,
            vocab_size=getattr(self.model_config, 'vocab_size', None)
        )
        
        logger.info(f"Added task: {task_name}")
    
    def remove_task(self, task_name: str):
        """Remove a task from the model"""
        if task_name not in self.config.tasks:
            raise ValueError(f"Task {task_name} not found")
        
        # Remove from config
        del self.config.tasks[task_name]
        
        # Remove task head
        self.task_heads.remove_task_head(task_name)
        
        # Update current task if needed
        if self.current_task == task_name:
            remaining_tasks = self.config.get_task_names()
            self.current_task = remaining_tasks[0] if remaining_tasks else None
        
        logger.info(f"Removed task: {task_name}")
    
    def evaluate_task(
        self,
        task_name: str,
        dataloader,
        device: torch.device,
        compute_metrics_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Evaluate model on specific task"""
        self.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.forward(task_name=task_name, **batch)
                
                # Collect predictions and labels
                if "logits" in outputs:
                    predictions = torch.argmax(outputs["logits"], dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                elif "start_logits" in outputs and "end_logits" in outputs:
                    # Question answering
                    start_predictions = torch.argmax(outputs["start_logits"], dim=-1)
                    end_predictions = torch.argmax(outputs["end_logits"], dim=-1)
                    all_predictions.extend(list(zip(start_predictions.cpu().numpy(), end_predictions.cpu().numpy())))
                
                if "labels" in batch:
                    all_labels.extend(batch["labels"].cpu().numpy())
                
                # Accumulate loss
                if "loss" in outputs:
                    total_loss += outputs["loss"].item()
                    num_batches += 1
        
        # Compute metrics
        results = {
            "loss": total_loss / max(num_batches, 1),
            "num_samples": len(all_labels)
        }
        
        if compute_metrics_fn and all_predictions and all_labels:
            metrics = compute_metrics_fn((all_predictions, all_labels))
            results.update(metrics)
        
        return results
    
    def evaluate_all_tasks(
        self,
        task_dataloaders: Dict[str, Any],
        device: torch.device,
        compute_metrics_fns: Optional[Dict[str, callable]] = None
    ) -> Dict[str, Any]:
        """Evaluate model on all tasks"""
        results = {}
        
        for task_name, dataloader in task_dataloaders.items():
            if task_name in self.config.tasks:
                compute_metrics_fn = None
                if compute_metrics_fns and task_name in compute_metrics_fns:
                    compute_metrics_fn = compute_metrics_fns[task_name]
                
                task_results = self.evaluate_task(
                    task_name=task_name,
                    dataloader=dataloader,
                    device=device,
                    compute_metrics_fn=compute_metrics_fn
                )
                
                results[task_name] = task_results
        
        # Compute average metrics
        if results:
            avg_loss = sum(r.get("loss", 0) for r in results.values()) / len(results)
            avg_accuracy = sum(r.get("accuracy", 0) for r in results.values()) / len(results)
            
            results["average"] = {
                "loss": avg_loss,
                "accuracy": avg_accuracy,
                "num_tasks": len(results)
            }
        
        return results
    
    def compute_task_interference(
        self,
        task_dataloaders: Dict[str, Any],
        device: torch.device,
        baseline_performance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Compute task interference metrics"""
        current_performance = {}
        
        # Evaluate current performance
        for task_name, dataloader in task_dataloaders.items():
            if task_name in self.config.tasks:
                results = self.evaluate_task(task_name, dataloader, device)
                current_performance[task_name] = results.get("accuracy", 0.0)
        
        # Compute interference if baseline provided
        interference_metrics = {}
        if baseline_performance:
            positive_transfer = 0
            negative_transfer = 0
            
            for task_name in current_performance:
                if task_name in baseline_performance:
                    current_acc = current_performance[task_name]
                    baseline_acc = baseline_performance[task_name]
                    
                    transfer = current_acc - baseline_acc
                    interference_metrics[f"{task_name}_transfer"] = transfer
                    
                    if transfer > 0:
                        positive_transfer += transfer
                    else:
                        negative_transfer += abs(transfer)
            
            interference_metrics.update({
                "positive_transfer": positive_transfer,
                "negative_transfer": negative_transfer,
                "net_transfer": positive_transfer - negative_transfer,
                "transfer_ratio": positive_transfer / max(negative_transfer, 1e-8)
            })
        
        interference_metrics["current_performance"] = current_performance
        
        return interference_metrics
    
    def save_pretrained(self, save_directory: str):
        """Save the multitask model"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save backbone
        self.backbone.save_pretrained(os.path.join(save_directory, "backbone"))
        
        # Save task heads
        torch.save(
            self.task_heads.state_dict(),
            os.path.join(save_directory, "task_heads.pt")
        )
        
        # Save configuration
        with open(os.path.join(save_directory, "multitask_config.json"), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
        
        # Save model state
        model_state = {
            "current_task": self.current_task,
            "training_step": self.training_step,
            "task_head_configs": self.task_heads.task_configs
        }
        
        with open(os.path.join(save_directory, "model_state.json"), 'w') as f:
            json.dump(model_state, f, indent=2)
        
        logger.info(f"Multitask model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ) -> "MultitaskModel":
        """Load multitask model from pretrained path"""
        # Load configuration
        config_path = os.path.join(model_path, "multitask_config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = MultitaskConfig.from_dict(config_dict)
        
        # Load tokenizer if not provided
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                logger.warning("Could not load tokenizer from model path")
        
        # Create model
        model = cls(config, tokenizer)
        
        # Load backbone
        backbone_path = os.path.join(model_path, "backbone")
        if os.path.exists(backbone_path):
            model.backbone = AutoModel.from_pretrained(backbone_path)
        
        # Load task heads
        task_heads_path = os.path.join(model_path, "task_heads.pt")
        if os.path.exists(task_heads_path):
            model.task_heads.load_state_dict(torch.load(task_heads_path, map_location="cpu"))
        
        # Load model state
        state_path = os.path.join(model_path, "model_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                model_state = json.load(f)
            
            model.current_task = model_state.get("current_task")
            model.training_step = model_state.get("training_step", 0)
        
        logger.info(f"Multitask model loaded from {model_path}")
        
        return model
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = self.task_heads.get_num_parameters()
        total_head_params = sum(head_params.values())
        
        stats = {
            "backbone_parameters": backbone_params,
            "task_head_parameters": head_params,
            "total_head_parameters": total_head_params,
            "total_parameters": backbone_params + total_head_params,
            "num_tasks": len(self.config.tasks),
            "task_names": self.config.get_task_names(),
            "current_task": self.current_task,
            "training_step": self.training_step,
            "architecture_type": self.config.architecture_type.value,
            "use_task_embeddings": self.config.use_task_embeddings
        }
        
        return stats
