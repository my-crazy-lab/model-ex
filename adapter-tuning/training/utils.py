"""
Training utilities for Adapter Tuning implementation
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Callable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    # Reduce noise from some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def compute_metrics(eval_pred, task_type: str = "classification") -> Dict[str, float]:
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred

    if task_type == "classification":
        return compute_classification_metrics(predictions, labels)
    elif task_type == "token_classification":
        return compute_token_classification_metrics(predictions, labels)
    else:
        # Default classification metrics
        return compute_classification_metrics(predictions, labels)


def compute_classification_metrics(predictions, labels) -> Dict[str, float]:
    """Compute classification metrics"""
    # Handle logits vs predictions
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Flatten if necessary
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Remove ignored labels (-100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def compute_token_classification_metrics(predictions, labels) -> Dict[str, float]:
    """Compute token classification metrics"""
    # Handle logits vs predictions
    if predictions.ndim > 2:
        predictions = np.argmax(predictions, axis=2)
    
    # Flatten and remove ignored labels
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                true_predictions.append(pred_id)
                true_labels.append(label_id)
    
    # Compute metrics
    accuracy = accuracy_score(true_labels, true_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


class AdapterTrainingCallback(TrainerCallback):
    """Custom callback for adapter training"""
    
    def __init__(
        self,
        log_adapter_info: bool = True,
        save_adapter_checkpoints: bool = True,
        adapter_checkpoint_dir: Optional[str] = None
    ):
        self.log_adapter_info = log_adapter_info
        self.save_adapter_checkpoints = save_adapter_checkpoints
        self.adapter_checkpoint_dir = adapter_checkpoint_dir
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Log adapter information at the beginning of training"""
        if self.log_adapter_info and hasattr(model, 'print_adapter_info'):
            model.print_adapter_info()
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Save adapter checkpoints"""
        if self.save_adapter_checkpoints and hasattr(model, 'save_adapters'):
            if self.adapter_checkpoint_dir:
                checkpoint_dir = os.path.join(
                    self.adapter_checkpoint_dir,
                    f"checkpoint-{state.global_step}"
                )
                model.save_adapters(checkpoint_dir)
                logging.info(f"Adapter checkpoint saved to {checkpoint_dir}")
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Log adapter-specific metrics"""
        if logs and hasattr(model, 'get_adapter_parameters'):
            # Log adapter parameter statistics
            adapter_params = model.get_adapter_parameters()
            if adapter_params:
                total_adapter_params = sum(p.numel() for p in adapter_params)
                logs["adapter_params"] = total_adapter_params
                
                # Log gradient norms for adapters
                adapter_grad_norm = 0.0
                for p in adapter_params:
                    if p.grad is not None:
                        adapter_grad_norm += p.grad.data.norm(2).item() ** 2
                adapter_grad_norm = adapter_grad_norm ** 0.5
                logs["adapter_grad_norm"] = adapter_grad_norm


def create_compute_metrics_fn(task_type: str) -> Callable:
    """Create a compute_metrics function for a specific task type"""
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, task_type)
    
    return compute_metrics_fn


def get_optimizer_for_adapters(model, learning_rate: float, weight_decay: float = 0.01):
    """Get optimizer specifically configured for adapter parameters"""
    try:
        from torch.optim import AdamW
    except ImportError:
        # Fallback if torch is not available
        raise ImportError("PyTorch is required for optimizer creation")
    
    # Get only adapter parameters
    adapter_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "adapter" in name.lower():
                adapter_params.append(param)
            else:
                base_params.append(param)
    
    # Create optimizer with different learning rates
    optimizer_grouped_parameters = []
    
    if adapter_params:
        optimizer_grouped_parameters.append({
            "params": adapter_params,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        })
    
    if base_params:
        optimizer_grouped_parameters.append({
            "params": base_params,
            "lr": learning_rate * 0.1,  # Lower LR for base model
            "weight_decay": weight_decay,
        })
    
    return AdamW(optimizer_grouped_parameters)


def save_training_config(config: Dict[str, Any], save_path: str):
    """Save training configuration to file"""
    import json
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logging.info(f"Training configuration saved to {save_path}")


def load_training_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from file"""
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logging.info(f"Training configuration loaded from {config_path}")
    return config
