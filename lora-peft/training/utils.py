"""
Training utilities for LoRA/PEFT implementation
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Callable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import wandb


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
    elif task_type == "regression":
        return compute_regression_metrics(predictions, labels)
    elif task_type == "generation":
        return compute_generation_metrics(predictions, labels)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


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


def compute_regression_metrics(predictions, labels) -> Dict[str, float]:
    """Compute regression metrics"""
    # Flatten predictions and labels
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Remove ignored labels (-100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Compute metrics
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(labels - predictions))
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def compute_generation_metrics(predictions, labels) -> Dict[str, float]:
    """Compute generation metrics (placeholder)"""
    # This would typically involve BLEU, ROUGE, etc.
    # For now, return perplexity-like metrics
    return {
        "perplexity": np.exp(np.mean(predictions)),
    }


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback"""
    
    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False
    ):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        
        self.best_metric = None
        self.patience_counter = 0
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Check for early stopping after evaluation"""
        if logs is None:
            return
        
        current_metric = logs.get(self.metric_for_best_model)
        if current_metric is None:
            return
        
        # Initialize best metric
        if self.best_metric is None:
            self.best_metric = current_metric
            return
        
        # Check if metric improved
        if self.greater_is_better:
            improved = current_metric > self.best_metric + self.early_stopping_threshold
        else:
            improved = current_metric < self.best_metric - self.early_stopping_threshold
        
        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Stop training if patience exceeded
        if self.patience_counter >= self.early_stopping_patience:
            logging.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
            control.should_training_stop = True


class WandBCallback(TrainerCallback):
    """Weights & Biases logging callback"""
    
    def __init__(self, project_name: str, run_name: Optional[str] = None, config: Optional[Dict] = None):
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.initialized = False
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize wandb at the beginning of training"""
        if not self.initialized:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                reinit=True
            )
            self.initialized = True
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Log metrics to wandb"""
        if self.initialized and logs:
            wandb.log(logs, step=state.global_step)
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Finish wandb run"""
        if self.initialized:
            wandb.finish()


def create_compute_metrics_fn(task_type: str) -> Callable:
    """Create a compute_metrics function for a specific task type"""
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, task_type)
    
    return compute_metrics_fn


def get_optimizer_and_scheduler(model, training_args: TrainingArguments):
    """Get optimizer and learning rate scheduler"""
    from transformers import get_linear_schedule_with_warmup
    from torch.optim import AdamW
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )
    
    # Calculate total training steps
    total_steps = (
        len(training_args.train_dataloader) 
        * training_args.num_train_epochs 
        // training_args.gradient_accumulation_steps
    )
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    return optimizer, scheduler


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
