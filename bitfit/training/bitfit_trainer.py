"""
BitFit trainer implementation
"""

import os
import logging
from typing import Dict, Optional, Any, Union
import torch
import torch.nn as nn
from transformers import (
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint

from ..config.training_config import TrainingConfig
from ..bitfit.bitfit_model import BitFitModel
from ..bitfit.bias_optimizer import BiasOptimizer
from .callbacks import BitFitCallbacks

logger = logging.getLogger(__name__)


class BitFitTrainer:
    """Trainer for BitFit models"""
    
    def __init__(
        self,
        model: BitFitModel,
        training_config: TrainingConfig,
        tokenizer=None
    ):
        self.model = model
        self.training_config = training_config
        self.tokenizer = tokenizer
        self.trainer = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        logger.info("BitFit trainer initialized")
    
    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for bias parameters"""
        if self.model.bitfit_config.use_different_lr_for_bias_types:
            # Use custom bias optimizer with different learning rates
            optimizer = BiasOptimizer(
                model=self.model.base_model,
                bitfit_config=self.model.bitfit_config,
                base_lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        else:
            # Use standard optimizer for all bias parameters
            trainable_params = [
                p for p in self.model.base_model.parameters() if p.requires_grad
            ]
            
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.training_config.learning_rate,
                weight_decay=self.model.bitfit_config.bias_weight_decay,
                betas=(self.training_config.adam_beta1, self.training_config.adam_beta2),
                eps=self.training_config.adam_epsilon
            )
        
        return optimizer
    
    def setup_scheduler(self, optimizer, num_training_steps: int):
        """Setup learning rate scheduler"""
        from transformers import get_scheduler
        
        scheduler = get_scheduler(
            name=self.training_config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        compute_metrics=None,
        data_collator=None
    ):
        """Train the BitFit model"""
        
        logger.info("Starting BitFit training...")
        
        # Print model comparison before training
        self.model.print_comparison()
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            adam_beta1=self.training_config.adam_beta1,
            adam_beta2=self.training_config.adam_beta2,
            adam_epsilon=self.training_config.adam_epsilon,
            max_grad_norm=self.training_config.max_grad_norm,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_ratio=self.training_config.warmup_ratio,
            warmup_steps=self.training_config.warmup_steps,
            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            logging_strategy=self.training_config.logging_strategy,
            logging_steps=self.training_config.logging_steps,
            report_to=self.training_config.report_to,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            dataloader_pin_memory=self.training_config.dataloader_pin_memory,
            remove_unused_columns=self.training_config.remove_unused_columns,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            seed=self.training_config.seed,
            data_seed=self.training_config.data_seed,
            run_name=self.training_config.run_name,
        )
        
        # Setup data collator
        if data_collator is None and self.tokenizer is not None:
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True
            )
        
        # Setup optimizer
        optimizer = self.setup_optimizer()
        
        # Calculate number of training steps
        num_training_steps = (
            len(train_dataset) // 
            (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        ) * training_args.num_train_epochs
        
        # Setup scheduler
        scheduler = self.setup_scheduler(optimizer, num_training_steps)
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        if self.training_config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold
                )
            )
        
        # BitFit-specific callbacks
        bitfit_callbacks = BitFitCallbacks(
            model=self.model,
            log_bias_statistics=self.model.bitfit_config.log_bias_statistics,
            track_bias_gradients=self.model.bitfit_config.track_bias_gradients
        )
        callbacks.append(bitfit_callbacks)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model.base_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=(optimizer, scheduler)
        )
        
        # Check for checkpoint
        checkpoint = None
        if self.training_config.resume_from_checkpoint is not None:
            checkpoint = self.training_config.resume_from_checkpoint
        elif os.path.isdir(training_args.output_dir):
            checkpoint = get_last_checkpoint(training_args.output_dir)
        
        # Train model
        logger.info("🚀 Starting training...")
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save model
        self.save_model()
        
        # Log training results
        logger.info("✅ Training completed!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        if hasattr(train_result, 'metrics'):
            for key, value in train_result.metrics.items():
                logger.info(f"{key}: {value}")
        
        return train_result
    
    def evaluate(self, eval_dataset=None, compute_metrics=None):
        """Evaluate the BitFit model"""
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info("Evaluating BitFit model...")
        
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        logger.info("Evaluation results:")
        for key, value in eval_result.items():
            logger.info(f"  {key}: {value}")
        
        return eval_result
    
    def save_model(self):
        """Save the BitFit model"""
        logger.info(f"Saving BitFit model to {self.training_config.output_dir}")
        
        # Save BitFit model (bias parameters only if configured)
        self.model.save_bitfit_model(self.training_config.output_dir)
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.training_config.output_dir)
        
        # Save training config
        import json
        with open(os.path.join(self.training_config.output_dir, "training_config.json"), 'w') as f:
            json.dump(self.training_config.__dict__, f, indent=2)
        
        logger.info("Model saved successfully")
    
    def predict(self, test_dataset):
        """Make predictions using the trained model"""
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info("Making predictions...")
        
        predictions = self.trainer.predict(test_dataset)
        
        return predictions
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        if self.trainer is None:
            return {}
        
        stats = {
            "parameter_efficiency": self.model.get_parameter_efficiency(),
            "bias_statistics": self.model.get_bias_statistics(),
            "comparison": self.model.compare_with_full_finetuning()
        }
        
        if hasattr(self.trainer.state, 'log_history'):
            stats["training_history"] = self.trainer.state.log_history
        
        return stats
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        training_config: Optional[TrainingConfig] = None,
        tokenizer=None
    ) -> "BitFitTrainer":
        """Create trainer from pretrained BitFit model"""
        
        # Load BitFit model
        model = BitFitModel.from_pretrained(model_path)
        
        # Use default training config if not provided
        if training_config is None:
            config_path = os.path.join(model_path, "training_config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                training_config = TrainingConfig(**config_dict)
            else:
                training_config = TrainingConfig()
        
        return cls(model, training_config, tokenizer)
