"""
PEFT Trainer implementation
"""

import os
import json
from typing import Optional, Dict, Any, List, Callable, Union
from datasets import Dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback as HFEarlyStoppingCallback
)
import torch
import logging

from ..models.peft_model import PEFTModelWrapper
from ..data.preprocessing import DataPreprocessor
from ..config.model_config import ModelConfig, PEFTConfig
from ..config.training_config import TrainingConfig
from .utils import (
    setup_logging,
    create_compute_metrics_fn,
    EarlyStoppingCallback,
    WandBCallback,
    save_training_config
)

logger = logging.getLogger(__name__)


class PEFTTrainer:
    """Main trainer class for PEFT models"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        peft_config: PEFTConfig,
        training_config: TrainingConfig,
        task_type: str = "classification"
    ):
        self.model_config = model_config
        self.peft_config = peft_config
        self.training_config = training_config
        self.task_type = task_type
        
        # Initialize components
        self.model_wrapper: Optional[PEFTModelWrapper] = None
        self.trainer: Optional[Trainer] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        
        # Setup logging
        setup_logging(
            log_file=os.path.join(training_config.output_dir, "training.log")
        )
    
    def setup_model(self) -> PEFTModelWrapper:
        """Setup PEFT model"""
        if self.model_wrapper is None:
            self.model_wrapper = PEFTModelWrapper(self.model_config, self.peft_config)
            self.model_wrapper.load_model()
            
            logger.info("PEFT model setup completed")
            logger.info(f"Model info: {self.model_wrapper.get_model_info()}")
        
        return self.model_wrapper
    
    def setup_data(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        preprocessor: Optional[DataPreprocessor] = None
    ) -> tuple:
        """Setup training and evaluation datasets"""
        
        if preprocessor is not None:
            self.preprocessor = preprocessor
        
        # Preprocess datasets
        if self.preprocessor is not None:
            logger.info("Preprocessing datasets...")
            
            train_dataset = self.preprocessor.preprocess_dataset(
                train_dataset,
                remove_columns=[col for col in train_dataset.column_names 
                              if col not in ["input_ids", "attention_mask", "labels"]]
            )
            
            if eval_dataset is not None:
                eval_dataset = self.preprocessor.preprocess_dataset(
                    eval_dataset,
                    remove_columns=[col for col in eval_dataset.column_names 
                                  if col not in ["input_ids", "attention_mask", "labels"]]
                )
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset is not None:
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List] = None
    ) -> Trainer:
        """Setup Hugging Face Trainer"""
        
        # Ensure model is setup
        if self.model_wrapper is None:
            self.setup_model()
        
        # Get model and tokenizer
        model = self.model_wrapper.peft_model
        tokenizer = self.model_wrapper.get_tokenizer()
        
        # Create training arguments
        training_args = TrainingArguments(**self.training_config.to_training_arguments())
        
        # Setup data collator
        if self.task_type == "generation":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # For causal LM
            )
        else:
            data_collator = DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding=True,
            )
        
        # Setup compute metrics
        if compute_metrics is None:
            compute_metrics = create_compute_metrics_fn(self.task_type)
        
        # Setup callbacks
        if callbacks is None:
            callbacks = []
        
        # Add early stopping if configured
        if self.training_config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold,
                    metric_for_best_model=self.training_config.metric_for_best_model,
                    greater_is_better=self.training_config.greater_is_better,
                )
            )
        
        # Add wandb callback if configured
        if self.training_config.report_to == "wandb":
            callbacks.append(
                WandBCallback(
                    project_name="lora-peft",
                    run_name=self.training_config.run_name,
                    config={
                        "model_config": self.model_config.__dict__,
                        "peft_config": self.peft_config.__dict__,
                        "training_config": self.training_config.__dict__,
                    }
                )
            )
        
        # Create trainer
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        
        logger.info("Trainer setup completed")
        return self.trainer
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the PEFT model"""
        
        logger.info("Starting PEFT training...")
        
        # Setup model
        self.setup_model()
        
        # Setup data
        train_dataset, eval_dataset = self.setup_data(
            train_dataset, eval_dataset, preprocessor
        )
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Save configuration
        self._save_configs()
        
        # Start training
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        self.save_model()
        
        # Log training results
        logger.info("Training completed!")
        logger.info(f"Training results: {train_result}")
        
        return train_result
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """Evaluate the model"""
        
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first or setup_trainer().")
        
        logger.info("Starting evaluation...")
        
        eval_results = self.trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=metric_key_prefix
        )
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def predict(
        self,
        test_dataset: Dataset,
        metric_key_prefix: str = "test"
    ) -> Dict[str, Any]:
        """Make predictions on test dataset"""
        
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first or setup_trainer().")
        
        logger.info("Making predictions...")
        
        predictions = self.trainer.predict(
            test_dataset=test_dataset,
            metric_key_prefix=metric_key_prefix
        )
        
        logger.info(f"Prediction results: {predictions.metrics}")
        return predictions
    
    def save_model(self, save_directory: Optional[str] = None):
        """Save the trained model"""
        
        if save_directory is None:
            save_directory = self.training_config.output_dir
        
        if self.model_wrapper is None:
            raise ValueError("Model not initialized")
        
        # Save PEFT model
        peft_save_dir = os.path.join(save_directory, "peft_model")
        self.model_wrapper.save_peft_model(peft_save_dir)
        
        # Save tokenizer
        tokenizer = self.model_wrapper.get_tokenizer()
        tokenizer.save_pretrained(save_directory)
        
        logger.info(f"Model saved to {save_directory}")
    
    def load_model(self, model_directory: str):
        """Load a trained model"""
        
        logger.info(f"Loading model from {model_directory}")
        
        # Setup model wrapper
        self.setup_model()
        
        # Load PEFT model
        peft_model_dir = os.path.join(model_directory, "peft_model")
        if os.path.exists(peft_model_dir):
            self.model_wrapper.load_peft_model(peft_model_dir)
        else:
            # Try loading from the directory directly
            self.model_wrapper.load_peft_model(model_directory)
        
        logger.info("Model loaded successfully")
    
    def _save_configs(self):
        """Save all configurations"""
        config_dir = os.path.join(self.training_config.output_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # Save configurations
        configs = {
            "model_config": self.model_config.__dict__,
            "peft_config": self.peft_config.__dict__,
            "training_config": self.training_config.__dict__,
            "task_type": self.task_type,
        }
        
        save_training_config(
            configs,
            os.path.join(config_dir, "training_config.json")
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if self.model_wrapper is None:
            return {"status": "Model not initialized"}
        
        return self.model_wrapper.get_model_info()
