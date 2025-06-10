"""
Adapter Trainer implementation
"""

import os
import json
from typing import Optional, Dict, Any, List, Callable, Union
from datasets import Dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    AutoTokenizer
)
import torch
import logging

from ..adapters.adapter_model import AdapterModel
from ..data.preprocessing import DataPreprocessor
from ..config.model_config import ModelConfig
from ..config.adapter_config import AdapterConfig
from ..config.training_config import TrainingConfig
from .utils import (
    setup_logging,
    create_compute_metrics_fn,
    AdapterTrainingCallback,
    get_optimizer_for_adapters
)

logger = logging.getLogger(__name__)


class AdapterTrainer:
    """Main trainer class for Adapter Tuning"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        adapter_config: AdapterConfig,
        training_config: TrainingConfig,
        task_type: str = "classification"
    ):
        self.model_config = model_config
        self.adapter_config = adapter_config
        self.training_config = training_config
        self.task_type = task_type
        
        # Initialize components
        self.adapter_model: Optional[AdapterModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.trainer: Optional[Trainer] = None
        
        # Setup logging
        setup_logging(
            log_file=os.path.join(training_config.output_dir, "training.log")
        )
    
    def setup_model(self) -> AdapterModel:
        """Setup adapter model and tokenizer"""
        if self.adapter_model is None:
            logger.info("Setting up adapter model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.tokenizer_name_or_path,
                cache_dir=self.model_config.cache_dir,
                trust_remote_code=self.model_config.trust_remote_code
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Create adapter model
            self.adapter_model = AdapterModel(
                self.model_config,
                self.adapter_config
            )
            
            # Resize token embeddings if tokenizer was modified
            if len(self.tokenizer) != self.adapter_model.base_model.config.vocab_size:
                self.adapter_model.base_model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info("Adapter model setup completed")
            self.adapter_model.print_adapter_info()
        
        return self.adapter_model
    
    def setup_data(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        preprocessor: Optional[DataPreprocessor] = None
    ) -> tuple:
        """Setup training and evaluation datasets"""
        
        if preprocessor is not None:
            logger.info("Preprocessing datasets...")
            
            # Remove columns that won't be used
            columns_to_remove = [
                col for col in train_dataset.column_names 
                if col not in ["input_ids", "attention_mask", "labels", "token_type_ids"]
            ]
            
            train_dataset = preprocessor.preprocess_dataset(
                train_dataset,
                remove_columns=columns_to_remove
            )
            
            if eval_dataset is not None:
                eval_dataset = preprocessor.preprocess_dataset(
                    eval_dataset,
                    remove_columns=columns_to_remove
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
        """Setup Hugging Face Trainer for adapter training"""
        
        # Ensure model is setup
        if self.adapter_model is None:
            self.setup_model()
        
        # Create training arguments
        training_args = TrainingArguments(**self.training_config.to_training_arguments())
        
        # Setup data collator based on task type
        if self.task_type in ["ner", "token_classification"]:
            data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                padding=True,
            )
        else:
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
            )
        
        # Setup compute metrics
        if compute_metrics is None:
            compute_metrics = create_compute_metrics_fn(self.task_type)
        
        # Setup callbacks
        if callbacks is None:
            callbacks = []
        
        # Add adapter-specific callback
        callbacks.append(
            AdapterTrainingCallback(
                log_adapter_info=True,
                save_adapter_checkpoints=self.training_config.save_adapters_only,
                adapter_checkpoint_dir=self.training_config.adapter_checkpoint_dir
            )
        )
        
        # Create custom optimizer if needed
        optimizers = (None, None)  # Default
        if self.training_config.train_adapters_only:
            optimizer = get_optimizer_for_adapters(
                self.adapter_model,
                self.training_config.adapter_learning_rate,
                self.training_config.weight_decay
            )
            optimizers = (optimizer, None)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.adapter_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers
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
        """Train the adapter model"""
        
        logger.info("Starting adapter training...")
        
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
        """Evaluate the adapter model"""
        
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
        """Save the trained adapter model"""
        
        if save_directory is None:
            save_directory = self.training_config.output_dir
        
        if self.adapter_model is None:
            raise ValueError("Adapter model not initialized")
        
        # Save adapters only (much smaller)
        if self.training_config.save_adapters_only:
            adapter_save_dir = os.path.join(save_directory, "adapters")
            self.adapter_model.save_adapters(adapter_save_dir)
        else:
            # Save full model
            self.trainer.save_model(save_directory)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        logger.info(f"Model saved to {save_directory}")
    
    def load_model(self, model_directory: str):
        """Load a trained adapter model"""
        
        logger.info(f"Loading adapter model from {model_directory}")
        
        # Setup model first
        self.setup_model()
        
        # Load adapters
        adapter_dir = os.path.join(model_directory, "adapters")
        if os.path.exists(adapter_dir):
            self.adapter_model.load_adapters(adapter_dir)
        else:
            # Try loading from the directory directly
            self.adapter_model.load_adapters(model_directory)
        
        logger.info("Adapter model loaded successfully")
    
    def _save_configs(self):
        """Save all configurations"""
        config_dir = os.path.join(self.training_config.output_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # Save configurations
        configs = {
            "model_config": self.model_config.__dict__,
            "adapter_config": self.adapter_config.to_dict(),
            "training_config": self.training_config.__dict__,
            "task_type": self.task_type,
        }
        
        config_path = os.path.join(config_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(configs, f, indent=2, default=str)
        
        logger.info(f"Configurations saved to {config_dir}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if self.adapter_model is None:
            return {"status": "Model not initialized"}
        
        return self.adapter_model.adapter_info
    
    def switch_adapter(self, adapter_name: str):
        """Switch to a different adapter (for multi-adapter models)"""
        if self.adapter_model is None:
            raise ValueError("Adapter model not initialized")
        
        # This would be implemented for multi-adapter scenarios
        logger.info(f"Switching to adapter: {adapter_name}")
        # Implementation depends on multi-adapter architecture
    
    def add_adapter(self, adapter_name: str, adapter_config: AdapterConfig):
        """Add a new adapter to the model"""
        if self.adapter_model is None:
            raise ValueError("Adapter model not initialized")
        
        # This would be implemented for dynamic adapter addition
        logger.info(f"Adding new adapter: {adapter_name}")
        # Implementation depends on multi-adapter architecture
