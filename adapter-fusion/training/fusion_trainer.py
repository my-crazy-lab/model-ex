"""
Fusion trainer for training fusion layers with pre-trained adapters
"""

import os
import logging
from typing import Dict, Optional
import torch
import torch.optim as optim
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from ..config import ModelConfig, FusionConfig, TrainingConfig
from ..fusion import FusionModel, AdapterManager

logger = logging.getLogger(__name__)


class FusionTrainer:
    """Trainer for adapter fusion"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        fusion_config: FusionConfig,
        training_config: TrainingConfig
    ):
        self.model_config = model_config
        self.fusion_config = fusion_config
        self.training_config = training_config
        
        self.fusion_model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup_model(self, adapter_paths: Dict[str, str]) -> FusionModel:
        """Setup fusion model with pre-trained adapters"""
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer_name_or_path
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get hidden size from model config
        from transformers import AutoConfig
        model_config_hf = AutoConfig.from_pretrained(self.model_config.model_name_or_path)
        hidden_size = getattr(model_config_hf, 'hidden_size', 768)
        
        # Create adapter manager from paths
        adapter_manager = AdapterManager.from_adapter_paths(
            hidden_size=hidden_size,
            adapter_paths=adapter_paths,
            freeze_adapters=self.fusion_config.freeze_adapters_during_fusion
        )
        
        # Create fusion model
        self.fusion_model = FusionModel(
            model_config=self.model_config,
            fusion_config=self.fusion_config,
            adapter_manager=adapter_manager
        )
        
        logger.info("✅ Fusion model setup completed")
        self.fusion_model.print_model_info()
        
        return self.fusion_model
    
    def train_fusion(
        self,
        train_dataset,
        eval_dataset=None,
        preprocessor=None
    ):
        """Train fusion layer"""
        
        if self.fusion_model is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        logger.info("🚀 Starting fusion training...")
        
        # Preprocess datasets
        if preprocessor is not None:
            logger.info("📊 Preprocessing datasets...")
            train_dataset = preprocessor.preprocess_dataset(train_dataset)
            if eval_dataset is not None:
                eval_dataset = preprocessor.preprocess_dataset(eval_dataset)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.fusion_checkpoint_dir,
            num_train_epochs=self.training_config.fusion_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            learning_rate=self.training_config.fusion_learning_rate,
            weight_decay=self.training_config.weight_decay,
            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            logging_steps=self.training_config.logging_steps,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            warmup_steps=self.training_config.fusion_warmup_steps,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            remove_unused_columns=self.training_config.remove_unused_columns,
            fp16=self.training_config.fp16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            seed=self.training_config.seed,
        )
        
        # Create data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Create optimizer
        if self.fusion_config.freeze_adapters_during_fusion:
            # Only train fusion layer
            optimizer = optim.AdamW(
                self.fusion_model.get_fusion_parameters(),
                lr=self.training_config.fusion_learning_rate,
                weight_decay=self.training_config.weight_decay
            )
            logger.info("Training fusion layer only (adapters frozen)")
        else:
            # Train both fusion and adapters
            fusion_params = self.fusion_model.get_fusion_parameters()
            adapter_params = self.fusion_model.get_adapter_parameters()
            
            optimizer = optim.AdamW([
                {
                    'params': fusion_params, 
                    'lr': self.training_config.fusion_learning_rate,
                    'weight_decay': self.training_config.weight_decay
                },
                {
                    'params': adapter_params, 
                    'lr': self.training_config.adapter_learning_rate,
                    'weight_decay': self.training_config.weight_decay
                }
            ])
            logger.info("Training both fusion layer and adapters")
        
        # Create compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            if predictions.ndim > 1:
                predictions = predictions.argmax(axis=1)
            
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
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
        
        # Create trainer
        self.trainer = Trainer(
            model=self.fusion_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None)  # (optimizer, scheduler)
        )
        
        # Start training
        logger.info("🏋️ Fusion training started...")
        train_result = self.trainer.train()
        
        # Save fusion model
        self.save_fusion_model()
        
        logger.info("✅ Fusion training completed!")
        return train_result
    
    def evaluate(self, eval_dataset=None, preprocessor=None):
        """Evaluate fusion model"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train_fusion() first.")
        
        if eval_dataset is not None and preprocessor is not None:
            eval_dataset = preprocessor.preprocess_dataset(eval_dataset)
        
        eval_result = self.trainer.evaluate(eval_dataset)
        logger.info(f"Evaluation results: {eval_result}")
        return eval_result
    
    def save_fusion_model(self):
        """Save fusion model components"""
        
        # Create output directories
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        os.makedirs(self.training_config.fusion_checkpoint_dir, exist_ok=True)
        
        # Save fusion layer
        self.fusion_model.save_fusion(self.training_config.fusion_checkpoint_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.training_config.output_dir)
        
        # Save model config
        import json
        with open(os.path.join(self.training_config.output_dir, "model_config.json"), 'w') as f:
            json.dump(self.model_config.__dict__, f, indent=2)
        
        # Save fusion config
        with open(os.path.join(self.training_config.output_dir, "fusion_config.json"), 'w') as f:
            json.dump(self.fusion_config.to_dict(), f, indent=2)
        
        logger.info(f"💾 Fusion model saved to {self.training_config.output_dir}")
    
    def load_fusion_model(self, model_path: str):
        """Load pre-trained fusion model"""
        
        # Load configs
        import json
        
        model_config_path = os.path.join(model_path, "model_config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                model_config_dict = json.load(f)
            self.model_config = ModelConfig(**model_config_dict)
        
        fusion_config_path = os.path.join(model_path, "fusion_config.json")
        if os.path.exists(fusion_config_path):
            with open(fusion_config_path, 'r') as f:
                fusion_config_dict = json.load(f)
            self.fusion_config = FusionConfig.from_dict(fusion_config_dict)
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Setup model (assuming adapter paths are in fusion config)
        if hasattr(self.fusion_config, 'adapter_paths') and self.fusion_config.adapter_paths:
            adapter_paths = {
                name: path for name, path in zip(
                    self.fusion_config.adapter_names,
                    self.fusion_config.adapter_paths
                )
            }
            self.setup_model(adapter_paths)
            
            # Load fusion weights
            fusion_path = os.path.join(model_path, "fusion")
            if os.path.exists(fusion_path):
                self.fusion_model.load_fusion(fusion_path)
        
        logger.info(f"Loaded fusion model from {model_path}")
        return self.fusion_model
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        training_config: Optional[TrainingConfig] = None
    ) -> "FusionTrainer":
        """Create trainer from pre-trained model"""
        
        if training_config is None:
            training_config = TrainingConfig()
        
        # Create dummy configs (will be overridden by load)
        model_config = ModelConfig()
        fusion_config = FusionConfig()
        
        trainer = cls(model_config, fusion_config, training_config)
        trainer.load_fusion_model(model_path)
        
        return trainer
