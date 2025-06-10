"""
Complete Adapter Fusion training example
"""

import os
import argparse
import logging
from datasets import load_dataset, Dataset
import torch

from config import ModelConfig, FusionConfig, TrainingConfig, AdapterConfig
from fusion import FusionModel, AdapterManager
from training import FusionTrainer
from adapters import BottleneckAdapter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_adapters(save_dir: str, tasks: dict, hidden_size: int = 768):
    """Create dummy pre-trained adapters for demonstration"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    for task_name, task_info in tasks.items():
        task_dir = os.path.join(save_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        # Create adapter config
        adapter_config = AdapterConfig(
            adapter_size=64,
            adapter_dropout=0.1,
            adapter_activation="relu",
            task_name=task_name,
            task_type="classification"
        )
        
        # Save adapter config
        import json
        with open(os.path.join(task_dir, "adapter_config.json"), 'w') as f:
            json.dump(adapter_config.to_dict(), f, indent=2)
        
        # Create and save dummy adapter weights
        adapter = BottleneckAdapter(
            input_size=hidden_size,
            adapter_size=adapter_config.adapter_size,
            dropout=adapter_config.adapter_dropout,
            activation=adapter_config.adapter_activation
        )
        
        torch.save(adapter.state_dict(), os.path.join(task_dir, "adapter_model.bin"))
        
        logger.info(f"Created dummy adapter for task '{task_name}' at {task_dir}")


def create_multi_task_dataset(tasks: dict, train_size_per_task: int = 500):
    """Create a simple multi-task dataset"""
    
    multi_task_data = []
    
    for task_name, task_info in tasks.items():
        if task_name == "sentiment":
            # Load IMDB dataset
            try:
                dataset = load_dataset("imdb")
                examples = dataset["train"].select(range(min(train_size_per_task, len(dataset["train"]))))
                
                for example in examples:
                    multi_task_data.append({
                        "text": example["text"],
                        "labels": example["label"],
                        "task": task_name
                    })
                    
                logger.info(f"Added {len(examples)} examples for task '{task_name}'")
                
            except Exception as e:
                logger.warning(f"Could not load IMDB dataset: {e}")
                # Create dummy data
                for i in range(train_size_per_task):
                    multi_task_data.append({
                        "text": f"This is a sample text for sentiment analysis {i}",
                        "labels": i % 2,
                        "task": task_name
                    })
        
        elif task_name == "nli":
            # Create dummy NLI data
            for i in range(train_size_per_task):
                multi_task_data.append({
                    "text": f"Premise: Sample premise {i}. Hypothesis: Sample hypothesis {i}.",
                    "labels": i % 3,  # 3 classes for NLI
                    "task": task_name
                })
            
            logger.info(f"Created {train_size_per_task} dummy examples for task '{task_name}'")
    
    return Dataset.from_list(multi_task_data)


class SimplePreprocessor:
    """Simple preprocessor for multi-task data"""
    
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def preprocess_dataset(self, dataset):
        """Preprocess dataset"""
        def preprocess_function(examples):
            # Tokenize texts
            result = self.tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )
            
            # Add labels
            result["labels"] = examples["labels"]
            
            return result
        
        return dataset.map(
            preprocess_function,
            batched=True,
            desc="Preprocessing dataset"
        )


def main():
    """Main fusion training example"""
    
    parser = argparse.ArgumentParser(description="Adapter Fusion Training Example")
    parser.add_argument("--output_dir", default="./fusion_results", help="Output directory")
    parser.add_argument("--adapter_dir", default="./dummy_adapters", help="Adapter directory")
    parser.add_argument("--train_size_per_task", type=int, default=100, help="Training size per task")
    parser.add_argument("--fusion_method", default="attention", choices=["attention", "weighted", "gating"])
    parser.add_argument("--epochs", type=int, default=2, help="Number of fusion training epochs")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Adapter Fusion training example...")
    
    # Define tasks
    tasks = {
        "sentiment": {"num_labels": 2, "dataset": "imdb"},
        "nli": {"num_labels": 3, "dataset": "dummy"},
    }
    
    # Step 1: Create dummy adapters (in real scenario, these would be pre-trained)
    logger.info("📦 Creating dummy pre-trained adapters...")
    create_dummy_adapters(args.adapter_dir, tasks)
    
    # Step 2: Setup configurations
    model_config = ModelConfig(
        model_name_or_path="distilbert-base-uncased",  # Smaller model for demo
        num_labels=2,  # Will be overridden per task
        max_length=128,
        multi_task=True,
        task_names=list(tasks.keys())
    )
    
    fusion_config = FusionConfig(
        fusion_method=args.fusion_method,
        num_attention_heads=8,
        fusion_dropout=0.1,
        freeze_adapters_during_fusion=True,
        adapter_names=list(tasks.keys()),
        adapter_paths=[os.path.join(args.adapter_dir, task) for task in tasks.keys()]
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        fusion_epochs=args.epochs,
        fusion_learning_rate=1e-4,
        per_device_train_batch_size=8,
        eval_steps=50,
        save_steps=50,
        logging_steps=25
    )
    
    # Step 3: Create multi-task dataset
    logger.info("📊 Creating multi-task dataset...")
    train_dataset = create_multi_task_dataset(tasks, args.train_size_per_task)
    eval_dataset = create_multi_task_dataset(tasks, args.train_size_per_task // 4)
    
    logger.info(f"Created train dataset with {len(train_dataset)} examples")
    logger.info(f"Created eval dataset with {len(eval_dataset)} examples")
    
    # Step 4: Setup adapter paths
    adapter_paths = {
        task_name: os.path.join(args.adapter_dir, task_name)
        for task_name in tasks.keys()
    }
    
    # Step 5: Setup fusion trainer
    trainer = FusionTrainer(
        model_config=model_config,
        fusion_config=fusion_config,
        training_config=training_config
    )
    
    # Step 6: Setup model with adapters
    logger.info("🔧 Setting up fusion model...")
    fusion_model = trainer.setup_model(adapter_paths)
    
    # Step 7: Setup preprocessor
    preprocessor = SimplePreprocessor(
        tokenizer=trainer.tokenizer,
        max_length=model_config.max_length
    )
    
    # Step 8: Train fusion
    logger.info("🏋️ Starting fusion training...")
    train_result = trainer.train_fusion(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocessor=preprocessor
    )
    
    logger.info(f"Fusion training completed: {train_result}")
    
    # Step 9: Test fusion model
    logger.info("🧪 Testing fusion model...")
    
    test_texts = [
        "This movie is absolutely fantastic!",
        "Terrible film, complete waste of time.",
        "The movie was okay, nothing special."
    ]
    
    fusion_model.eval()
    
    for text in test_texts:
        inputs = trainer.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=model_config.max_length
        )
        
        with torch.no_grad():
            # Test with different adapter combinations
            outputs_sentiment = fusion_model(**inputs, adapter_names=["sentiment"])
            outputs_all = fusion_model(**inputs, adapter_names=list(tasks.keys()))
            
            sentiment_probs = torch.softmax(outputs_sentiment.logits, dim=-1)
            all_probs = torch.softmax(outputs_all.logits, dim=-1)
            
            print(f"\nText: '{text}'")
            print(f"Sentiment only: {sentiment_probs.cpu().numpy()}")
            print(f"Fused (all adapters): {all_probs.cpu().numpy()}")
            print(f"Prediction: {'Positive' if sentiment_probs[0][1] > 0.5 else 'Negative'}")
    
    # Step 10: Print model information
    logger.info("\n📊 Final model information:")
    fusion_model.print_model_info()
    
    logger.info("✅ Adapter Fusion example completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
