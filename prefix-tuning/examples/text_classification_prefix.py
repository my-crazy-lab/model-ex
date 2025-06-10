"""
Complete prefix tuning text classification example
"""

import argparse
import logging
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
import os

from config import PrefixConfig, ModelConfig
from prefix_tuning import PrefixTuningModel
from training import PrefixTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_dataset(dataset_name: str = "imdb", sample_size: int = 1000):
    """Load and sample dataset for demonstration"""
    
    logger.info(f"Loading {dataset_name} dataset...")
    
    if dataset_name == "imdb":
        dataset = load_dataset("imdb")
        
        # Sample data for quick demonstration
        train_data = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
        test_data = dataset["test"].select(range(min(sample_size // 4, len(dataset["test"]))))
        
        return train_data, test_data, 2  # 2 classes
    
    elif dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")
        
        train_data = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
        test_data = dataset["validation"].select(range(min(sample_size // 4, len(dataset["validation"]))))
        
        return train_data, test_data, 2  # 2 classes
    
    elif dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
        
        train_data = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
        test_data = dataset["test"].select(range(min(sample_size // 4, len(dataset["test"]))))
        
        return train_data, test_data, 4  # 4 classes
    
    else:
        # Create dummy dataset
        logger.info("Creating dummy dataset...")
        
        texts = [
            "This movie is amazing and wonderful!",
            "Great acting and fantastic storyline.",
            "Excellent cinematography and direction.",
            "Terrible film, complete waste of time.",
            "Boring and very poorly made movie.",
            "Worst movie I have ever seen."
        ] * (sample_size // 6)
        
        labels = [1, 1, 1, 0, 0, 0] * (sample_size // 6)
        
        # Create datasets
        train_size = int(0.8 * len(texts))
        train_data = Dataset.from_dict({
            "text": texts[:train_size],
            "label": labels[:train_size]
        })
        test_data = Dataset.from_dict({
            "text": texts[train_size:],
            "label": labels[train_size:]
        })
        
        return train_data, test_data, 2  # 2 classes


def preprocess_dataset(dataset, tokenizer, max_length=128):
    """Preprocess dataset for training"""
    
    def preprocess_function(examples):
        # Handle different text column names
        if "text" in examples:
            texts = examples["text"]
        elif "sentence" in examples:
            texts = examples["sentence"]
        else:
            raise ValueError("No text column found in dataset")
        
        # Tokenize texts
        result = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        
        # Add labels
        if "label" in examples:
            result["labels"] = examples["label"]
        
        return result
    
    return dataset.map(
        preprocess_function,
        batched=True,
        desc="Preprocessing dataset"
    )


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    
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


def compare_with_full_finetuning(
    model_config: ModelConfig,
    train_dataset,
    eval_dataset,
    tokenizer
):
    """Compare prefix tuning with full fine-tuning (simulation)"""
    
    logger.info("🔄 Simulating full fine-tuning comparison...")
    
    # Create standard model for comparison
    full_model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=model_config.num_labels
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in full_model.parameters())
    
    # Simulate training metrics
    full_training_time = len(train_dataset) * 0.001  # seconds per sample
    full_memory_mb = total_params * 4 / (1024 * 1024)  # float32
    
    return {
        "full_finetuning": {
            "trainable_params": total_params,
            "training_time_estimate": full_training_time,
            "memory_mb": full_memory_mb,
            "parameter_efficiency": 100.0
        }
    }


def test_different_prefix_lengths(
    model_config: ModelConfig,
    train_dataset,
    eval_dataset,
    tokenizer,
    prefix_lengths: list = [5, 10, 15, 20]
):
    """Test different prefix lengths"""
    
    logger.info("🧪 Testing different prefix lengths...")
    
    results = {}
    
    for prefix_length in prefix_lengths:
        logger.info(f"Testing prefix length: {prefix_length}")
        
        # Create prefix config
        prefix_config = PrefixConfig(
            prefix_length=prefix_length,
            prefix_dropout=0.1,
            reparameterization=True,
            prefix_learning_rate=1e-3,
            num_labels=model_config.num_labels
        )
        
        # Create model
        model = PrefixTuningModel(model_config, prefix_config, tokenizer=tokenizer)
        
        # Get parameter efficiency
        efficiency = model.get_parameter_efficiency()
        
        results[prefix_length] = {
            "prefix_parameters": efficiency["prefix_parameters"],
            "parameter_efficiency": efficiency["parameter_efficiency"],
            "reduction_factor": efficiency["reduction_factor"]
        }
        
        logger.info(f"Prefix length {prefix_length}: {efficiency['prefix_parameters']:,} params "
                   f"({efficiency['parameter_efficiency']:.4f}%)")
    
    return results


def main():
    """Main prefix tuning example"""
    
    parser = argparse.ArgumentParser(description="Prefix Tuning Text Classification Example")
    parser.add_argument("--dataset", default="dummy", choices=["imdb", "sst2", "ag_news", "dummy"], 
                       help="Dataset to use")
    parser.add_argument("--model_name", default="distilbert-base-uncased", 
                       help="Model name")
    parser.add_argument("--sample_size", type=int, default=500, 
                       help="Sample size for demonstration")
    parser.add_argument("--output_dir", default="./prefix_results", 
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                       help="Learning rate")
    parser.add_argument("--prefix_length", type=int, default=10, 
                       help="Prefix length")
    parser.add_argument("--prefix_dropout", type=float, default=0.1, 
                       help="Prefix dropout")
    parser.add_argument("--reparameterization", action="store_true", 
                       help="Use reparameterization")
    parser.add_argument("--test_prefix_lengths", action="store_true", 
                       help="Test different prefix lengths")
    
    args = parser.parse_args()
    
    logger.info("🎯 Starting Prefix Tuning text classification example...")
    
    # Load dataset
    train_dataset, test_dataset, num_labels = load_sample_dataset(args.dataset, args.sample_size)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Number of labels: {num_labels}")
    
    # Setup configurations
    model_config = ModelConfig(
        model_name_or_path=args.model_name,
        task_type="classification",
        num_labels=num_labels,
        max_length=128
    )
    
    prefix_config = PrefixConfig(
        prefix_length=args.prefix_length,
        prefix_dropout=args.prefix_dropout,
        reparameterization=args.reparameterization,
        reparameterization_type="mlp",
        reparameterization_hidden_size=512,
        prefix_learning_rate=args.learning_rate,
        num_labels=num_labels,
        freeze_base_model=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    
    # Preprocess datasets
    logger.info("📊 Preprocessing datasets...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer, model_config.max_length)
    test_dataset = preprocess_dataset(test_dataset, tokenizer, model_config.max_length)
    
    # Test different prefix lengths if requested
    if args.test_prefix_lengths:
        prefix_length_results = test_different_prefix_lengths(
            model_config, train_dataset, test_dataset, tokenizer
        )
        
        logger.info("\n📊 PREFIX LENGTH COMPARISON:")
        logger.info("=" * 60)
        for length, results in prefix_length_results.items():
            logger.info(f"Length {length:2d}: {results['prefix_parameters']:6,} params "
                       f"({results['parameter_efficiency']:6.4f}%) "
                       f"- {results['reduction_factor']:6.1f}x reduction")
    
    # Create prefix tuning model
    logger.info("🔧 Creating Prefix Tuning model...")
    model = PrefixTuningModel(model_config, prefix_config, tokenizer=tokenizer)
    
    # Print parameter summary
    model.print_parameter_summary()
    
    # Compare with full fine-tuning
    comparison_baseline = compare_with_full_finetuning(
        model_config, train_dataset, test_dataset, tokenizer
    )
    
    # Create trainer
    from training import PrefixTrainer
    
    training_config = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,
        "learning_rate": args.learning_rate,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "save_steps": 100,
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
        "seed": 42
    }
    
    trainer = PrefixTrainer(model, training_config, tokenizer)
    
    # Train model
    logger.info("🏋️ Training Prefix Tuning model...")
    train_result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Evaluate model
    logger.info("📈 Evaluating model...")
    eval_result = trainer.evaluate()
    
    # Get efficiency metrics
    efficiency = model.get_parameter_efficiency()
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("📊 PREFIX TUNING RESULTS")
    logger.info("="*60)
    
    # Model efficiency
    logger.info(f"Parameter Efficiency:")
    logger.info(f"  Total parameters: {efficiency['total_parameters']:,}")
    logger.info(f"  Prefix parameters: {efficiency['prefix_parameters']:,}")
    logger.info(f"  Trainable parameters: {efficiency['total_trainable_parameters']:,}")
    logger.info(f"  Parameter efficiency: {efficiency['parameter_efficiency']:.4f}%")
    logger.info(f"  Reduction factor: {efficiency['reduction_factor']:.1f}x")
    
    # Performance
    logger.info(f"\nPerformance:")
    logger.info(f"  Training loss: {train_result.training_loss:.4f}")
    logger.info(f"  Eval accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
    logger.info(f"  Eval F1: {eval_result.get('eval_f1', 0):.4f}")
    
    # Comparison with full fine-tuning
    baseline = comparison_baseline['full_finetuning']
    logger.info(f"\nComparison with Full Fine-tuning:")
    logger.info(f"  Full FT trainable params: {baseline['trainable_params']:,}")
    logger.info(f"  Prefix trainable params: {efficiency['total_trainable_parameters']:,}")
    logger.info(f"  Parameter reduction: {efficiency['reduction_factor']:.1f}x")
    logger.info(f"  Memory reduction: ~{efficiency['reduction_factor']:.1f}x")
    
    # Configuration summary
    logger.info(f"\nConfiguration:")
    logger.info(f"  Prefix length: {prefix_config.prefix_length}")
    logger.info(f"  Reparameterization: {prefix_config.reparameterization}")
    logger.info(f"  Dropout: {prefix_config.prefix_dropout}")
    logger.info(f"  Learning rate: {prefix_config.prefix_learning_rate}")
    
    logger.info("="*60)
    
    # Test inference
    logger.info("🧪 Testing inference...")
    test_texts = [
        "This movie is absolutely fantastic and amazing!",
        "Terrible film, complete waste of time and money.",
        "An okay movie, nothing particularly special about it."
    ]
    
    model.eval()
    for text in test_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_config.max_length
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs['logits']
            
            predictions = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        if num_labels == 2:
            sentiment = "Positive" if predicted_class == 1 else "Negative"
        else:
            sentiment = f"Class {predicted_class}"
        
        logger.info(f"Text: '{text}'")
        logger.info(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
        logger.info("-" * 40)
    
    # Save model
    model.save_prefix_tuning_model(args.output_dir)
    
    # Save results
    results_summary = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "prefix_config": prefix_config.to_dict(),
        "efficiency": efficiency,
        "performance": {
            "training_loss": train_result.training_loss,
            "eval_accuracy": eval_result.get('eval_accuracy', 0),
            "eval_f1": eval_result.get('eval_f1', 0)
        },
        "comparison": comparison_baseline
    }
    
    if args.test_prefix_lengths:
        results_summary["prefix_length_comparison"] = prefix_length_results
    
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("✅ Prefix tuning example completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Parameter efficiency: {efficiency['parameter_efficiency']:.4f}%")
    logger.info(f"Reduction factor: {efficiency['reduction_factor']:.1f}x")
    logger.info(f"Final accuracy: {eval_result.get('eval_accuracy', 0):.4f}")


if __name__ == "__main__":
    main()
