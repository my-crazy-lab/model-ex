"""
Complete BitFit text classification example
"""

import argparse
import logging
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from config import ModelConfig, BitFitConfig, TrainingConfig
from bitfit import BitFitModel
from training import BitFitTrainer

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
        
        return train_data, test_data
    
    elif dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")
        
        train_data = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
        test_data = dataset["validation"].select(range(min(sample_size // 4, len(dataset["validation"]))))
        
        return train_data, test_data
    
    else:
        # Create dummy dataset
        logger.info("Creating dummy dataset...")
        
        texts = [
            "This movie is amazing!",
            "Great acting and storyline.",
            "Wonderful cinematography.",
            "Terrible film, waste of time.",
            "Boring and poorly made.",
            "Worst movie ever made."
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
        
        return train_data, test_data


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
    tokenizer,
    training_config: TrainingConfig
):
    """Compare BitFit with full fine-tuning (simulation)"""
    
    logger.info("🔄 Simulating full fine-tuning comparison...")
    
    # Create standard model for comparison
    from transformers import AutoModelForSequenceClassification
    full_model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=model_config.num_labels
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in full_model.parameters())
    
    # Simulate training time and memory (rough estimates)
    full_training_time = len(train_dataset) * training_config.num_train_epochs * 0.001  # seconds
    full_memory_mb = total_params * 4 / (1024 * 1024)  # float32
    
    return {
        "full_finetuning": {
            "trainable_params": total_params,
            "training_time_estimate": full_training_time,
            "memory_mb": full_memory_mb,
            "parameter_efficiency": 100.0
        }
    }


def main():
    """Main BitFit training example"""
    
    parser = argparse.ArgumentParser(description="BitFit Text Classification Example")
    parser.add_argument("--dataset", default="dummy", choices=["imdb", "sst2", "dummy"], help="Dataset to use")
    parser.add_argument("--model_name", default="distilbert-base-uncased", help="Model name")
    parser.add_argument("--sample_size", type=int, default=500, help="Sample size for demonstration")
    parser.add_argument("--output_dir", default="./bitfit_results", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--bias_types", nargs="+", default=["all"], 
                       choices=["attention", "feedforward", "layer_norm", "classifier", "all"],
                       help="Bias types to train")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting BitFit text classification example...")
    
    # Load dataset
    train_dataset, test_dataset = load_sample_dataset(args.dataset, args.sample_size)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Setup configurations
    model_config = ModelConfig(
        model_name_or_path=args.model_name,
        num_labels=2,
        task_type="classification",
        max_length=128
    )
    
    # Configure bias types to train
    bias_config = {
        "train_attention_bias": "attention" in args.bias_types or "all" in args.bias_types,
        "train_feedforward_bias": "feedforward" in args.bias_types or "all" in args.bias_types,
        "train_layer_norm_bias": "layer_norm" in args.bias_types or "all" in args.bias_types,
        "train_classifier_bias": "classifier" in args.bias_types or "all" in args.bias_types,
    }
    
    bitfit_config = BitFitConfig(
        freeze_all_weights=True,
        train_bias_only=True,
        bias_learning_rate=args.learning_rate,
        bias_gradient_clipping=True,
        max_bias_grad_norm=1.0,
        track_bias_gradients=True,
        log_bias_statistics=True,
        **bias_config
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        seed=42
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    
    # Preprocess datasets
    logger.info("📊 Preprocessing datasets...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer, model_config.max_length)
    test_dataset = preprocess_dataset(test_dataset, tokenizer, model_config.max_length)
    
    # Create BitFit model
    logger.info("🔧 Creating BitFit model...")
    model = BitFitModel(model_config, bitfit_config)
    
    # Create trainer
    trainer = BitFitTrainer(model, training_config, tokenizer)
    
    # Compare with full fine-tuning
    comparison_baseline = compare_with_full_finetuning(
        model_config, train_dataset, test_dataset, tokenizer, training_config
    )
    
    # Train model
    logger.info("🏋️ Training BitFit model...")
    train_result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Evaluate model
    logger.info("📈 Evaluating model...")
    eval_result = trainer.evaluate()
    
    # Get training statistics
    training_stats = trainer.get_training_statistics()
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("📊 BITFIT TRAINING RESULTS")
    logger.info("="*60)
    
    # Model efficiency
    efficiency = training_stats["parameter_efficiency"]
    logger.info(f"Parameter Efficiency:")
    logger.info(f"  Total parameters: {efficiency.get('total', 0):,}")
    logger.info(f"  Trainable parameters: {efficiency.get('trainable', 0):,}")
    logger.info(f"  Parameter efficiency: {efficiency.get('efficiency', 0):.4f}%")
    logger.info(f"  Reduction factor: {efficiency.get('reduction_factor', 0):.1f}x")
    
    # Performance
    logger.info(f"\nPerformance:")
    logger.info(f"  Training loss: {train_result.training_loss:.4f}")
    logger.info(f"  Eval accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
    logger.info(f"  Eval F1: {eval_result.get('eval_f1', 0):.4f}")
    
    # Comparison with full fine-tuning
    comparison = training_stats["comparison"]
    logger.info(f"\nComparison with Full Fine-tuning:")
    logger.info(f"  Full FT trainable params: {comparison['full_finetuning']['trainable_params']:,}")
    logger.info(f"  BitFit trainable params: {comparison['bitfit']['trainable_params']:,}")
    logger.info(f"  Parameter reduction: {comparison['improvement']['parameter_reduction']:.1f}x")
    logger.info(f"  Memory reduction: {comparison['improvement']['memory_reduction']:.1f}x")
    
    # Bias statistics
    bias_stats = training_stats["bias_statistics"]
    logger.info(f"\nBias Parameter Statistics:")
    for bias_type, count in bias_stats["bias_parameter_counts"].items():
        if count > 0:
            norm = bias_stats["bias_parameter_norms"].get(bias_type, 0)
            logger.info(f"  {bias_type}: {count:,} params, norm: {norm:.4f}")
    
    logger.info("="*60)
    
    # Test inference
    logger.info("🧪 Testing inference...")
    test_texts = [
        "This movie is absolutely fantastic!",
        "Terrible film, complete waste of time.",
        "An okay movie, nothing special."
    ]
    
    model.base_model.eval()
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
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        logger.info(f"Text: '{text}'")
        logger.info(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
        logger.info("-" * 40)
    
    logger.info("✅ BitFit example completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    import torch
    main()
