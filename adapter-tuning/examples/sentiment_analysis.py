"""
Sentiment Analysis example using Adapter Tuning
"""

import os
import logging
import argparse
from datasets import load_dataset

from config import ModelConfig, AdapterConfig, TrainingConfig
from data import load_dataset_from_hub, TextClassificationPreprocessor
from training import AdapterTrainer
from evaluation import AdapterEvaluator
from inference import AdapterInferencePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Main function for sentiment analysis example"""
    
    # Configuration for sentiment analysis
    model_config = ModelConfig(
        model_name_or_path=args.model,
        num_labels=2,  # Positive/Negative
        max_length=256,  # Shorter for sentiment
        task_type="classification"
    )
    
    adapter_config = AdapterConfig(
        adapter_size=32,  # Smaller adapter for sentiment
        adapter_dropout=0.1,
        adapter_activation="relu",
        adapter_location="feedforward",  # Only in feedforward layers
        freeze_base_model=True
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=2,  # Fewer epochs for sentiment
        per_device_train_batch_size=32,
        learning_rate=2e-3,  # Higher learning rate
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        freeze_base_model=True,
        train_adapters_only=True
    )
    
    logger.info("Starting sentiment analysis with adapter tuning")
    
    # Load IMDB dataset for sentiment analysis
    logger.info("Loading IMDB dataset for sentiment analysis...")
    dataset = load_dataset_from_hub("imdb")
    
    # Take smaller subsets for quick demo
    train_dataset = dataset["train"].select(range(args.train_size))
    eval_dataset = dataset["test"].select(range(args.eval_size))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Setup trainer
    trainer = AdapterTrainer(
        model_config=model_config,
        adapter_config=adapter_config,
        training_config=training_config,
        task_type="classification"
    )
    
    # Setup model and preprocessor
    adapter_model = trainer.setup_model()
    tokenizer = adapter_model.tokenizer if hasattr(adapter_model, 'tokenizer') else trainer.tokenizer
    
    preprocessor = TextClassificationPreprocessor(
        tokenizer=tokenizer,
        text_column="text",
        label_column="label",
        max_length=model_config.max_length,
    )
    
    # Train model
    logger.info("Starting training...")
    train_result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocessor=preprocessor
    )
    
    logger.info(f"Training completed: {train_result}")
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate(eval_dataset)
    logger.info(f"Evaluation results: {eval_results}")
    
    # Test inference with movie reviews
    logger.info("Testing sentiment analysis inference...")
    
    # Load the trained model for inference
    inference_pipeline = AdapterInferencePipeline.from_pretrained(
        model_path=training_config.output_dir,
        adapter_path=os.path.join(training_config.output_dir, "adapters")
    )
    
    # Test movie reviews
    movie_reviews = [
        "This movie is absolutely brilliant! The acting is superb and the plot is engaging.",
        "Worst movie I've ever seen. Complete waste of time and money.",
        "The movie was okay, nothing special but not terrible either.",
        "Amazing cinematography and outstanding performances by all actors.",
        "Boring and predictable. I fell asleep halfway through.",
        "A masterpiece of modern cinema. Highly recommended!",
        "The plot was confusing and the ending made no sense.",
        "Great movie with excellent character development and storyline."
    ]
    
    # Get sentiment predictions
    predictions = inference_pipeline.classify_text(
        movie_reviews,
        return_all_scores=True
    )
    
    logger.info("Sentiment Analysis Results:")
    for review, pred in zip(movie_reviews, predictions):
        sentiment = "Positive" if pred[0]["label"] == "LABEL_1" else "Negative"
        confidence = pred[0]["score"]
        
        logger.info(f"Review: {review}")
        logger.info(f"Sentiment: {sentiment} (confidence: {confidence:.3f})")
        logger.info("-" * 80)
    
    # Test with custom sentences
    logger.info("\nTesting with custom sentences...")
    
    custom_sentences = [
        "I love this!",
        "This is terrible.",
        "Not bad at all.",
        "Absolutely fantastic!",
        "Could be better.",
        "Perfect!",
        "Disappointing.",
        "Excellent work!"
    ]
    
    custom_predictions = inference_pipeline.classify_text(custom_sentences)
    
    for sentence, pred in zip(custom_sentences, custom_predictions):
        sentiment = "Positive" if pred["label"] == "LABEL_1" else "Negative"
        confidence = pred["score"]
        
        logger.info(f"'{sentence}' → {sentiment} ({confidence:.3f})")
    
    # Detailed evaluation
    logger.info("\nPerforming detailed evaluation...")
    evaluator = AdapterEvaluator(
        adapter_model=adapter_model,
        task_type="classification"
    )
    
    detailed_results = evaluator.evaluate_dataset(
        eval_dataset.select(range(200)),  # Small subset for demo
        preprocessor=preprocessor,
        save_predictions=True,
        output_dir=os.path.join(training_config.output_dir, "evaluation")
    )
    
    logger.info(f"Detailed metrics: {detailed_results['metrics']}")
    logger.info(f"Efficiency metrics: {detailed_results['efficiency_metrics']}")
    
    # Print adapter information
    logger.info("\nAdapter model information:")
    adapter_model.print_adapter_info()
    
    # Compare with different adapter sizes
    if args.compare_adapters:
        logger.info("\nComparing different adapter sizes...")
        
        # Train with different adapter sizes
        adapter_sizes = [16, 32, 64, 128]
        adapter_results = {}
        
        for size in adapter_sizes:
            logger.info(f"Training adapter with size {size}...")
            
            # Create new config with different adapter size
            size_adapter_config = AdapterConfig(
                adapter_size=size,
                adapter_dropout=0.1,
                adapter_activation="relu",
                adapter_location="feedforward",
                freeze_base_model=True
            )
            
            size_training_config = TrainingConfig(
                output_dir=f"{args.output_dir}_size_{size}",
                num_train_epochs=1,  # Quick training
                per_device_train_batch_size=32,
                learning_rate=2e-3,
                evaluation_strategy="epoch",
                logging_steps=50,
                freeze_base_model=True,
                train_adapters_only=True
            )
            
            # Train adapter
            size_trainer = AdapterTrainer(
                model_config=model_config,
                adapter_config=size_adapter_config,
                training_config=size_training_config,
                task_type="classification"
            )
            
            size_trainer.train(
                train_dataset=train_dataset.select(range(500)),  # Small subset
                eval_dataset=eval_dataset.select(range(100)),
                preprocessor=preprocessor
            )
            
            # Evaluate
            size_evaluator = AdapterEvaluator(
                adapter_model=size_trainer.adapter_model,
                task_type="classification"
            )
            
            size_results = size_evaluator.evaluate_dataset(
                eval_dataset.select(range(100)),
                preprocessor=preprocessor
            )
            
            adapter_results[f"adapter_size_{size}"] = size_results
        
        # Compare results
        logger.info("\nAdapter Size Comparison:")
        for adapter_name, results in adapter_results.items():
            metrics = results["metrics"]
            efficiency = results["efficiency_metrics"]
            
            logger.info(f"{adapter_name}:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"  F1: {metrics['f1']:.3f}")
            logger.info(f"  Adapter params: {efficiency['adapter_params']:,}")
            logger.info(f"  Parameter efficiency: {efficiency['parameter_efficiency']:.2f}%")
    
    logger.info("Sentiment analysis example completed!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Sentiment Analysis with Adapter Tuning")
    
    # Model arguments
    parser.add_argument("--model", default="distilbert-base-uncased", help="Base model name")
    
    # Dataset arguments
    parser.add_argument("--train-size", type=int, default=1000, help="Training set size")
    parser.add_argument("--eval-size", type=int, default=500, help="Eval set size")
    
    # Training arguments
    parser.add_argument("--output-dir", default="./results/sentiment_analysis", help="Output directory")
    
    # Comparison
    parser.add_argument("--compare-adapters", action="store_true", help="Compare different adapter sizes")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
