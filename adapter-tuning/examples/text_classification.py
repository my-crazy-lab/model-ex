"""
Text Classification example using Adapter Tuning
"""

import os
import logging
import argparse
from datasets import load_dataset

from config import ModelConfig, AdapterConfig, TrainingConfig
from config import BERT_BASE_CONFIG, CLASSIFICATION_ADAPTER_CONFIG, STANDARD_ADAPTER_TRAINING
from data import load_dataset_from_hub, TextClassificationPreprocessor
from training import AdapterTrainer
from evaluation import AdapterEvaluator
from inference import AdapterInferencePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Main function for text classification example"""
    
    # Configuration
    model_config = ModelConfig(
        model_name_or_path=args.model,
        num_labels=args.num_labels,
        max_length=args.max_length,
        task_type="classification"
    )
    
    adapter_config = AdapterConfig(
        adapter_size=args.adapter_size,
        adapter_dropout=0.1,
        adapter_activation="relu",
        adapter_location="both",
        freeze_base_model=True
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        freeze_base_model=True,
        train_adapters_only=True
    )
    
    logger.info("Starting text classification with adapter tuning")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "imdb":
        dataset = load_dataset_from_hub("imdb")
        text_column = "text"
        label_column = "label"
    elif args.dataset == "ag_news":
        dataset = load_dataset_from_hub("ag_news")
        text_column = "text"
        label_column = "label"
    elif args.dataset == "sst2":
        dataset = load_dataset_from_hub("glue", "sst2")
        text_column = "sentence"
        label_column = "label"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Take smaller subsets for quick testing if specified
    if args.train_size:
        dataset["train"] = dataset["train"].select(range(args.train_size))
    if args.eval_size and "validation" in dataset:
        dataset["validation"] = dataset["validation"].select(range(args.eval_size))
    elif args.eval_size and "test" in dataset:
        dataset["test"] = dataset["test"].select(range(args.eval_size))
    
    # Use test set as validation if no validation set
    if "validation" not in dataset and "test" in dataset:
        dataset["validation"] = dataset["test"]
    
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation", dataset.get("test"))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset) if eval_dataset else 0}")
    
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
        text_column=text_column,
        label_column=label_column,
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
    if eval_dataset:
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate(eval_dataset)
        logger.info(f"Evaluation results: {eval_results}")
    
    # Test inference
    logger.info("Testing inference...")
    
    # Load the trained model for inference
    inference_pipeline = AdapterInferencePipeline.from_pretrained(
        model_path=training_config.output_dir,
        adapter_path=os.path.join(training_config.output_dir, "adapters")
    )
    
    # Test samples based on dataset
    if args.dataset == "imdb":
        test_texts = [
            "This movie is absolutely fantastic! I loved every minute of it.",
            "Terrible movie, waste of time. Very disappointing.",
            "The plot was okay but the acting was great.",
            "Not the best movie I've seen, but not the worst either."
        ]
    elif args.dataset == "ag_news":
        test_texts = [
            "Apple announces new iPhone with revolutionary features.",
            "Stock market reaches new highs amid economic recovery.",
            "Scientists discover new species in Amazon rainforest.",
            "Local team wins championship in thrilling final match."
        ]
    else:
        test_texts = [
            "This is a positive example.",
            "This is a negative example.",
            "This is a neutral example.",
            "This could be anything."
        ]
    
    # Get predictions
    predictions = inference_pipeline.classify_text(
        test_texts,
        return_all_scores=True
    )
    
    logger.info("Inference results:")
    for text, pred in zip(test_texts, predictions):
        logger.info(f"Text: {text}")
        logger.info(f"Prediction: {pred}")
        logger.info("-" * 50)
    
    # Detailed evaluation with AdapterEvaluator
    if eval_dataset:
        logger.info("Performing detailed evaluation...")
        evaluator = AdapterEvaluator(
            adapter_model=adapter_model,
            task_type="classification"
        )
        
        detailed_results = evaluator.evaluate_dataset(
            eval_dataset.select(range(min(100, len(eval_dataset)))),  # Small subset for demo
            preprocessor=preprocessor,
            save_predictions=True,
            output_dir=os.path.join(training_config.output_dir, "evaluation")
        )
        
        logger.info(f"Detailed evaluation: {detailed_results['metrics']}")
        logger.info(f"Efficiency metrics: {detailed_results['efficiency_metrics']}")
    
    # Print adapter information
    logger.info("Adapter model information:")
    adapter_model.print_adapter_info()
    
    logger.info("Text classification example completed!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Text Classification with Adapter Tuning")
    
    # Model arguments
    parser.add_argument("--model", default="bert-base-uncased", help="Base model name")
    parser.add_argument("--adapter-size", type=int, default=64, help="Adapter bottleneck size")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    # Dataset arguments
    parser.add_argument("--dataset", default="imdb", choices=["imdb", "ag_news", "sst2"], help="Dataset name")
    parser.add_argument("--num-labels", type=int, default=2, help="Number of labels")
    parser.add_argument("--train-size", type=int, help="Limit training set size")
    parser.add_argument("--eval-size", type=int, help="Limit eval set size")
    
    # Training arguments
    parser.add_argument("--output-dir", default="./results/text_classification", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save steps")
    parser.add_argument("--logging-steps", type=int, default=100, help="Logging steps")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
