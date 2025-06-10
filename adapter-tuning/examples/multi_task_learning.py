"""
Multi-Task Learning example using Adapter Tuning
"""

import os
import logging
import argparse
from datasets import load_dataset, concatenate_datasets

from config import ModelConfig, AdapterConfig, TrainingConfig
from data import load_dataset_from_hub, TextClassificationPreprocessor, MultiTaskPreprocessor
from training import AdapterTrainer
from evaluation import AdapterEvaluator
from inference import AdapterInferencePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_multi_task_dataset(tasks, train_size_per_task=500, eval_size_per_task=200):
    """Create a multi-task dataset from multiple single-task datasets"""
    
    multi_task_data = {"train": [], "validation": []}
    
    for task_name, task_config in tasks.items():
        logger.info(f"Loading dataset for task: {task_name}")
        
        # Load dataset
        if task_config["dataset"] == "glue":
            dataset = load_dataset("glue", task_config["subset"])
        else:
            dataset = load_dataset(task_config["dataset"])
        
        # Get train and validation splits
        train_split = dataset["train"].select(range(min(train_size_per_task, len(dataset["train"]))))
        
        if "validation" in dataset:
            val_split = dataset["validation"].select(range(min(eval_size_per_task, len(dataset["validation"]))))
        else:
            val_split = dataset["test"].select(range(min(eval_size_per_task, len(dataset["test"]))))
        
        # Add task information to each example
        def add_task_info(examples):
            examples["task_name"] = [task_name] * len(examples[task_config["text_column"]])
            examples["task_id"] = [list(tasks.keys()).index(task_name)] * len(examples[task_config["text_column"]])
            # Standardize column names
            examples["text"] = examples[task_config["text_column"]]
            examples["labels"] = examples[task_config["label_column"]]
            return examples
        
        train_split = train_split.map(add_task_info, batched=True)
        val_split = val_split.map(add_task_info, batched=True)
        
        multi_task_data["train"].append(train_split)
        multi_task_data["validation"].append(val_split)
    
    # Concatenate all datasets
    train_dataset = concatenate_datasets(multi_task_data["train"])
    val_dataset = concatenate_datasets(multi_task_data["validation"])
    
    # Shuffle the datasets
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)
    
    return {"train": train_dataset, "validation": val_dataset}


def main(args):
    """Main function for multi-task learning example"""
    
    # Define tasks for multi-task learning
    tasks = {
        "sentiment": {
            "dataset": "imdb",
            "text_column": "text",
            "label_column": "label",
            "num_labels": 2
        },
        "topic": {
            "dataset": "ag_news", 
            "text_column": "text",
            "label_column": "label",
            "num_labels": 4
        },
        "cola": {
            "dataset": "glue",
            "subset": "cola",
            "text_column": "sentence",
            "label_column": "label", 
            "num_labels": 2
        }
    }
    
    # Configuration
    model_config = ModelConfig(
        model_name_or_path=args.model,
        num_labels=max(task["num_labels"] for task in tasks.values()),  # Max labels across tasks
        max_length=256,
        task_type="classification"
    )
    
    # Multi-adapter configuration
    adapter_config = AdapterConfig(
        adapter_size=64,
        adapter_dropout=0.1,
        adapter_activation="relu",
        adapter_location="both",
        adapter_names=list(tasks.keys()),  # One adapter per task
        adapter_fusion=True,
        fusion_type="attention",
        freeze_base_model=True
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=1e-3,
        evaluation_strategy="steps",
        eval_steps=300,
        save_steps=300,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        multi_task_training=True,
        task_sampling_strategy="proportional",
        freeze_base_model=True,
        train_adapters_only=True
    )
    
    logger.info("Starting multi-task learning with adapter tuning")
    
    # Create multi-task dataset
    logger.info("Creating multi-task dataset...")
    multi_task_dataset = create_multi_task_dataset(
        tasks, 
        train_size_per_task=args.train_size_per_task,
        eval_size_per_task=args.eval_size_per_task
    )
    
    train_dataset = multi_task_dataset["train"]
    eval_dataset = multi_task_dataset["validation"]
    
    logger.info(f"Multi-task train dataset size: {len(train_dataset)}")
    logger.info(f"Multi-task eval dataset size: {len(eval_dataset)}")
    
    # Print task distribution
    task_counts = {}
    for example in train_dataset:
        task_name = example["task_name"]
        task_counts[task_name] = task_counts.get(task_name, 0) + 1
    
    logger.info("Task distribution in training set:")
    for task_name, count in task_counts.items():
        logger.info(f"  {task_name}: {count} examples")
    
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
    
    # Use standard text classification preprocessor
    preprocessor = TextClassificationPreprocessor(
        tokenizer=tokenizer,
        text_column="text",
        label_column="labels",  # Note: using "labels" from our standardized format
        max_length=model_config.max_length,
    )
    
    # Train model
    logger.info("Starting multi-task training...")
    train_result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocessor=preprocessor
    )
    
    logger.info(f"Multi-task training completed: {train_result}")
    
    # Evaluate model
    logger.info("Evaluating multi-task model...")
    eval_results = trainer.evaluate(eval_dataset)
    logger.info(f"Multi-task evaluation results: {eval_results}")
    
    # Test inference on each task
    logger.info("Testing multi-task inference...")
    
    # Load the trained model for inference
    inference_pipeline = AdapterInferencePipeline.from_pretrained(
        model_path=training_config.output_dir,
        adapter_path=os.path.join(training_config.output_dir, "adapters")
    )
    
    # Test examples for each task
    test_examples = {
        "sentiment": [
            "This movie is absolutely amazing!",
            "Worst film I've ever seen.",
            "The movie was okay, nothing special."
        ],
        "topic": [
            "Apple announces new iPhone with revolutionary features.",
            "Stock market reaches new highs amid economic recovery.",
            "Scientists discover new species in Amazon rainforest.",
            "Local team wins championship in thrilling final match."
        ],
        "cola": [
            "The cat sat on the mat.",
            "Cat the on sat mat the.",
            "She is very happy today."
        ]
    }
    
    for task_name, examples in test_examples.items():
        logger.info(f"\nTesting {task_name} task:")
        
        predictions = inference_pipeline.classify_text(examples)
        
        for example, pred in zip(examples, predictions):
            logger.info(f"  Text: {example}")
            logger.info(f"  Prediction: {pred}")
            logger.info("-" * 50)
    
    # Evaluate each task separately
    logger.info("\nEvaluating each task separately...")
    
    evaluator = AdapterEvaluator(
        adapter_model=adapter_model,
        task_type="classification"
    )
    
    task_results = {}
    
    for task_name in tasks.keys():
        # Filter evaluation dataset for this task
        task_eval_data = eval_dataset.filter(lambda x: x["task_name"] == task_name)
        
        if len(task_eval_data) > 0:
            logger.info(f"Evaluating {task_name} task ({len(task_eval_data)} examples)...")
            
            task_result = evaluator.evaluate_dataset(
                task_eval_data,
                preprocessor=preprocessor,
                save_predictions=True,
                output_dir=os.path.join(training_config.output_dir, "evaluation"),
                adapter_name=task_name
            )
            
            task_results[task_name] = task_result
            
            logger.info(f"{task_name} results: {task_result['metrics']}")
    
    # Compare task performance
    logger.info("\nTask Performance Comparison:")
    for task_name, results in task_results.items():
        metrics = results["metrics"]
        efficiency = results["efficiency_metrics"]
        
        logger.info(f"{task_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  F1: {metrics['f1']:.3f}")
        logger.info(f"  Examples/sec: {efficiency['examples_per_second']:.1f}")
    
    # Print adapter information
    logger.info("\nMulti-task adapter model information:")
    adapter_model.print_adapter_info()
    
    # Test adapter switching (if implemented)
    if hasattr(inference_pipeline, 'switch_adapter'):
        logger.info("\nTesting adapter switching...")
        for task_name in tasks.keys():
            try:
                inference_pipeline.switch_adapter(task_name)
                logger.info(f"Switched to {task_name} adapter")
            except Exception as e:
                logger.warning(f"Could not switch to {task_name} adapter: {e}")
    
    logger.info("Multi-task learning example completed!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-Task Learning with Adapter Tuning")
    
    # Model arguments
    parser.add_argument("--model", default="bert-base-uncased", help="Base model name")
    
    # Dataset arguments
    parser.add_argument("--train-size-per-task", type=int, default=500, help="Training examples per task")
    parser.add_argument("--eval-size-per-task", type=int, default=200, help="Eval examples per task")
    
    # Training arguments
    parser.add_argument("--output-dir", default="./results/multi_task", help="Output directory")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
