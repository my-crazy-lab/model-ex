"""
Complete lifelong learning text classification example
"""

import argparse
import logging
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import json
import os

from config import LifelongConfig, TaskConfig
from models import LifelongModel
from training import LifelongTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_task_sequence(task_names: list, sample_size: int = 1000):
    """Create sequence of tasks for continual learning"""
    
    task_datasets = {}
    task_configs = {}
    
    for task_id, task_name in enumerate(task_names):
        logger.info(f"Loading task {task_id}: {task_name}")
        
        if task_name == "sentiment_imdb":
            dataset = load_dataset("imdb")
            train_data = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
            test_data = dataset["test"].select(range(min(sample_size // 4, len(dataset["test"]))))
            
            task_config = TaskConfig(
                task_id=task_id,
                task_name=task_name,
                num_labels=2,
                label_names=["negative", "positive"],
                text_column="text",
                label_column="label"
            )
            
        elif task_name == "sentiment_sst2":
            dataset = load_dataset("glue", "sst2")
            train_data = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
            test_data = dataset["validation"].select(range(min(sample_size // 4, len(dataset["validation"]))))
            
            task_config = TaskConfig(
                task_id=task_id,
                task_name=task_name,
                num_labels=2,
                label_names=["negative", "positive"],
                text_column="sentence",
                label_column="label"
            )
            
        elif task_name == "topic_ag_news":
            dataset = load_dataset("ag_news")
            train_data = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
            test_data = dataset["test"].select(range(min(sample_size // 4, len(dataset["test"]))))
            
            task_config = TaskConfig(
                task_id=task_id,
                task_name=task_name,
                num_labels=4,
                label_names=["World", "Sports", "Business", "Technology"],
                text_column="text",
                label_column="label"
            )
            
        else:
            # Create dummy task
            texts = [
                f"This is a sample text for task {task_name}.",
                f"Another example for {task_name} classification.",
                f"Task {task_name} requires different knowledge."
            ] * (sample_size // 3)
            
            labels = [0, 1, 0] * (sample_size // 3)
            
            train_size = int(0.8 * len(texts))
            train_data = Dataset.from_dict({
                "text": texts[:train_size],
                "label": labels[:train_size]
            })
            test_data = Dataset.from_dict({
                "text": texts[train_size:],
                "label": labels[train_size:]
            })
            
            task_config = TaskConfig(
                task_id=task_id,
                task_name=task_name,
                num_labels=2,
                label_names=["class_0", "class_1"],
                text_column="text",
                label_column="label"
            )
        
        task_datasets[task_id] = {
            "train": train_data,
            "test": test_data
        }
        task_configs[task_id] = task_config
        
        logger.info(f"Task {task_id} ({task_name}): {len(train_data)} train, {len(test_data)} test")
    
    return task_datasets, task_configs


def preprocess_task_data(dataset, tokenizer, task_config, max_length=128):
    """Preprocess data for specific task"""
    
    def preprocess_function(examples):
        # Get text column
        texts = examples[task_config.text_column]
        
        # Tokenize
        result = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        
        # Add labels
        result["labels"] = examples[task_config.label_column]
        
        return result
    
    return dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Preprocessing {task_config.task_name}"
    )


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {"accuracy": accuracy}


def evaluate_forgetting(
    model: LifelongModel,
    task_dataloaders: dict,
    baseline_performance: dict,
    device: torch.device
):
    """Evaluate catastrophic forgetting"""
    
    logger.info("🔍 Evaluating catastrophic forgetting...")
    
    current_performance = {}
    forgetting_scores = {}
    
    model.eval()
    
    for task_id, dataloader in task_dataloaders.items():
        if task_id in model.completed_tasks:
            # Evaluate current performance
            task_results = model.get_task_performance(task_id, dataloader, device)
            current_acc = task_results['accuracy']
            current_performance[task_id] = current_acc
            
            # Compute forgetting
            if task_id in baseline_performance:
                baseline_acc = baseline_performance[task_id]
                forgetting = max(0, baseline_acc - current_acc)
                forgetting_scores[task_id] = forgetting
                
                logger.info(f"Task {task_id}: {baseline_acc:.3f} → {current_acc:.3f} (forgetting: {forgetting:.3f})")
    
    # Compute average forgetting
    if forgetting_scores:
        avg_forgetting = sum(forgetting_scores.values()) / len(forgetting_scores)
        logger.info(f"Average forgetting: {avg_forgetting:.3f}")
    else:
        avg_forgetting = 0.0
    
    return {
        "current_performance": current_performance,
        "forgetting_scores": forgetting_scores,
        "average_forgetting": avg_forgetting
    }


def main():
    """Main lifelong learning example"""
    
    parser = argparse.ArgumentParser(description="Lifelong Learning Text Classification")
    parser.add_argument("--technique", default="ewc", 
                       choices=["ewc", "rehearsal", "l2_reg", "combined"],
                       help="Lifelong learning technique")
    parser.add_argument("--model_name", default="distilbert-base-uncased", 
                       help="Base model name")
    parser.add_argument("--tasks", nargs="+", 
                       default=["sentiment_imdb", "sentiment_sst2", "topic_ag_news"],
                       help="Task sequence")
    parser.add_argument("--sample_size", type=int, default=500, 
                       help="Sample size per task")
    parser.add_argument("--output_dir", default="./lifelong_results", 
                       help="Output directory")
    parser.add_argument("--epochs_per_task", type=int, default=3, 
                       help="Training epochs per task")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--ewc_lambda", type=float, default=1000.0, 
                       help="EWC regularization strength")
    parser.add_argument("--memory_size", type=int, default=500, 
                       help="Memory buffer size for rehearsal")
    
    args = parser.parse_args()
    
    logger.info("🧠 Starting Lifelong Learning Example...")
    logger.info(f"Technique: {args.technique}")
    logger.info(f"Tasks: {args.tasks}")
    
    # Create task sequence
    task_datasets, task_configs = create_task_sequence(args.tasks, args.sample_size)
    
    # Setup lifelong learning configuration
    if args.technique == "ewc":
        lifelong_config = LifelongConfig(
            technique="ewc",
            ewc_lambda=args.ewc_lambda,
            fisher_estimation_samples=200,
            learning_rate=args.learning_rate,
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size
        )
    elif args.technique == "rehearsal":
        lifelong_config = LifelongConfig(
            technique="rehearsal",
            memory_size=args.memory_size,
            memory_strategy="balanced",
            replay_batch_size=args.batch_size // 2,
            learning_rate=args.learning_rate,
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size
        )
    elif args.technique == "combined":
        lifelong_config = LifelongConfig(
            technique="combined",
            combine_techniques=True,
            combined_techniques=["ewc", "rehearsal"],
            ewc_lambda=args.ewc_lambda / 2,  # Reduced for combination
            memory_size=args.memory_size // 2,  # Reduced for combination
            learning_rate=args.learning_rate,
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size
        )
    else:  # l2_reg
        lifelong_config = LifelongConfig(
            technique="l2_reg",
            l2_lambda=0.01,
            learning_rate=args.learning_rate,
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create lifelong model
    num_labels_per_task = {task_id: config.num_labels 
                          for task_id, config in task_configs.items()}
    
    model = LifelongModel(
        base_model_name=args.model_name,
        lifelong_config=lifelong_config,
        num_labels_per_task=num_labels_per_task
    )
    
    # Create trainer
    trainer = LifelongTrainer(model, lifelong_config, tokenizer)
    
    # Track performance
    baseline_performance = {}  # Best performance on each task
    all_results = {}
    
    # Sequential task learning
    logger.info("🚀 Starting sequential task learning...")
    
    for task_id, task_config in task_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 Learning Task {task_id}: {task_config.task_name}")
        logger.info(f"{'='*60}")
        
        # Preprocess data
        train_dataset = preprocess_task_data(
            task_datasets[task_id]["train"], 
            tokenizer, 
            task_config
        )
        test_dataset = preprocess_task_data(
            task_datasets[task_id]["test"], 
            tokenizer, 
            task_config
        )
        
        # Add task to model
        model.add_task(task_id, task_config.num_labels)
        
        # Train on current task
        train_result = trainer.learn_task(
            task_id=task_id,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        # Evaluate on current task
        current_task_results = trainer.evaluate_task(task_id, test_dataset)
        baseline_performance[task_id] = current_task_results['eval_accuracy']
        
        logger.info(f"Task {task_id} performance: {current_task_results['eval_accuracy']:.3f}")
        
        # Complete task (for EWC Fisher computation)
        from torch.utils.data import DataLoader
        eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        model.complete_task(task_id, eval_dataloader)
        
        # Evaluate on all previous tasks
        if len(model.completed_tasks) > 1:
            logger.info("📊 Evaluating on all previous tasks...")
            
            task_dataloaders = {}
            for prev_task_id in model.completed_tasks:
                prev_test_dataset = preprocess_task_data(
                    task_datasets[prev_task_id]["test"],
                    tokenizer,
                    task_configs[prev_task_id]
                )
                task_dataloaders[prev_task_id] = DataLoader(
                    prev_test_dataset, 
                    batch_size=args.batch_size
                )
            
            # Evaluate forgetting
            forgetting_results = evaluate_forgetting(
                model, 
                task_dataloaders, 
                baseline_performance,
                trainer.device
            )
            
            all_results[f"after_task_{task_id}"] = {
                "current_performance": forgetting_results["current_performance"],
                "forgetting_scores": forgetting_results["forgetting_scores"],
                "average_forgetting": forgetting_results["average_forgetting"]
            }
    
    # Final evaluation
    logger.info(f"\n{'='*60}")
    logger.info("🎯 FINAL EVALUATION")
    logger.info(f"{'='*60}")
    
    # Create final dataloaders
    final_dataloaders = {}
    for task_id in task_configs.keys():
        test_dataset = preprocess_task_data(
            task_datasets[task_id]["test"],
            tokenizer,
            task_configs[task_id]
        )
        final_dataloaders[task_id] = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Final performance evaluation
    final_results = model.evaluate_all_tasks(final_dataloaders, trainer.device)
    
    # Print results summary
    logger.info("\n📊 LIFELONG LEARNING RESULTS:")
    logger.info("=" * 50)
    
    for task_id, config in task_configs.items():
        baseline_acc = baseline_performance.get(task_id, 0.0)
        final_acc = final_results.get(f'task_{task_id}', {}).get('accuracy', 0.0)
        forgetting = max(0, baseline_acc - final_acc)
        
        logger.info(f"Task {task_id} ({config.task_name}):")
        logger.info(f"  Baseline: {baseline_acc:.3f}")
        logger.info(f"  Final: {final_acc:.3f}")
        logger.info(f"  Forgetting: {forgetting:.3f}")
    
    if 'average' in final_results:
        logger.info(f"\nAverage Performance: {final_results['average']['accuracy']:.3f}")
    
    # Compute overall forgetting
    total_forgetting = sum(
        max(0, baseline_performance[tid] - final_results.get(f'task_{tid}', {}).get('accuracy', 0))
        for tid in baseline_performance.keys()
    ) / len(baseline_performance)
    
    logger.info(f"Average Forgetting: {total_forgetting:.3f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_summary = {
        "technique": args.technique,
        "tasks": args.tasks,
        "baseline_performance": baseline_performance,
        "final_results": final_results,
        "average_forgetting": total_forgetting,
        "all_results": all_results,
        "config": lifelong_config.to_dict()
    }
    
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save model
    model.save_model(args.output_dir)
    
    logger.info(f"\n✅ Lifelong learning completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Technique: {args.technique}")
    logger.info(f"Average final performance: {final_results.get('average', {}).get('accuracy', 0):.3f}")
    logger.info(f"Average forgetting: {total_forgetting:.3f}")


if __name__ == "__main__":
    main()
