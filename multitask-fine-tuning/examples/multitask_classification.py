"""
Complete multitask classification example
"""

import argparse
import logging
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
import os

from config import MultitaskConfig, TaskConfig
from multitask import MultitaskModel
from training import MultitaskTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_classification_datasets(sample_size: int = 1000):
    """Load multiple classification datasets"""
    
    datasets = {}
    
    # Sentiment Analysis (IMDB)
    logger.info("Loading IMDB dataset...")
    imdb = load_dataset("imdb")
    datasets["sentiment"] = {
        "train": imdb["train"].select(range(min(sample_size, len(imdb["train"])))),
        "test": imdb["test"].select(range(min(sample_size // 4, len(imdb["test"])))),
        "num_labels": 2,
        "label_names": ["negative", "positive"]
    }
    
    # Topic Classification (AG News)
    logger.info("Loading AG News dataset...")
    ag_news = load_dataset("ag_news")
    datasets["topic"] = {
        "train": ag_news["train"].select(range(min(sample_size, len(ag_news["train"])))),
        "test": ag_news["test"].select(range(min(sample_size // 4, len(ag_news["test"])))),
        "num_labels": 4,
        "label_names": ["World", "Sports", "Business", "Technology"]
    }
    
    # Natural Language Inference (SNLI) - simplified
    logger.info("Loading SNLI dataset...")
    try:
        snli = load_dataset("snli")
        # Filter out examples with label -1 (no consensus)
        train_snli = snli["train"].filter(lambda x: x["label"] != -1)
        test_snli = snli["test"].filter(lambda x: x["label"] != -1)
        
        datasets["nli"] = {
            "train": train_snli.select(range(min(sample_size, len(train_snli)))),
            "test": test_snli.select(range(min(sample_size // 4, len(test_snli)))),
            "num_labels": 3,
            "label_names": ["entailment", "neutral", "contradiction"]
        }
    except:
        # Create dummy NLI dataset if SNLI not available
        logger.warning("SNLI not available, creating dummy NLI dataset")
        
        premises = ["The cat is sleeping on the mat."] * (sample_size // 3)
        hypotheses = ["The cat is resting.", "The dog is barking.", "The cat is flying."] * (sample_size // 9)
        labels = [0, 1, 2] * (sample_size // 9)  # entailment, neutral, contradiction
        
        train_size = int(0.8 * len(premises))
        
        datasets["nli"] = {
            "train": Dataset.from_dict({
                "premise": premises[:train_size],
                "hypothesis": hypotheses[:train_size],
                "label": labels[:train_size]
            }),
            "test": Dataset.from_dict({
                "premise": premises[train_size:],
                "hypothesis": hypotheses[train_size:],
                "label": labels[train_size:]
            }),
            "num_labels": 3,
            "label_names": ["entailment", "neutral", "contradiction"]
        }
    
    return datasets


def preprocess_datasets(datasets, tokenizer, max_length=128):
    """Preprocess all datasets"""
    
    processed_datasets = {}
    
    for task_name, task_data in datasets.items():
        logger.info(f"Preprocessing {task_name} dataset...")
        
        def preprocess_function(examples):
            if task_name == "sentiment":
                # Single text classification
                texts = examples["text"]
                result = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None
                )
            
            elif task_name == "topic":
                # Single text classification
                texts = examples["text"]
                result = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None
                )
            
            elif task_name == "nli":
                # Pair text classification
                premises = examples["premise"]
                hypotheses = examples["hypothesis"]
                result = tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None
                )
            
            # Add labels and task information
            result["labels"] = examples["label"]
            result["task_name"] = [task_name] * len(examples["label"])
            
            return result
        
        # Process train and test sets
        processed_datasets[task_name] = {
            "train": task_data["train"].map(preprocess_function, batched=True),
            "test": task_data["test"].map(preprocess_function, batched=True),
            "num_labels": task_data["num_labels"],
            "label_names": task_data["label_names"]
        }
    
    return processed_datasets


def compute_classification_metrics(eval_pred):
    """Compute classification metrics"""
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


def analyze_task_interference(
    model: MultitaskModel,
    task_dataloaders: dict,
    device: torch.device,
    baseline_performance: dict = None
):
    """Analyze task interference and transfer"""
    
    logger.info("🔍 Analyzing task interference...")
    
    # Evaluate current multitask performance
    current_results = model.evaluate_all_tasks(
        task_dataloaders, 
        device,
        compute_metrics_fns={task: compute_classification_metrics for task in task_dataloaders}
    )
    
    # Compute interference metrics
    interference_results = model.compute_task_interference(
        task_dataloaders,
        device,
        baseline_performance
    )
    
    # Print analysis
    logger.info("\n📊 TASK INTERFERENCE ANALYSIS:")
    logger.info("=" * 50)
    
    for task_name, results in current_results.items():
        if task_name != "average":
            accuracy = results.get("accuracy", 0)
            logger.info(f"{task_name.capitalize()}: {accuracy:.4f} accuracy")
    
    if "average" in current_results:
        avg_acc = current_results["average"]["accuracy"]
        logger.info(f"Average: {avg_acc:.4f} accuracy")
    
    if baseline_performance and "positive_transfer" in interference_results:
        logger.info(f"\nTransfer Analysis:")
        logger.info(f"  Positive transfer: {interference_results['positive_transfer']:.4f}")
        logger.info(f"  Negative transfer: {interference_results['negative_transfer']:.4f}")
        logger.info(f"  Net transfer: {interference_results['net_transfer']:.4f}")
        logger.info(f"  Transfer ratio: {interference_results['transfer_ratio']:.2f}")
    
    logger.info("=" * 50)
    
    return current_results, interference_results


def test_zero_shot_transfer(
    model: MultitaskModel,
    test_datasets: dict,
    tokenizer,
    device: torch.device
):
    """Test zero-shot transfer to new tasks"""
    
    logger.info("🧪 Testing zero-shot transfer...")
    
    # Create a simple new task (dummy)
    new_task_texts = [
        "This is a positive example for the new task.",
        "This is a negative example for the new task.",
        "Another positive case here.",
        "Another negative case here."
    ]
    new_task_labels = [1, 0, 1, 0]
    
    new_task_dataset = Dataset.from_dict({
        "text": new_task_texts,
        "label": new_task_labels
    })
    
    # Preprocess
    def preprocess_new_task(examples):
        result = tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors=None
        )
        result["labels"] = examples["label"]
        return result
    
    new_task_dataset = new_task_dataset.map(preprocess_new_task, batched=True)
    
    # Test with existing task heads (sentiment should work best)
    model.eval()
    
    results = {}
    for task_name in model.config.get_task_names():
        if task_name == "sentiment":  # Most similar to our dummy task
            from torch.utils.data import DataLoader
            dataloader = DataLoader(new_task_dataset, batch_size=4)
            
            task_results = model.evaluate_task(
                task_name=task_name,
                dataloader=dataloader,
                device=device,
                compute_metrics_fn=compute_classification_metrics
            )
            
            results[task_name] = task_results
    
    logger.info("Zero-shot transfer results:")
    for task_name, task_results in results.items():
        accuracy = task_results.get("accuracy", 0)
        logger.info(f"  {task_name} head: {accuracy:.4f} accuracy")
    
    return results


def main():
    """Main multitask classification example"""
    
    parser = argparse.ArgumentParser(description="Multitask Classification Example")
    parser.add_argument("--model_name", default="bert-base-uncased", 
                       help="Base model name")
    parser.add_argument("--sample_size", type=int, default=500, 
                       help="Sample size per task")
    parser.add_argument("--output_dir", default="./multitask_results", 
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                       help="Learning rate")
    parser.add_argument("--task_sampling", default="proportional", 
                       choices=["proportional", "equal", "temperature"],
                       help="Task sampling strategy")
    parser.add_argument("--loss_weighting", default="equal", 
                       choices=["equal", "proportional", "uncertainty"],
                       help="Loss weighting strategy")
    parser.add_argument("--use_task_embeddings", action="store_true", 
                       help="Use task embeddings")
    parser.add_argument("--analyze_interference", action="store_true", 
                       help="Analyze task interference")
    
    args = parser.parse_args()
    
    logger.info("🎯 Starting Multitask Classification example...")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Task sampling: {args.task_sampling}")
    logger.info(f"Loss weighting: {args.loss_weighting}")
    
    # Load datasets
    datasets = load_classification_datasets(args.sample_size)
    
    logger.info("Dataset sizes:")
    for task_name, task_data in datasets.items():
        train_size = len(task_data["train"])
        test_size = len(task_data["test"])
        num_labels = task_data["num_labels"]
        logger.info(f"  {task_name}: {train_size} train, {test_size} test, {num_labels} labels")
    
    # Setup task configurations
    task_configs = {}
    for task_name, task_data in datasets.items():
        task_configs[task_name] = TaskConfig(
            task_type="classification",
            num_labels=task_data["num_labels"],
            label_names=task_data["label_names"],
            dataset_name=task_name
        )
    
    # Setup multitask configuration
    multitask_config = MultitaskConfig(
        model_name_or_path=args.model_name,
        tasks=task_configs,
        task_sampling_strategy=args.task_sampling,
        loss_weighting_strategy=args.loss_weighting,
        use_task_embeddings=args.use_task_embeddings,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        output_dir=args.output_dir,
        eval_steps=100,
        logging_steps=50
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Preprocess datasets
    processed_datasets = preprocess_datasets(datasets, tokenizer)
    
    # Create multitask model
    logger.info("🔧 Creating multitask model...")
    model = MultitaskModel(multitask_config, tokenizer)
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_dataloaders = {}
    test_dataloaders = {}
    
    for task_name, task_data in processed_datasets.items():
        train_dataloaders[task_name] = DataLoader(
            task_data["train"], 
            batch_size=args.batch_size, 
            shuffle=True
        )
        test_dataloaders[task_name] = DataLoader(
            task_data["test"], 
            batch_size=args.batch_size * 2, 
            shuffle=False
        )
    
    # Create trainer
    from training import MultitaskTrainer
    
    trainer = MultitaskTrainer(model, multitask_config, tokenizer)
    
    # Train model
    logger.info("🏋️ Training multitask model...")
    training_stats = trainer.train(
        train_dataloaders=train_dataloaders,
        eval_dataloaders=test_dataloaders,
        num_epochs=args.epochs,
        compute_metrics_fn=compute_classification_metrics
    )
    
    # Final evaluation
    logger.info("📈 Final evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    final_results = model.evaluate_all_tasks(
        test_dataloaders,
        device,
        compute_metrics_fns={task: compute_classification_metrics for task in test_dataloaders}
    )
    
    # Analyze task interference if requested
    interference_results = None
    if args.analyze_interference:
        _, interference_results = analyze_task_interference(
            model, test_dataloaders, device
        )
    
    # Test zero-shot transfer
    zero_shot_results = test_zero_shot_transfer(
        model, processed_datasets, tokenizer, device
    )
    
    # Print comprehensive results
    logger.info("\n" + "="*60)
    logger.info("🎯 MULTITASK CLASSIFICATION RESULTS")
    logger.info("="*60)
    
    # Model statistics
    model_stats = model.get_model_statistics()
    logger.info(f"Model Statistics:")
    logger.info(f"  Total parameters: {model_stats['total_parameters']:,}")
    logger.info(f"  Backbone parameters: {model_stats['backbone_parameters']:,}")
    logger.info(f"  Task head parameters: {model_stats['total_head_parameters']:,}")
    logger.info(f"  Number of tasks: {model_stats['num_tasks']}")
    
    # Task performance
    logger.info(f"\nTask Performance:")
    for task_name, results in final_results.items():
        if task_name != "average":
            accuracy = results.get("accuracy", 0)
            f1 = results.get("f1", 0)
            logger.info(f"  {task_name.capitalize()}: {accuracy:.4f} accuracy, {f1:.4f} F1")
    
    if "average" in final_results:
        avg_acc = final_results["average"]["accuracy"]
        logger.info(f"  Average: {avg_acc:.4f} accuracy")
    
    # Training statistics
    logger.info(f"\nTraining Statistics:")
    logger.info(f"  Total steps: {training_stats.get('total_steps', 0)}")
    logger.info(f"  Average loss: {training_stats.get('avg_loss', 0):.4f}")
    
    logger.info("="*60)
    
    # Test inference on sample texts
    logger.info("🧪 Testing inference on sample texts...")
    
    test_examples = {
        "sentiment": "This movie is absolutely fantastic and amazing!",
        "topic": "The stock market reached new highs today as investors celebrated.",
        "nli": ("The cat is sleeping.", "The cat is resting.")
    }
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for task_name, example in test_examples.items():
        if task_name in model.config.tasks:
            if task_name == "nli":
                premise, hypothesis = example
                inputs = tokenizer(
                    premise, hypothesis,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(device)
            else:
                inputs = tokenizer(
                    example,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(device)
            
            with torch.no_grad():
                outputs = model(task_name=task_name, **inputs)
                predictions = torch.softmax(outputs["logits"], dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            label_names = datasets[task_name]["label_names"]
            predicted_label = label_names[predicted_class]
            
            logger.info(f"{task_name.capitalize()}: '{example}' → {predicted_label} ({confidence:.3f})")
    
    # Save results
    results_summary = {
        "model_config": multitask_config.to_dict(),
        "model_statistics": model_stats,
        "final_results": final_results,
        "training_statistics": training_stats,
        "interference_results": interference_results,
        "zero_shot_results": zero_shot_results
    }
    
    with open(os.path.join(args.output_dir, "multitask_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("✅ Multitask classification example completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Average accuracy: {final_results.get('average', {}).get('accuracy', 0):.4f}")
    logger.info(f"Number of tasks: {len(model.config.tasks)}")


if __name__ == "__main__":
    main()
