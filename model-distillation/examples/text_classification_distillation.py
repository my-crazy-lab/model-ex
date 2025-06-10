"""
Complete model distillation text classification example
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
import time

from config import DistillationConfig, TeacherConfig, StudentConfig
from distillation import Distiller, TeacherModel, StudentModel

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
            "This movie is absolutely fantastic and amazing!",
            "Great acting and wonderful storyline here.",
            "Excellent cinematography and brilliant direction.",
            "Terrible film, complete waste of time and money.",
            "Boring and very poorly made movie overall.",
            "Worst movie I have ever seen in my life."
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
    
    return {"accuracy": accuracy}


def benchmark_inference_speed(model, dataloader, model_name: str, device: torch.device):
    """Benchmark model inference speed"""
    
    logger.info(f"🚀 Benchmarking {model_name} inference speed...")
    
    model.eval()
    total_time = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            start_time = time.time()
            outputs = model(**batch)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_samples += batch['input_ids'].size(0)
    
    avg_time_per_sample = total_time / total_samples
    samples_per_second = total_samples / total_time
    
    logger.info(f"{model_name} Performance:")
    logger.info(f"  Total time: {total_time:.3f}s")
    logger.info(f"  Avg time per sample: {avg_time_per_sample*1000:.2f}ms")
    logger.info(f"  Samples per second: {samples_per_second:.1f}")
    
    return {
        "total_time": total_time,
        "avg_time_per_sample": avg_time_per_sample,
        "samples_per_second": samples_per_second
    }


def analyze_model_size(model, model_name: str):
    """Analyze model size and memory usage"""
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size (float32)
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    logger.info(f"{model_name} Analysis:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {model_size_mb:.1f} MB")
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": model_size_mb
    }


def main():
    """Main model distillation example"""
    
    parser = argparse.ArgumentParser(description="Model Distillation Text Classification Example")
    parser.add_argument("--dataset", default="dummy", choices=["imdb", "sst2", "ag_news", "dummy"], 
                       help="Dataset to use")
    parser.add_argument("--teacher_model", default="bert-base-uncased", 
                       help="Teacher model name")
    parser.add_argument("--student_model", default="distilbert-base-uncased", 
                       help="Student model name")
    parser.add_argument("--sample_size", type=int, default=500, 
                       help="Sample size for demonstration")
    parser.add_argument("--output_dir", default="./distillation_results", 
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                       help="Learning rate")
    parser.add_argument("--temperature", type=float, default=4.0, 
                       help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7, 
                       help="Distillation loss weight")
    parser.add_argument("--distillation_type", default="logit", 
                       choices=["logit", "feature", "attention", "combined"],
                       help="Type of distillation")
    parser.add_argument("--benchmark_speed", action="store_true", 
                       help="Benchmark inference speed")
    
    args = parser.parse_args()
    
    logger.info("🎓 Starting Model Distillation example...")
    logger.info(f"Teacher: {args.teacher_model}")
    logger.info(f"Student: {args.student_model}")
    logger.info(f"Distillation type: {args.distillation_type}")
    
    # Load dataset
    train_dataset, test_dataset, num_labels = load_sample_dataset(args.dataset, args.sample_size)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Number of labels: {num_labels}")
    
    # Setup configurations
    teacher_config = TeacherConfig(
        model_name_or_path=args.teacher_model,
        task_type="classification",
        num_labels=num_labels,
        max_length=128
    )
    
    student_config = StudentConfig(
        model_name_or_path=args.student_model,
        task_type="classification",
        num_labels=num_labels,
        max_length=128
    )
    
    distillation_config = DistillationConfig(
        distillation_type=args.distillation_type,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=1.0 - args.alpha,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        eval_steps=100,
        logging_steps=50
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_config.model_name_or_path)
    
    # Preprocess datasets
    logger.info("📊 Preprocessing datasets...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer, teacher_config.max_length)
    test_dataset = preprocess_dataset(test_dataset, tokenizer, teacher_config.max_length)
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False
    )
    
    # Create teacher and student models
    logger.info("🏫 Creating teacher model...")
    teacher = TeacherModel(teacher_config)
    
    logger.info("🎓 Creating student model...")
    student = StudentModel(student_config)
    
    # Analyze model sizes
    teacher_analysis = analyze_model_size(teacher, "Teacher")
    student_analysis = analyze_model_size(student, "Student")
    
    compression_ratio = teacher_analysis["total_parameters"] / student_analysis["total_parameters"]
    size_reduction = (1 - student_analysis["model_size_mb"] / teacher_analysis["model_size_mb"]) * 100
    
    logger.info(f"\n📊 COMPRESSION ANALYSIS:")
    logger.info(f"Compression ratio: {compression_ratio:.1f}x")
    logger.info(f"Size reduction: {size_reduction:.1f}%")
    
    # Create distiller
    logger.info("🔬 Creating distiller...")
    distiller = Distiller(teacher, student, distillation_config, tokenizer)
    
    # Benchmark speed before distillation (optional)
    if args.benchmark_speed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher_speed = benchmark_inference_speed(teacher, test_dataloader, "Teacher", device)
        student_speed = benchmark_inference_speed(student, test_dataloader, "Student (before distillation)", device)
        
        speedup = teacher_speed["avg_time_per_sample"] / student_speed["avg_time_per_sample"]
        logger.info(f"Student speedup: {speedup:.1f}x faster than teacher")
    
    # Perform distillation
    logger.info("🎯 Starting knowledge distillation...")
    training_stats = distiller.distill(
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        num_epochs=args.epochs,
        save_dir=args.output_dir
    )
    
    # Final evaluation and comparison
    logger.info("📈 Final evaluation...")
    comparison = distiller.compare_models(test_dataloader)
    
    # Print comprehensive results
    logger.info("\n" + "="*60)
    logger.info("🎓 MODEL DISTILLATION RESULTS")
    logger.info("="*60)
    
    # Model comparison
    logger.info(f"Model Compression:")
    logger.info(f"  Teacher parameters: {teacher_analysis['total_parameters']:,}")
    logger.info(f"  Student parameters: {student_analysis['total_parameters']:,}")
    logger.info(f"  Compression ratio: {compression_ratio:.1f}x")
    logger.info(f"  Size reduction: {size_reduction:.1f}%")
    
    # Performance comparison
    logger.info(f"\nPerformance Comparison:")
    logger.info(f"  Teacher accuracy: {comparison['teacher']['accuracy']:.4f}")
    logger.info(f"  Student accuracy: {comparison['student']['accuracy']:.4f}")
    logger.info(f"  Performance retention: {comparison['performance_retention']:.1f}%")
    logger.info(f"  Accuracy drop: {comparison['accuracy_drop']:.4f}")
    
    # Speed comparison
    if args.benchmark_speed:
        teacher_speed_final = comparison['teacher']['samples_per_second']
        student_speed_final = comparison['student']['samples_per_second']
        speedup_final = student_speed_final / teacher_speed_final
        
        logger.info(f"\nSpeed Comparison:")
        logger.info(f"  Teacher speed: {teacher_speed_final:.1f} samples/sec")
        logger.info(f"  Student speed: {student_speed_final:.1f} samples/sec")
        logger.info(f"  Speedup: {speedup_final:.1f}x")
    
    # Training statistics
    final_stats = distiller.get_training_statistics()
    logger.info(f"\nTraining Statistics:")
    logger.info(f"  Total steps: {final_stats['total_steps']}")
    logger.info(f"  Average total loss: {final_stats.get('avg_total_loss', 0):.4f}")
    logger.info(f"  Average distillation loss: {final_stats.get('avg_distillation_loss', 0):.4f}")
    logger.info(f"  Average task loss: {final_stats.get('avg_task_loss', 0):.4f}")
    logger.info(f"  Final temperature: {final_stats['final_temperature']:.2f}")
    
    logger.info("="*60)
    
    # Test inference on sample texts
    logger.info("🧪 Testing inference on sample texts...")
    test_texts = [
        "This movie is absolutely fantastic and amazing!",
        "Terrible film, complete waste of time and money.",
        "An okay movie, nothing particularly special about it."
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    student.to(device)
    
    teacher.eval()
    student.eval()
    
    for text in test_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=teacher_config.max_length
        ).to(device)
        
        with torch.no_grad():
            # Teacher prediction
            teacher_outputs = teacher(**inputs)
            teacher_probs = torch.softmax(teacher_outputs.logits, dim=-1)
            teacher_pred = torch.argmax(teacher_probs, dim=-1).item()
            teacher_conf = teacher_probs[0][teacher_pred].item()
            
            # Student prediction
            student_outputs = student(**inputs)
            student_probs = torch.softmax(student_outputs.logits, dim=-1)
            student_pred = torch.argmax(student_probs, dim=-1).item()
            student_conf = student_probs[0][student_pred].item()
        
        if num_labels == 2:
            teacher_sentiment = "Positive" if teacher_pred == 1 else "Negative"
            student_sentiment = "Positive" if student_pred == 1 else "Negative"
        else:
            teacher_sentiment = f"Class {teacher_pred}"
            student_sentiment = f"Class {student_pred}"
        
        logger.info(f"Text: '{text}'")
        logger.info(f"Teacher: {teacher_sentiment} (confidence: {teacher_conf:.3f})")
        logger.info(f"Student: {student_sentiment} (confidence: {student_conf:.3f})")
        logger.info("-" * 40)
    
    # Save comprehensive results
    results_summary = {
        "dataset": args.dataset,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "distillation_config": distillation_config.to_dict(),
        "model_analysis": {
            "teacher": teacher_analysis,
            "student": student_analysis,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": size_reduction
        },
        "performance_comparison": comparison,
        "training_statistics": final_stats
    }
    
    with open(os.path.join(args.output_dir, "distillation_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("✅ Model distillation example completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Compression ratio: {compression_ratio:.1f}x")
    logger.info(f"Performance retention: {comparison['performance_retention']:.1f}%")
    logger.info(f"Final student accuracy: {comparison['student']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
