"""
Complete BERT quantization example
"""

import argparse
import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import numpy as np
import json
import os
import time

from config import QuantizationConfig, QuantizationType, QuantizationBackend
from compression import ModelQuantizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_dataset(dataset_name: str = "glue", task: str = "sst2", sample_size: int = 1000):
    """Load and sample dataset for demonstration"""
    
    logger.info(f"Loading {dataset_name}/{task} dataset...")
    
    if dataset_name == "glue":
        dataset = load_dataset("glue", task)
        
        # Sample data for quick demonstration
        if task == "sst2":
            train_data = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
            test_data = dataset["validation"].select(range(min(sample_size // 4, len(dataset["validation"]))))
        else:
            train_data = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
            test_data = dataset["validation"].select(range(min(sample_size // 4, len(dataset["validation"]))))
        
        return train_data, test_data, 2  # Binary classification
    
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
        
        from datasets import Dataset
        
        # Create datasets
        train_size = int(0.8 * len(texts))
        train_data = Dataset.from_dict({
            "sentence": texts[:train_size],
            "label": labels[:train_size]
        })
        test_data = Dataset.from_dict({
            "sentence": texts[train_size:],
            "label": labels[train_size:]
        })
        
        return train_data, test_data, 2  # 2 classes


def preprocess_dataset(dataset, tokenizer, max_length=128):
    """Preprocess dataset for training"""
    
    def preprocess_function(examples):
        # Handle different text column names
        if "sentence" in examples:
            texts = examples["sentence"]
        elif "text" in examples:
            texts = examples["text"]
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


def evaluate_model(model, dataloader, device, model_name="Model"):
    """Evaluate model accuracy"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_time = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Measure inference time
            start_time = time.time()
            outputs = model(**batch)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            total_time += (end_time - start_time)
            
            # Collect predictions
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_time_per_sample = total_time / len(all_labels)
    samples_per_second = len(all_labels) / total_time
    
    logger.info(f"{model_name} Performance:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Avg time per sample: {avg_time_per_sample*1000:.2f}ms")
    logger.info(f"  Samples per second: {samples_per_second:.1f}")
    
    return {
        "accuracy": accuracy,
        "avg_time_per_sample": avg_time_per_sample,
        "samples_per_second": samples_per_second,
        "total_time": total_time
    }


def analyze_model_size(model, model_name="Model"):
    """Analyze model size and memory usage"""
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate model size
    if hasattr(model, 'state_dict'):
        # PyTorch model
        model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    else:
        # Estimate based on parameters
        model_size_bytes = total_params * 4  # Assume FP32
    
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    logger.info(f"{model_name} Analysis:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Model size: {model_size_mb:.1f} MB")
    
    return {
        "total_parameters": total_params,
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_mb
    }


def compare_quantization_methods(
    original_model,
    tokenizer,
    calibration_dataloader,
    test_dataloader,
    device,
    example_inputs
):
    """Compare different quantization methods"""
    
    logger.info("🔍 Comparing quantization methods...")
    
    results = {}
    
    # Original model
    logger.info("Evaluating original model...")
    original_results = evaluate_model(original_model, test_dataloader, device, "Original FP32")
    original_analysis = analyze_model_size(original_model, "Original FP32")
    
    results["original"] = {
        "performance": original_results,
        "analysis": original_analysis
    }
    
    # INT8 Dynamic Quantization
    logger.info("Testing INT8 Dynamic Quantization...")
    try:
        int8_config = QuantizationConfig(
            quantization_type=QuantizationType.DYNAMIC,
            backend=QuantizationBackend.PYTORCH,
            bits=8
        )
        int8_quantizer = ModelQuantizer(int8_config)
        int8_model = int8_quantizer.quantize(original_model.cpu())
        int8_model.to(device)
        
        int8_results = evaluate_model(int8_model, test_dataloader, device, "INT8 Dynamic")
        int8_analysis = analyze_model_size(int8_model, "INT8 Dynamic")
        
        results["int8_dynamic"] = {
            "performance": int8_results,
            "analysis": int8_analysis
        }
    except Exception as e:
        logger.error(f"INT8 Dynamic quantization failed: {e}")
        results["int8_dynamic"] = {"error": str(e)}
    
    # INT8 Static Quantization
    logger.info("Testing INT8 Static Quantization...")
    try:
        int8_static_config = QuantizationConfig(
            quantization_type=QuantizationType.STATIC,
            backend=QuantizationBackend.PYTORCH,
            bits=8,
            num_calibration_samples=100
        )
        int8_static_quantizer = ModelQuantizer(int8_static_config)
        int8_static_model = int8_static_quantizer.quantize(
            original_model.cpu(), 
            calibration_dataloader
        )
        int8_static_model.to(device)
        
        int8_static_results = evaluate_model(int8_static_model, test_dataloader, device, "INT8 Static")
        int8_static_analysis = analyze_model_size(int8_static_model, "INT8 Static")
        
        results["int8_static"] = {
            "performance": int8_static_results,
            "analysis": int8_static_analysis
        }
    except Exception as e:
        logger.error(f"INT8 Static quantization failed: {e}")
        results["int8_static"] = {"error": str(e)}
    
    # FP16 Mixed Precision
    logger.info("Testing FP16 Mixed Precision...")
    try:
        fp16_config = QuantizationConfig(
            quantization_type=QuantizationType.FP16,
            backend=QuantizationBackend.PYTORCH
        )
        fp16_quantizer = ModelQuantizer(fp16_config)
        fp16_model = fp16_quantizer.quantize(original_model.cpu())
        fp16_model.to(device)
        
        fp16_results = evaluate_model(fp16_model, test_dataloader, device, "FP16")
        fp16_analysis = analyze_model_size(fp16_model, "FP16")
        
        results["fp16"] = {
            "performance": fp16_results,
            "analysis": fp16_analysis
        }
    except Exception as e:
        logger.error(f"FP16 quantization failed: {e}")
        results["fp16"] = {"error": str(e)}
    
    return results


def main():
    """Main BERT quantization example"""
    
    parser = argparse.ArgumentParser(description="BERT Quantization Example")
    parser.add_argument("--model_name", default="bert-base-uncased", 
                       help="Model name or path")
    parser.add_argument("--dataset", default="glue", 
                       help="Dataset name")
    parser.add_argument("--task", default="sst2", 
                       help="Task name")
    parser.add_argument("--sample_size", type=int, default=500, 
                       help="Sample size for demonstration")
    parser.add_argument("--output_dir", default="./quantization_results", 
                       help="Output directory")
    parser.add_argument("--quantization_type", default="int8", 
                       choices=["int8", "int4", "fp16", "dynamic", "static"],
                       help="Quantization type")
    parser.add_argument("--backend", default="pytorch", 
                       choices=["pytorch", "bitsandbytes", "onnx"],
                       help="Quantization backend")
    parser.add_argument("--compare_methods", action="store_true", 
                       help="Compare different quantization methods")
    parser.add_argument("--calibration_samples", type=int, default=100, 
                       help="Number of calibration samples")
    
    args = parser.parse_args()
    
    logger.info("⚡ Starting BERT Quantization example...")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Quantization: {args.quantization_type}")
    logger.info(f"Backend: {args.backend}")
    
    # Load dataset
    train_dataset, test_dataset, num_labels = load_sample_dataset(
        args.dataset, args.task, args.sample_size
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels
    )
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer)
    test_dataset = preprocess_dataset(test_dataset, tokenizer)
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    calibration_dataloader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    logger.info(f"Using device: {device}")
    
    # Create example inputs for benchmarking
    example_batch = next(iter(test_dataloader))
    example_inputs = example_batch["input_ids"][:1]  # Single example
    
    if args.compare_methods:
        # Compare different quantization methods
        comparison_results = compare_quantization_methods(
            model, tokenizer, calibration_dataloader, test_dataloader, device, example_inputs
        )
        
        # Print comparison results
        logger.info("\n" + "="*80)
        logger.info("⚡ QUANTIZATION COMPARISON RESULTS")
        logger.info("="*80)
        
        for method_name, method_results in comparison_results.items():
            if "error" in method_results:
                logger.info(f"{method_name.upper()}: FAILED - {method_results['error']}")
                continue
            
            perf = method_results["performance"]
            analysis = method_results["analysis"]
            
            logger.info(f"\n{method_name.upper()}:")
            logger.info(f"  Accuracy: {perf['accuracy']:.4f}")
            logger.info(f"  Model size: {analysis['model_size_mb']:.1f} MB")
            logger.info(f"  Samples/sec: {perf['samples_per_second']:.1f}")
            
            if method_name != "original":
                original_size = comparison_results["original"]["analysis"]["model_size_mb"]
                original_speed = comparison_results["original"]["performance"]["samples_per_second"]
                original_acc = comparison_results["original"]["performance"]["accuracy"]
                
                size_reduction = (1 - analysis["model_size_mb"] / original_size) * 100
                speed_improvement = perf["samples_per_second"] / original_speed
                accuracy_drop = original_acc - perf["accuracy"]
                
                logger.info(f"  Size reduction: {size_reduction:.1f}%")
                logger.info(f"  Speed improvement: {speed_improvement:.1f}x")
                logger.info(f"  Accuracy drop: {accuracy_drop:.4f}")
        
        logger.info("="*80)
        
        # Save comparison results
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "quantization_comparison.json"), 'w') as f:
            json.dump(comparison_results, f, indent=2)
    
    else:
        # Single quantization method
        logger.info(f"🔧 Applying {args.quantization_type} quantization...")
        
        # Setup quantization configuration
        if args.quantization_type == "int8":
            quant_type = QuantizationType.INT8
        elif args.quantization_type == "int4":
            quant_type = QuantizationType.INT4
        elif args.quantization_type == "fp16":
            quant_type = QuantizationType.FP16
        elif args.quantization_type == "dynamic":
            quant_type = QuantizationType.DYNAMIC
        elif args.quantization_type == "static":
            quant_type = QuantizationType.STATIC
        else:
            raise ValueError(f"Unsupported quantization type: {args.quantization_type}")
        
        config = QuantizationConfig(
            quantization_type=quant_type,
            backend=QuantizationBackend(args.backend),
            num_calibration_samples=args.calibration_samples,
            output_dir=args.output_dir
        )
        
        # Create quantizer
        quantizer = ModelQuantizer(config)
        
        # Analyze original model
        logger.info("📊 Analyzing original model...")
        original_results = evaluate_model(model, test_dataloader, device, "Original")
        original_analysis = analyze_model_size(model, "Original")
        
        # Quantize model
        logger.info("⚡ Quantizing model...")
        if quant_type == QuantizationType.STATIC:
            quantized_model = quantizer.quantize(model.cpu(), calibration_dataloader, example_inputs)
        else:
            quantized_model = quantizer.quantize(model.cpu(), example_inputs=example_inputs)
        
        quantized_model.to(device)
        
        # Analyze quantized model
        logger.info("📊 Analyzing quantized model...")
        quantized_results = evaluate_model(quantized_model, test_dataloader, device, "Quantized")
        quantized_analysis = analyze_model_size(quantized_model, "Quantized")
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("⚡ QUANTIZATION RESULTS")
        logger.info("="*60)
        
        logger.info(f"Original Model:")
        logger.info(f"  Accuracy: {original_results['accuracy']:.4f}")
        logger.info(f"  Model size: {original_analysis['model_size_mb']:.1f} MB")
        logger.info(f"  Samples/sec: {original_results['samples_per_second']:.1f}")
        
        logger.info(f"\nQuantized Model ({args.quantization_type.upper()}):")
        logger.info(f"  Accuracy: {quantized_results['accuracy']:.4f}")
        logger.info(f"  Model size: {quantized_analysis['model_size_mb']:.1f} MB")
        logger.info(f"  Samples/sec: {quantized_results['samples_per_second']:.1f}")
        
        # Compute improvements
        size_reduction = (1 - quantized_analysis["model_size_mb"] / original_analysis["model_size_mb"]) * 100
        speed_improvement = quantized_results["samples_per_second"] / original_results["samples_per_second"]
        accuracy_drop = original_results["accuracy"] - quantized_results["accuracy"]
        
        logger.info(f"\nImprovements:")
        logger.info(f"  Size reduction: {size_reduction:.1f}%")
        logger.info(f"  Speed improvement: {speed_improvement:.1f}x")
        logger.info(f"  Accuracy drop: {accuracy_drop:.4f}")
        
        logger.info("="*60)
        
        # Save quantized model
        quantizer.save_quantized_model(quantized_model, args.output_dir)
        
        # Save results
        results_summary = {
            "quantization_config": config.to_dict(),
            "original_model": {
                "performance": original_results,
                "analysis": original_analysis
            },
            "quantized_model": {
                "performance": quantized_results,
                "analysis": quantized_analysis
            },
            "improvements": {
                "size_reduction_percent": size_reduction,
                "speed_improvement": speed_improvement,
                "accuracy_drop": accuracy_drop
            }
        }
        
        with open(os.path.join(args.output_dir, "quantization_results.json"), 'w') as f:
            json.dump(results_summary, f, indent=2)
    
    logger.info("✅ BERT quantization example completed!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
