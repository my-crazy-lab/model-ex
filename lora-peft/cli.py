#!/usr/bin/env python3
"""
Command Line Interface for LoRA/PEFT implementation
"""

import argparse
import logging
import sys
from pathlib import Path

from config import ModelConfig, PEFTConfig, TrainingConfig, CLASSIFICATION_CONFIG, GENERATION_CONFIG, QA_CONFIG
from data import load_dataset_from_hub, get_preprocessor
from training import PEFTTrainer
from evaluation import ModelEvaluator
from inference import InferencePipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_command(args):
    """Train a PEFT model"""
    logger.info(f"Starting training for {args.task} task")
    
    # Model configuration
    model_config = ModelConfig(
        model_name_or_path=args.model,
        num_labels=args.num_labels,
        max_length=args.max_length,
        use_quantization=args.quantization,
        quantization_bits=args.quantization_bits,
    )
    
    # PEFT configuration
    if args.task == "classification":
        peft_config = CLASSIFICATION_CONFIG
    elif args.task == "generation":
        peft_config = GENERATION_CONFIG
    elif args.task == "qa":
        peft_config = QA_CONFIG
    else:
        raise ValueError(f"Unsupported task: {args.task}")
    
    # Override PEFT parameters if provided
    if args.lora_r:
        peft_config.r = args.lora_r
    if args.lora_alpha:
        peft_config.lora_alpha = args.lora_alpha
    if args.lora_dropout:
        peft_config.lora_dropout = args.lora_dropout
    
    # Training configuration
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps" if args.eval_steps else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="wandb" if args.wandb else None,
    )
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset in ["imdb", "sst2", "cola"]:
        from data.data_loader import COMMON_DATASETS
        dataset_info = COMMON_DATASETS[args.dataset]
        dataset = load_dataset_from_hub(dataset_info["name"], dataset_info.get("subset"))
    else:
        dataset = load_dataset_from_hub(args.dataset)
    
    # Take subset if specified
    if args.train_size:
        dataset["train"] = dataset["train"].select(range(args.train_size))
    if args.eval_size and "validation" in dataset:
        dataset["validation"] = dataset["validation"].select(range(args.eval_size))
    
    # Setup trainer
    trainer = PEFTTrainer(
        model_config=model_config,
        peft_config=peft_config,
        training_config=training_config,
        task_type=args.task
    )
    
    # Setup preprocessor
    model_wrapper = trainer.setup_model()
    tokenizer = model_wrapper.get_tokenizer()
    
    preprocessor = get_preprocessor(
        task_type=args.task,
        tokenizer=tokenizer,
        max_length=model_config.max_length
    )
    
    # Train
    train_result = trainer.train(
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        preprocessor=preprocessor
    )
    
    logger.info(f"Training completed: {train_result}")


def evaluate_command(args):
    """Evaluate a trained model"""
    logger.info(f"Evaluating model: {args.model_path}")
    
    # Load model for inference
    pipeline = InferencePipeline.from_pretrained(args.model_path)
    
    # Load dataset
    if args.dataset:
        dataset = load_dataset_from_hub(args.dataset)
        eval_dataset = dataset.get("test", dataset.get("validation"))
        
        if args.eval_size:
            eval_dataset = eval_dataset.select(range(args.eval_size))
        
        # Create evaluator
        evaluator = ModelEvaluator(
            model_wrapper=pipeline.model_wrapper,
            task_type=args.task
        )
        
        # Evaluate
        results = evaluator.evaluate_dataset(
            eval_dataset,
            save_predictions=True,
            output_dir=args.output_dir
        )
        
        logger.info(f"Evaluation results: {results['metrics']}")
    else:
        logger.info("No dataset specified for evaluation")


def infer_command(args):
    """Run inference with a trained model"""
    logger.info(f"Running inference with model: {args.model_path}")
    
    # Load model
    pipeline = InferencePipeline.from_pretrained(args.model_path)
    
    if args.text:
        # Single text inference
        if args.task == "classification":
            result = pipeline.classify_text(args.text, return_all_scores=True)
            logger.info(f"Input: {args.text}")
            logger.info(f"Result: {result}")
        elif args.task == "generation":
            result = pipeline.generate_text(
                args.text,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            logger.info(f"Input: {args.text}")
            logger.info(f"Generated: {result}")
        elif args.task == "qa":
            if not args.context:
                logger.error("Context required for QA task")
                return
            result = pipeline.answer_question(args.text, args.context)
            logger.info(f"Question: {args.text}")
            logger.info(f"Context: {args.context}")
            logger.info(f"Answer: {result}")
    
    elif args.input_file:
        # Batch inference from file
        with open(args.input_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if args.task == "classification":
            results = pipeline.classify_text(texts)
        elif args.task == "generation":
            results = pipeline.generate_text(texts, max_new_tokens=args.max_tokens)
        else:
            logger.error("Batch QA not supported via CLI")
            return
        
        # Save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                for text, result in zip(texts, results):
                    f.write(f"Input: {text}\n")
                    f.write(f"Output: {result}\n\n")
            logger.info(f"Results saved to {args.output_file}")
        else:
            for text, result in zip(texts, results):
                logger.info(f"Input: {text}")
                logger.info(f"Output: {result}")
                logger.info("-" * 50)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="LoRA/PEFT CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a PEFT model")
    train_parser.add_argument("--task", choices=["classification", "generation", "qa"], required=True)
    train_parser.add_argument("--model", default="bert-base-uncased", help="Base model name")
    train_parser.add_argument("--dataset", required=True, help="Dataset name")
    train_parser.add_argument("--output-dir", default="./results", help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    train_parser.add_argument("--num-labels", type=int, default=2, help="Number of labels")
    train_parser.add_argument("--lora-r", type=int, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    train_parser.add_argument("--lora-dropout", type=float, help="LoRA dropout")
    train_parser.add_argument("--quantization", action="store_true", help="Use quantization")
    train_parser.add_argument("--quantization-bits", type=int, default=4, choices=[4, 8])
    train_parser.add_argument("--train-size", type=int, help="Limit training set size")
    train_parser.add_argument("--eval-size", type=int, help="Limit eval set size")
    train_parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation steps")
    train_parser.add_argument("--save-steps", type=int, default=500, help="Save steps")
    train_parser.add_argument("--logging-steps", type=int, default=100, help="Logging steps")
    train_parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model-path", required=True, help="Path to trained model")
    eval_parser.add_argument("--dataset", help="Dataset for evaluation")
    eval_parser.add_argument("--task", choices=["classification", "generation", "qa"], default="classification")
    eval_parser.add_argument("--eval-size", type=int, help="Limit eval set size")
    eval_parser.add_argument("--output-dir", default="./evaluation", help="Output directory")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", required=True, help="Path to trained model")
    infer_parser.add_argument("--task", choices=["classification", "generation", "qa"], default="classification")
    infer_parser.add_argument("--text", help="Input text")
    infer_parser.add_argument("--context", help="Context for QA")
    infer_parser.add_argument("--input-file", help="File with input texts")
    infer_parser.add_argument("--output-file", help="Output file for results")
    infer_parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens for generation")
    infer_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "infer":
        infer_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
