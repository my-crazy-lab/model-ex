"""
Model evaluator for LoRA/PEFT implementation
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset
import torch
from transformers import pipeline
import logging

from ..models.peft_model import PEFTModelWrapper
from ..data.preprocessing import DataPreprocessor
from .metrics import get_metrics_calculator, MetricsCalculator

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluator"""
    
    def __init__(
        self,
        model_wrapper: PEFTModelWrapper,
        task_type: str = "classification",
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        self.model_wrapper = model_wrapper
        self.task_type = task_type
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize metrics calculator
        self.metrics_calculator = get_metrics_calculator(task_type)
        
        # Prepare model for inference
        self.model_wrapper.prepare_for_inference()
    
    def evaluate_dataset(
        self,
        dataset: Dataset,
        preprocessor: Optional[DataPreprocessor] = None,
        save_predictions: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset"""
        
        logger.info(f"Evaluating on dataset with {len(dataset)} examples")
        
        # Preprocess dataset if needed
        if preprocessor is not None:
            dataset = preprocessor.preprocess_dataset(dataset)
        
        # Get predictions
        predictions, labels = self._get_predictions(dataset)
        
        # Calculate metrics
        if self.task_type == "generation":
            # For generation tasks, convert token IDs back to text
            tokenizer = self.model_wrapper.get_tokenizer()
            pred_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
            label_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            
            metrics = self.metrics_calculator.calculate(pred_texts, label_texts)
            detailed_report = self.metrics_calculator.get_detailed_report(pred_texts, label_texts)
        else:
            metrics = self.metrics_calculator.calculate(predictions, labels)
            detailed_report = self.metrics_calculator.get_detailed_report(predictions, labels)
        
        # Prepare results
        results = {
            "metrics": metrics,
            "detailed_report": detailed_report,
            "num_examples": len(dataset),
            "task_type": self.task_type,
        }
        
        # Save predictions if requested
        if save_predictions and output_dir is not None:
            self._save_predictions(predictions, labels, output_dir)
        
        # Save evaluation results
        if output_dir is not None:
            self._save_results(results, output_dir)
        
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return results
    
    def _get_predictions(self, dataset: Dataset) -> tuple:
        """Get model predictions for dataset"""
        
        model = self.model_wrapper.peft_model
        tokenizer = self.model_wrapper.get_tokenizer()
        
        all_predictions = []
        all_labels = []
        
        # Process in batches
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            
            # Prepare inputs
            inputs = {
                "input_ids": torch.tensor(batch["input_ids"]).to(self.device),
                "attention_mask": torch.tensor(batch["attention_mask"]).to(self.device),
            }
            
            # Get predictions
            with torch.no_grad():
                if self.task_type == "generation":
                    # For generation tasks
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    predictions = outputs[:, inputs["input_ids"].shape[1]:]  # Remove input tokens
                else:
                    # For classification/regression tasks
                    outputs = model(**inputs)
                    predictions = outputs.logits
                
                all_predictions.append(predictions.cpu().numpy())
            
            # Get labels
            if "labels" in batch:
                labels = np.array(batch["labels"])
                all_labels.append(labels)
        
        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions, axis=0)
        if all_labels:
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_labels = np.array([])
        
        return all_predictions, all_labels
    
    def evaluate_samples(
        self,
        texts: List[str],
        labels: Optional[List] = None,
        save_predictions: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate model on individual text samples"""
        
        logger.info(f"Evaluating on {len(texts)} text samples")
        
        # Create pipeline for inference
        if self.task_type == "classification":
            pipe = pipeline(
                "text-classification",
                model=self.model_wrapper.peft_model,
                tokenizer=self.model_wrapper.get_tokenizer(),
                device=0 if self.device == "cuda" else -1,
            )
        elif self.task_type == "generation":
            pipe = pipeline(
                "text-generation",
                model=self.model_wrapper.peft_model,
                tokenizer=self.model_wrapper.get_tokenizer(),
                device=0 if self.device == "cuda" else -1,
            )
        else:
            raise ValueError(f"Pipeline not supported for task type: {self.task_type}")
        
        # Get predictions
        predictions = []
        for text in texts:
            try:
                result = pipe(text)
                if self.task_type == "classification":
                    predictions.append(result[0]["label"])
                elif self.task_type == "generation":
                    predictions.append(result[0]["generated_text"])
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                predictions.append(None)
        
        # Calculate metrics if labels provided
        results = {
            "predictions": predictions,
            "num_examples": len(texts),
            "task_type": self.task_type,
        }
        
        if labels is not None:
            if self.task_type == "generation":
                metrics = self.metrics_calculator.calculate(predictions, labels)
                detailed_report = self.metrics_calculator.get_detailed_report(predictions, labels)
            else:
                # Convert string labels to numeric if needed
                if isinstance(labels[0], str):
                    label_map = {label: i for i, label in enumerate(set(labels))}
                    numeric_labels = [label_map[label] for label in labels]
                else:
                    numeric_labels = labels
                
                # Convert predictions to numeric
                if isinstance(predictions[0], str):
                    pred_map = {pred: i for i, pred in enumerate(set(predictions))}
                    numeric_predictions = [pred_map.get(pred, -1) for pred in predictions]
                else:
                    numeric_predictions = predictions
                
                metrics = self.metrics_calculator.calculate(
                    np.array(numeric_predictions), np.array(numeric_labels)
                )
                detailed_report = self.metrics_calculator.get_detailed_report(
                    np.array(numeric_predictions), np.array(numeric_labels)
                )
            
            results.update({
                "metrics": metrics,
                "detailed_report": detailed_report,
                "labels": labels,
            })
        
        # Save results if requested
        if save_predictions and output_dir is not None:
            self._save_sample_results(texts, predictions, labels, output_dir)
        
        logger.info(f"Sample evaluation completed")
        return results
    
    def compare_models(
        self,
        other_evaluators: List["ModelEvaluator"],
        dataset: Dataset,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple models on the same dataset"""
        
        if model_names is None:
            model_names = [f"Model_{i}" for i in range(len(other_evaluators) + 1)]
        
        logger.info(f"Comparing {len(other_evaluators) + 1} models")
        
        # Evaluate all models
        all_results = {}
        
        # Evaluate current model
        results = self.evaluate_dataset(dataset)
        all_results[model_names[0]] = results["metrics"]
        
        # Evaluate other models
        for i, evaluator in enumerate(other_evaluators):
            results = evaluator.evaluate_dataset(dataset)
            all_results[model_names[i + 1]] = results["metrics"]
        
        # Create comparison report
        comparison = {
            "model_comparison": all_results,
            "best_model": {},
            "metric_differences": {},
        }
        
        # Find best model for each metric
        for metric in all_results[model_names[0]].keys():
            metric_values = {name: results.get(metric, 0) for name, results in all_results.items()}
            best_model = max(metric_values, key=metric_values.get)
            comparison["best_model"][metric] = best_model
        
        logger.info("Model comparison completed")
        return comparison
    
    def _save_predictions(self, predictions: np.ndarray, labels: np.ndarray, output_dir: str):
        """Save predictions to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, "predictions.npy"), predictions)
        np.save(os.path.join(output_dir, "labels.npy"), labels)
        
        logger.info(f"Predictions saved to {output_dir}")
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy(results)
        
        with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _save_sample_results(
        self,
        texts: List[str],
        predictions: List,
        labels: Optional[List],
        output_dir: str
    ):
        """Save sample evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, text in enumerate(texts):
            result = {
                "text": text,
                "prediction": predictions[i],
            }
            if labels is not None:
                result["label"] = labels[i]
            results.append(result)
        
        with open(os.path.join(output_dir, "sample_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Sample results saved to {output_dir}")
