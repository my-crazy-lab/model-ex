"""
Model evaluator for Adapter Tuning implementation
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset
import torch
from transformers import pipeline
import logging

from ..adapters.adapter_model import AdapterModel
from ..data.preprocessing import DataPreprocessor
from .metrics import get_metrics_calculator, MetricsCalculator, AdapterComparisonMetrics

logger = logging.getLogger(__name__)


class AdapterEvaluator:
    """Comprehensive evaluator for adapter models"""
    
    def __init__(
        self,
        adapter_model: AdapterModel,
        task_type: str = "classification",
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        self.adapter_model = adapter_model
        self.task_type = task_type
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize metrics calculator
        self.metrics_calculator = get_metrics_calculator(task_type)
        
        # Prepare model for inference
        self.adapter_model.eval()
        
        # Get tokenizer
        self.tokenizer = None
        if hasattr(adapter_model, 'tokenizer'):
            self.tokenizer = adapter_model.tokenizer
    
    def evaluate_dataset(
        self,
        dataset: Dataset,
        preprocessor: Optional[DataPreprocessor] = None,
        save_predictions: bool = False,
        output_dir: Optional[str] = None,
        adapter_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate adapter model on a dataset"""
        
        logger.info(f"Evaluating adapter model on dataset with {len(dataset)} examples")
        
        # Preprocess dataset if needed
        if preprocessor is not None:
            dataset = preprocessor.preprocess_dataset(dataset)
        
        # Measure evaluation time
        start_time = time.time()
        
        # Get predictions
        predictions, labels = self._get_predictions(dataset)
        
        evaluation_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate(predictions, labels)
        detailed_report = self.metrics_calculator.get_detailed_report(predictions, labels)
        
        # Add efficiency metrics
        adapter_info = self.adapter_model.adapter_info
        efficiency_metrics = self._calculate_efficiency_metrics(
            adapter_info, evaluation_time, len(dataset)
        )
        
        # Prepare results
        results = {
            "metrics": metrics,
            "detailed_report": detailed_report,
            "efficiency_metrics": efficiency_metrics,
            "num_examples": len(dataset),
            "task_type": self.task_type,
            "adapter_name": adapter_name,
            "evaluation_time": evaluation_time,
        }
        
        # Save predictions if requested
        if save_predictions and output_dir is not None:
            self._save_predictions(predictions, labels, output_dir, adapter_name)
        
        # Save evaluation results
        if output_dir is not None:
            self._save_results(results, output_dir, adapter_name)
        
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return results
    
    def _get_predictions(self, dataset: Dataset) -> tuple:
        """Get model predictions for dataset"""
        
        model = self.adapter_model
        
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
            
            # Add token_type_ids if present
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = torch.tensor(batch["token_type_ids"]).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                
                if self.task_type == "token_classification":
                    # For token classification, we need the logits
                    predictions = outputs.logits.cpu().numpy()
                else:
                    # For sequence classification
                    predictions = outputs.logits.cpu().numpy()
                
                all_predictions.append(predictions)
            
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
        output_dir: Optional[str] = None,
        adapter_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate adapter model on individual text samples"""
        
        logger.info(f"Evaluating adapter model on {len(texts)} text samples")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available for sample evaluation")
        
        # Create pipeline for inference
        if self.task_type == "classification":
            pipe = pipeline(
                "text-classification",
                model=self.adapter_model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )
        elif self.task_type == "token_classification":
            pipe = pipeline(
                "token-classification",
                model=self.adapter_model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )
        else:
            raise ValueError(f"Pipeline not supported for task type: {self.task_type}")
        
        # Get predictions
        start_time = time.time()
        predictions = []
        
        for text in texts:
            try:
                result = pipe(text)
                if self.task_type == "classification":
                    predictions.append(result[0]["label"])
                elif self.task_type == "token_classification":
                    # Extract predicted labels for each token
                    token_predictions = [token["entity"] for token in result]
                    predictions.append(token_predictions)
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                predictions.append(None)
        
        evaluation_time = time.time() - start_time
        
        # Prepare results
        results = {
            "predictions": predictions,
            "num_examples": len(texts),
            "task_type": self.task_type,
            "adapter_name": adapter_name,
            "evaluation_time": evaluation_time,
        }
        
        # Calculate metrics if labels provided
        if labels is not None:
            # This would need task-specific implementation
            # For now, just store the labels
            results["labels"] = labels
        
        # Save results if requested
        if save_predictions and output_dir is not None:
            self._save_sample_results(texts, predictions, labels, output_dir, adapter_name)
        
        logger.info(f"Sample evaluation completed")
        return results
    
    def compare_adapters(
        self,
        other_evaluators: List["AdapterEvaluator"],
        dataset: Dataset,
        adapter_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple adapter models on the same dataset"""
        
        if adapter_names is None:
            adapter_names = [f"Adapter_{i}" for i in range(len(other_evaluators) + 1)]
        
        logger.info(f"Comparing {len(other_evaluators) + 1} adapter models")
        
        # Evaluate all adapters
        all_results = {}
        
        # Evaluate current adapter
        results = self.evaluate_dataset(dataset)
        all_results[adapter_names[0]] = results
        
        # Evaluate other adapters
        for i, evaluator in enumerate(other_evaluators):
            results = evaluator.evaluate_dataset(dataset)
            all_results[adapter_names[i + 1]] = results
        
        # Create comparison using AdapterComparisonMetrics
        comparison_metrics = AdapterComparisonMetrics(self.task_type)
        comparison = comparison_metrics.compare_adapters(all_results)
        
        logger.info("Adapter comparison completed")
        return comparison
    
    def _calculate_efficiency_metrics(
        self,
        adapter_info: Dict[str, Any],
        evaluation_time: float,
        num_examples: int
    ) -> Dict[str, float]:
        """Calculate efficiency metrics"""
        
        total_params = adapter_info.get("total_params", 0)
        adapter_params = adapter_info.get("total_adapter_params", 0)
        
        efficiency_metrics = {
            "parameter_efficiency": adapter_info.get("adapter_percentage", 0),
            "examples_per_second": num_examples / evaluation_time if evaluation_time > 0 else 0,
            "time_per_example": evaluation_time / num_examples if num_examples > 0 else 0,
            "adapter_params": adapter_params,
            "total_params": total_params,
        }
        
        return efficiency_metrics
    
    def _save_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        output_dir: str,
        adapter_name: Optional[str] = None
    ):
        """Save predictions to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        prefix = f"{adapter_name}_" if adapter_name else ""
        
        np.save(os.path.join(output_dir, f"{prefix}predictions.npy"), predictions)
        np.save(os.path.join(output_dir, f"{prefix}labels.npy"), labels)
        
        logger.info(f"Predictions saved to {output_dir}")
    
    def _save_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        adapter_name: Optional[str] = None
    ):
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
        
        prefix = f"{adapter_name}_" if adapter_name else ""
        filename = f"{prefix}evaluation_results.json"
        
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _save_sample_results(
        self,
        texts: List[str],
        predictions: List,
        labels: Optional[List],
        output_dir: str,
        adapter_name: Optional[str] = None
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
        
        prefix = f"{adapter_name}_" if adapter_name else ""
        filename = f"{prefix}sample_results.json"
        
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Sample results saved to {output_dir}")
