"""
Evaluation metrics for Adapter Tuning implementation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator(ABC):
    """Abstract base class for metrics calculation"""
    
    @abstractmethod
    def calculate(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate metrics"""
        pass
    
    @abstractmethod
    def get_detailed_report(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Get detailed evaluation report"""
        pass


class ClassificationMetrics(MetricsCalculator):
    """Metrics calculator for classification tasks"""
    
    def __init__(self, num_labels: int, label_names: Optional[List[str]] = None):
        self.num_labels = num_labels
        self.label_names = label_names or [f"Label_{i}" for i in range(num_labels)]
    
    def calculate(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics"""
        
        # Handle logits vs predictions
        if predictions.ndim > 1:
            pred_probs = predictions
            predictions = np.argmax(predictions, axis=1)
        else:
            pred_probs = None
        
        # Flatten if necessary
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        # Remove ignored labels (-100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        if len(predictions) == 0:
            logger.warning("No valid predictions found")
            return {}
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        metrics.update({
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
        })
        
        # AUC for binary classification
        if self.num_labels == 2 and pred_probs is not None:
            try:
                # Use probabilities for positive class
                if pred_probs.shape[1] == 2:
                    auc = roc_auc_score(labels, pred_probs[:, 1])
                    metrics["auc"] = auc
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
        
        return metrics
    
    def get_detailed_report(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Get detailed classification report"""
        
        # Handle logits vs predictions
        if predictions.ndim > 1:
            pred_probs = predictions
            predictions = np.argmax(predictions, axis=1)
        else:
            pred_probs = None
        
        # Flatten if necessary
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        # Remove ignored labels (-100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        if len(predictions) == 0:
            return {"error": "No valid predictions found"}
        
        # Basic metrics
        metrics = self.calculate(np.expand_dims(predictions, 1) if pred_probs is None else pred_probs, labels)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(
            labels, predictions,
            target_names=self.label_names[:len(np.unique(labels))],
            output_dict=True,
            zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for i, label_name in enumerate(self.label_names):
            if str(i) in report:
                per_class_metrics[label_name] = report[str(i)]
        
        return {
            "overall_metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": per_class_metrics,
            "classification_report": report,
        }


class TokenClassificationMetrics(MetricsCalculator):
    """Metrics calculator for token classification tasks (NER, POS tagging)"""
    
    def __init__(self, label_names: Optional[List[str]] = None, ignore_index: int = -100):
        self.label_names = label_names
        self.ignore_index = ignore_index
    
    def calculate(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate token classification metrics"""
        
        # Handle logits vs predictions
        if predictions.ndim > 2:
            predictions = np.argmax(predictions, axis=2)
        
        # Flatten and remove ignored labels
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for pred_id, label_id in zip(prediction, label):
                if label_id != self.ignore_index:
                    true_predictions.append(pred_id)
                    true_labels.append(label_id)
        
        if len(true_predictions) == 0:
            logger.warning("No valid predictions found")
            return {}
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, true_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_predictions, average='weighted', zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, true_predictions, average='macro', zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
        }
    
    def get_detailed_report(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Get detailed token classification report"""
        
        # Handle logits vs predictions
        if predictions.ndim > 2:
            predictions = np.argmax(predictions, axis=2)
        
        # Flatten and remove ignored labels
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for pred_id, label_id in zip(prediction, label):
                if label_id != self.ignore_index:
                    true_predictions.append(pred_id)
                    true_labels.append(label_id)
        
        if len(true_predictions) == 0:
            return {"error": "No valid predictions found"}
        
        # Basic metrics
        metrics = self.calculate(predictions, labels)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, true_predictions)
        
        # Classification report
        target_names = self.label_names if self.label_names else None
        report = classification_report(
            true_labels, true_predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            "overall_metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_tokens": len(true_predictions),
        }


class AdapterComparisonMetrics:
    """Metrics for comparing different adapters"""
    
    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        if task_type == "classification":
            self.metrics_calculator = ClassificationMetrics(num_labels=2)
        elif task_type == "token_classification":
            self.metrics_calculator = TokenClassificationMetrics()
        else:
            self.metrics_calculator = ClassificationMetrics(num_labels=2)
    
    def compare_adapters(
        self,
        adapter_results: Dict[str, Dict[str, Any]],
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """Compare multiple adapter results"""
        
        comparison = {
            "dataset": dataset_name,
            "adapters": {},
            "best_adapter": {},
            "metric_differences": {},
        }
        
        # Extract metrics for each adapter
        for adapter_name, results in adapter_results.items():
            if "metrics" in results:
                comparison["adapters"][adapter_name] = results["metrics"]
        
        # Find best adapter for each metric
        if comparison["adapters"]:
            all_metrics = set()
            for metrics in comparison["adapters"].values():
                all_metrics.update(metrics.keys())
            
            for metric in all_metrics:
                metric_values = {}
                for adapter_name, metrics in comparison["adapters"].items():
                    if metric in metrics:
                        metric_values[adapter_name] = metrics[metric]
                
                if metric_values:
                    best_adapter = max(metric_values, key=metric_values.get)
                    comparison["best_adapter"][metric] = {
                        "adapter": best_adapter,
                        "value": metric_values[best_adapter]
                    }
        
        return comparison
    
    def calculate_efficiency_metrics(
        self,
        adapter_info: Dict[str, Any],
        training_time: float,
        inference_time: float
    ) -> Dict[str, float]:
        """Calculate efficiency metrics for adapters"""
        
        total_params = adapter_info.get("total_params", 0)
        adapter_params = adapter_info.get("total_adapter_params", 0)
        
        efficiency_metrics = {
            "parameter_efficiency": (adapter_params / total_params) * 100 if total_params > 0 else 0,
            "training_time": training_time,
            "inference_time": inference_time,
            "params_per_second_training": adapter_params / training_time if training_time > 0 else 0,
            "params_per_second_inference": adapter_params / inference_time if inference_time > 0 else 0,
        }
        
        return efficiency_metrics


def get_metrics_calculator(task_type: str, **kwargs) -> MetricsCalculator:
    """Factory function to get appropriate metrics calculator"""
    
    if task_type in ["classification", "text_classification"]:
        return ClassificationMetrics(**kwargs)
    elif task_type in ["token_classification", "ner"]:
        return TokenClassificationMetrics(**kwargs)
    else:
        # Default to classification
        return ClassificationMetrics(**kwargs)
