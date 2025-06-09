"""
Evaluation metrics for LoRA/PEFT implementation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
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


class RegressionMetrics(MetricsCalculator):
    """Metrics calculator for regression tasks"""
    
    def calculate(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        
        # Flatten predictions and labels
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        # Remove ignored labels (-100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        if len(predictions) == 0:
            logger.warning("No valid predictions found")
            return {}
        
        # Calculate metrics
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        
        # Additional metrics
        mape = np.mean(np.abs((labels - predictions) / (labels + 1e-8))) * 100
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
        }
    
    def get_detailed_report(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Get detailed regression report"""
        
        metrics = self.calculate(predictions, labels)
        
        # Flatten for analysis
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        # Remove ignored labels (-100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        if len(predictions) == 0:
            return {"error": "No valid predictions found"}
        
        # Residual analysis
        residuals = labels - predictions
        
        return {
            "overall_metrics": metrics,
            "residual_stats": {
                "mean": np.mean(residuals),
                "std": np.std(residuals),
                "min": np.min(residuals),
                "max": np.max(residuals),
                "median": np.median(residuals),
            },
            "prediction_stats": {
                "mean": np.mean(predictions),
                "std": np.std(predictions),
                "min": np.min(predictions),
                "max": np.max(predictions),
            },
            "label_stats": {
                "mean": np.mean(labels),
                "std": np.std(labels),
                "min": np.min(labels),
                "max": np.max(labels),
            },
        }


class GenerationMetrics(MetricsCalculator):
    """Metrics calculator for text generation tasks"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def calculate(self, predictions: List[str], labels: List[str]) -> Dict[str, float]:
        """Calculate generation metrics"""
        
        if len(predictions) != len(labels):
            logger.warning("Predictions and labels have different lengths")
            min_len = min(len(predictions), len(labels))
            predictions = predictions[:min_len]
            labels = labels[:min_len]
        
        if len(predictions) == 0:
            return {}
        
        # Basic metrics
        exact_match = sum(1 for p, l in zip(predictions, labels) if p.strip() == l.strip()) / len(predictions)
        
        # Length statistics
        pred_lengths = [len(p.split()) for p in predictions]
        label_lengths = [len(l.split()) for l in labels]
        
        avg_pred_length = np.mean(pred_lengths)
        avg_label_length = np.mean(label_lengths)
        
        metrics = {
            "exact_match": exact_match,
            "avg_prediction_length": avg_pred_length,
            "avg_label_length": avg_label_length,
            "length_ratio": avg_pred_length / (avg_label_length + 1e-8),
        }
        
        # BLEU score (if available)
        try:
            from evaluate import load
            bleu = load("bleu")
            bleu_score = bleu.compute(predictions=predictions, references=[[l] for l in labels])
            metrics["bleu"] = bleu_score["bleu"]
        except Exception as e:
            logger.warning(f"Could not calculate BLEU score: {e}")
        
        # ROUGE score (if available)
        try:
            from evaluate import load
            rouge = load("rouge")
            rouge_score = rouge.compute(predictions=predictions, references=labels)
            metrics.update({
                "rouge1": rouge_score["rouge1"],
                "rouge2": rouge_score["rouge2"],
                "rougeL": rouge_score["rougeL"],
            })
        except Exception as e:
            logger.warning(f"Could not calculate ROUGE score: {e}")
        
        return metrics
    
    def get_detailed_report(self, predictions: List[str], labels: List[str]) -> Dict[str, Any]:
        """Get detailed generation report"""
        
        metrics = self.calculate(predictions, labels)
        
        # Sample predictions
        sample_size = min(10, len(predictions))
        samples = []
        for i in range(sample_size):
            samples.append({
                "prediction": predictions[i],
                "label": labels[i],
                "exact_match": predictions[i].strip() == labels[i].strip(),
            })
        
        return {
            "overall_metrics": metrics,
            "samples": samples,
            "total_examples": len(predictions),
        }


def get_metrics_calculator(task_type: str, **kwargs) -> MetricsCalculator:
    """Factory function to get appropriate metrics calculator"""
    
    if task_type == "classification":
        return ClassificationMetrics(**kwargs)
    elif task_type == "regression":
        return RegressionMetrics(**kwargs)
    elif task_type == "generation":
        return GenerationMetrics(**kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
