"""
SHAP (SHapley Additive exPlanations) Implementation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Callable
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseSHAPExplainer:
    """
    Base class for SHAP explainers
    """
    
    def __init__(self, model, explainer_type: str = 'auto', **kwargs):
        self.model = model
        self.explainer_type = explainer_type
        self.explainer = None
        self.kwargs = kwargs
        
        logger.info(f"Initialized SHAP explainer with type: {explainer_type}")
    
    def explain_instance(self, instance, **kwargs):
        """
        Explain a single instance
        """
        raise NotImplementedError("Subclasses must implement explain_instance")
    
    def explain_batch(self, instances, **kwargs):
        """
        Explain multiple instances
        """
        raise NotImplementedError("Subclasses must implement explain_batch")
    
    def plot_explanation(self, shap_values, **kwargs):
        """
        Plot SHAP explanation
        """
        raise NotImplementedError("Subclasses must implement plot_explanation")


class SHAPTextExplainer(BaseSHAPExplainer):
    """
    SHAP explainer for text models
    """
    
    def __init__(
        self,
        model,
        tokenizer=None,
        explainer_type: str = 'auto',
        background_data: Optional[List[str]] = None,
        max_evals: int = 500,
        batch_size: int = 50,
        **kwargs
    ):
        super().__init__(model, explainer_type, **kwargs)
        
        self.tokenizer = tokenizer
        self.background_data = background_data
        self.max_evals = max_evals
        self.batch_size = batch_size
        
        # Initialize appropriate SHAP explainer
        self._initialize_explainer()
        
        logger.info("Initialized SHAP text explainer")
    
    def _initialize_explainer(self):
        """
        Initialize the appropriate SHAP explainer based on model type
        """
        try:
            if self.explainer_type == 'auto':
                # Auto-detect explainer type
                if hasattr(self.model, 'predict_proba'):
                    # Scikit-learn style model
                    self.explainer = shap.Explainer(self.model)
                elif isinstance(self.model, (nn.Module, torch.nn.Module)):
                    # PyTorch model
                    if self.background_data is not None:
                        self.explainer = shap.DeepExplainer(self.model, self.background_data)
                    else:
                        # Use partition explainer for transformers
                        self.explainer = shap.Explainer(self.model, self.tokenizer)
                else:
                    # Use kernel explainer as fallback
                    self.explainer = shap.KernelExplainer(
                        self._predict_fn,
                        self.background_data or shap.sample(self.background_data, 100)
                    )
            
            elif self.explainer_type == 'kernel':
                self.explainer = shap.KernelExplainer(
                    self._predict_fn,
                    self.background_data
                )
            
            elif self.explainer_type == 'deep':
                if not isinstance(self.model, (nn.Module, torch.nn.Module)):
                    raise ValueError("Deep explainer requires PyTorch model")
                self.explainer = shap.DeepExplainer(self.model, self.background_data)
            
            elif self.explainer_type == 'partition':
                self.explainer = shap.Explainer(self.model, self.tokenizer)
            
            else:
                raise ValueError(f"Unknown explainer type: {self.explainer_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise
    
    def _predict_fn(self, texts):
        """
        Prediction function wrapper for SHAP
        """
        try:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(texts)
            elif hasattr(self.model, 'predict'):
                predictions = self.model.predict(texts)
                # Convert to probabilities if needed
                if len(predictions.shape) == 1:
                    predictions = np.column_stack([1 - predictions, predictions])
                return predictions
            else:
                return self.model(texts)
                
        except Exception as e:
            logger.error(f"Error in prediction function: {e}")
            raise
    
    def explain_instance(
        self,
        text: Union[str, List[str]],
        max_evals: Optional[int] = None,
        silent: bool = False,
        **kwargs
    ):
        """
        Explain a single text instance or batch
        
        Args:
            text: Input text(s) to explain
            max_evals: Maximum number of evaluations
            silent: Whether to suppress progress output
            
        Returns:
            SHAP values
        """
        try:
            max_evals = max_evals or self.max_evals
            
            if isinstance(text, str):
                text = [text]
            
            logger.info(f"Explaining {len(text)} text instance(s)")
            
            # Get SHAP values
            if hasattr(self.explainer, 'shap_values'):
                # For older SHAP versions
                shap_values = self.explainer.shap_values(
                    text,
                    max_evals=max_evals,
                    silent=silent,
                    **kwargs
                )
            else:
                # For newer SHAP versions
                shap_values = self.explainer(
                    text,
                    max_evals=max_evals,
                    silent=silent,
                    **kwargs
                )
            
            logger.info("Successfully generated SHAP values")
            return shap_values
            
        except Exception as e:
            logger.error(f"Failed to explain text instance: {e}")
            raise
    
    def explain_batch(self, texts: List[str], **kwargs):
        """
        Explain multiple text instances
        """
        return self.explain_instance(texts, **kwargs)
    
    def plot_explanation(
        self,
        shap_values,
        text: Optional[str] = None,
        plot_type: str = 'text',
        max_display: int = 20,
        **kwargs
    ):
        """
        Plot SHAP explanation
        
        Args:
            shap_values: SHAP values to plot
            text: Original text (for text plots)
            plot_type: Type of plot ('text', 'bar', 'waterfall', 'force')
            max_display: Maximum number of features to display
        """
        try:
            if plot_type == 'text':
                if hasattr(shap_values, 'data') and hasattr(shap_values, 'values'):
                    # New SHAP format
                    shap.plots.text(shap_values, **kwargs)
                else:
                    # Legacy format
                    if text is None:
                        raise ValueError("Text is required for text plot with legacy SHAP format")
                    shap.text_plot(shap_values, text, **kwargs)
            
            elif plot_type == 'bar':
                if hasattr(shap_values, 'values'):
                    shap.plots.bar(shap_values, max_display=max_display, **kwargs)
                else:
                    shap.summary_plot(
                        shap_values,
                        plot_type='bar',
                        max_display=max_display,
                        **kwargs
                    )
            
            elif plot_type == 'waterfall':
                if hasattr(shap_values, 'values'):
                    shap.plots.waterfall(shap_values[0], **kwargs)
                else:
                    shap.waterfall_plot(
                        shap_values.base_values[0],
                        shap_values.values[0],
                        shap_values.data[0] if hasattr(shap_values, 'data') else None,
                        **kwargs
                    )
            
            elif plot_type == 'force':
                if hasattr(shap_values, 'values'):
                    shap.plots.force(shap_values[0], **kwargs)
                else:
                    shap.force_plot(
                        shap_values.base_values[0],
                        shap_values.values[0],
                        shap_values.data[0] if hasattr(shap_values, 'data') else None,
                        **kwargs
                    )
            
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
                
        except Exception as e:
            logger.error(f"Failed to plot SHAP explanation: {e}")
            raise
    
    def get_explanation_dict(self, shap_values, text: str, class_idx: int = 0) -> Dict[str, Any]:
        """
        Convert SHAP values to dictionary format
        """
        try:
            if hasattr(shap_values, 'values'):
                # New SHAP format
                values = shap_values.values[0]
                data = shap_values.data[0] if hasattr(shap_values, 'data') else text.split()
                base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0
            else:
                # Legacy format
                values = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
                data = text.split()
                base_value = 0
            
            # Ensure values and data have same length
            min_len = min(len(values), len(data))
            values = values[:min_len]
            data = data[:min_len]
            
            return {
                'base_value': float(base_value),
                'prediction': float(base_value + np.sum(values)),
                'features': [
                    {
                        'token': str(token),
                        'shap_value': float(value),
                        'contribution': float(value) / (float(base_value + np.sum(values)) + 1e-8)
                    }
                    for token, value in zip(data, values)
                ],
                'total_shap_sum': float(np.sum(values))
            }
            
        except Exception as e:
            logger.error(f"Failed to convert SHAP values to dict: {e}")
            raise


class SHAPTabularExplainer(BaseSHAPExplainer):
    """
    SHAP explainer for tabular data models
    """
    
    def __init__(
        self,
        model,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        explainer_type: str = 'auto',
        **kwargs
    ):
        super().__init__(model, explainer_type, **kwargs)
        
        self.background_data = background_data
        self.feature_names = feature_names or [f'feature_{i}' for i in range(background_data.shape[1])]
        
        # Initialize appropriate SHAP explainer
        self._initialize_explainer()
        
        logger.info("Initialized SHAP tabular explainer")
    
    def _initialize_explainer(self):
        """
        Initialize the appropriate SHAP explainer
        """
        try:
            if self.explainer_type == 'auto':
                # Auto-detect explainer type
                if hasattr(self.model, 'predict_proba'):
                    # Try tree explainer first for tree-based models
                    try:
                        self.explainer = shap.TreeExplainer(self.model)
                        logger.info("Using TreeExplainer")
                    except:
                        # Fall back to explainer
                        self.explainer = shap.Explainer(self.model, self.background_data)
                        logger.info("Using general Explainer")
                else:
                    self.explainer = shap.Explainer(self.model, self.background_data)
            
            elif self.explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            
            elif self.explainer_type == 'kernel':
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    self.background_data
                )
            
            elif self.explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, self.background_data)
            
            else:
                raise ValueError(f"Unknown explainer type: {self.explainer_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise
    
    def explain_instance(self, instance: np.ndarray, **kwargs):
        """
        Explain a single tabular instance
        """
        try:
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)
            
            logger.info(f"Explaining tabular instance with shape {instance.shape}")
            
            # Get SHAP values
            if hasattr(self.explainer, 'shap_values'):
                shap_values = self.explainer.shap_values(instance, **kwargs)
            else:
                shap_values = self.explainer(instance, **kwargs)
            
            logger.info("Successfully generated SHAP values")
            return shap_values
            
        except Exception as e:
            logger.error(f"Failed to explain tabular instance: {e}")
            raise
    
    def explain_batch(self, instances: np.ndarray, **kwargs):
        """
        Explain multiple tabular instances
        """
        return self.explain_instance(instances, **kwargs)
    
    def plot_explanation(
        self,
        shap_values,
        instance: Optional[np.ndarray] = None,
        plot_type: str = 'summary',
        max_display: int = 20,
        **kwargs
    ):
        """
        Plot SHAP explanation for tabular data
        """
        try:
            if plot_type == 'summary':
                shap.summary_plot(
                    shap_values,
                    instance,
                    feature_names=self.feature_names,
                    max_display=max_display,
                    **kwargs
                )
            
            elif plot_type == 'bar':
                shap.summary_plot(
                    shap_values,
                    instance,
                    plot_type='bar',
                    feature_names=self.feature_names,
                    max_display=max_display,
                    **kwargs
                )
            
            elif plot_type == 'waterfall':
                if hasattr(shap_values, 'values'):
                    shap.plots.waterfall(shap_values[0], **kwargs)
                else:
                    shap.waterfall_plot(
                        shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0,
                        shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values.values[0],
                        instance[0] if instance is not None else None,
                        feature_names=self.feature_names,
                        **kwargs
                    )
            
            elif plot_type == 'force':
                if hasattr(shap_values, 'values'):
                    shap.plots.force(shap_values[0], **kwargs)
                else:
                    shap.force_plot(
                        shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0,
                        shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values.values[0],
                        instance[0] if instance is not None else None,
                        feature_names=self.feature_names,
                        **kwargs
                    )
            
            elif plot_type == 'dependence':
                feature_idx = kwargs.get('feature_idx', 0)
                shap.dependence_plot(
                    feature_idx,
                    shap_values,
                    instance,
                    feature_names=self.feature_names,
                    **kwargs
                )
            
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
                
        except Exception as e:
            logger.error(f"Failed to plot SHAP explanation: {e}")
            raise


class SHAPImageExplainer(BaseSHAPExplainer):
    """
    SHAP explainer for image models
    """
    
    def __init__(
        self,
        model,
        background_data: Optional[np.ndarray] = None,
        explainer_type: str = 'partition',
        **kwargs
    ):
        super().__init__(model, explainer_type, **kwargs)
        
        self.background_data = background_data
        
        # Initialize appropriate SHAP explainer
        self._initialize_explainer()
        
        logger.info("Initialized SHAP image explainer")
    
    def _initialize_explainer(self):
        """
        Initialize the appropriate SHAP explainer for images
        """
        try:
            if self.explainer_type == 'partition':
                self.explainer = shap.Explainer(self.model, self.background_data)
            
            elif self.explainer_type == 'deep':
                if not isinstance(self.model, (nn.Module, torch.nn.Module)):
                    raise ValueError("Deep explainer requires PyTorch model")
                self.explainer = shap.DeepExplainer(self.model, self.background_data)
            
            elif self.explainer_type == 'gradient':
                if not isinstance(self.model, (nn.Module, torch.nn.Module)):
                    raise ValueError("Gradient explainer requires PyTorch model")
                self.explainer = shap.GradientExplainer(self.model, self.background_data)
            
            else:
                raise ValueError(f"Unknown explainer type: {self.explainer_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize SHAP image explainer: {e}")
            raise
    
    def explain_instance(self, image: np.ndarray, **kwargs):
        """
        Explain a single image instance
        """
        try:
            if image.ndim == 3:
                image = image[np.newaxis, ...]
            
            logger.info(f"Explaining image with shape {image.shape}")
            
            # Get SHAP values
            if hasattr(self.explainer, 'shap_values'):
                shap_values = self.explainer.shap_values(image, **kwargs)
            else:
                shap_values = self.explainer(image, **kwargs)
            
            logger.info("Successfully generated SHAP values")
            return shap_values
            
        except Exception as e:
            logger.error(f"Failed to explain image instance: {e}")
            raise
    
    def plot_explanation(
        self,
        shap_values,
        image: np.ndarray,
        plot_type: str = 'image',
        **kwargs
    ):
        """
        Plot SHAP explanation for images
        """
        try:
            if plot_type == 'image':
                if hasattr(shap_values, 'values'):
                    shap.image_plot(shap_values.values, image, **kwargs)
                else:
                    shap.image_plot(shap_values, image, **kwargs)
            
            else:
                raise ValueError(f"Unknown plot type for images: {plot_type}")
                
        except Exception as e:
            logger.error(f"Failed to plot SHAP image explanation: {e}")
            raise


def create_shap_explainer(
    model,
    data_type: str,
    background_data=None,
    **kwargs
) -> BaseSHAPExplainer:
    """
    Factory function to create appropriate SHAP explainer
    
    Args:
        model: Model to explain
        data_type: Type of data ('text', 'tabular', 'image')
        background_data: Background data for explainer
        **kwargs: Additional arguments for explainer
        
    Returns:
        Appropriate SHAP explainer instance
    """
    if data_type.lower() == 'text':
        return SHAPTextExplainer(model, background_data=background_data, **kwargs)
    elif data_type.lower() == 'tabular':
        if background_data is None:
            raise ValueError("Background data is required for tabular SHAP explainer")
        return SHAPTabularExplainer(model, background_data, **kwargs)
    elif data_type.lower() == 'image':
        return SHAPImageExplainer(model, background_data=background_data, **kwargs)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


# Example usage and testing
if __name__ == "__main__":
    # Example with a simple tabular classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = SHAPTabularExplainer(model, X[:100])
    
    # Explain instance
    shap_values = explainer.explain_instance(X[0:1])
    
    # Plot explanation
    explainer.plot_explanation(shap_values, X[0:1], plot_type='bar')
    
    print("SHAP explainer test completed successfully!")
