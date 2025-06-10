"""
LIME (Local Interpretable Model-agnostic Explanations) Implementation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from lime import lime_text, lime_image, lime_tabular
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

logger = logging.getLogger(__name__)


class BaseLIMEExplainer:
    """
    Base class for LIME explainers
    """
    
    def __init__(self, model, mode: str = 'classification', **kwargs):
        self.model = model
        self.mode = mode
        self.explainer = None
        self.kwargs = kwargs
        
        logger.info(f"Initialized LIME explainer in {mode} mode")
    
    def _predict_fn(self, instances):
        """
        Prediction function wrapper for LIME
        """
        raise NotImplementedError("Subclasses must implement _predict_fn")
    
    def explain_instance(self, instance, **kwargs):
        """
        Explain a single instance
        """
        raise NotImplementedError("Subclasses must implement explain_instance")
    
    def explain_batch(self, instances: List, **kwargs) -> List:
        """
        Explain multiple instances
        """
        explanations = []
        for instance in instances:
            try:
                explanation = self.explain_instance(instance, **kwargs)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Failed to explain instance: {e}")
                explanations.append(None)
        
        return explanations


class LIMETextExplainer(BaseLIMEExplainer):
    """
    LIME explainer for text classification models
    """
    
    def __init__(
        self,
        model,
        mode: str = 'classification',
        class_names: Optional[List[str]] = None,
        feature_selection: str = 'auto',
        split_expression: str = r'\W+',
        bow: bool = True,
        mask_string: Optional[str] = None,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(model, mode, **kwargs)
        
        self.class_names = class_names or ['negative', 'positive']
        self.feature_selection = feature_selection
        self.split_expression = split_expression
        self.bow = bow
        self.mask_string = mask_string
        self.random_state = random_state
        
        # Initialize LIME text explainer
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            feature_selection=self.feature_selection,
            split_expression=self.split_expression,
            bow=self.bow,
            mask_string=self.mask_string,
            random_state=self.random_state,
            mode=self.mode
        )
        
        logger.info("Initialized LIME text explainer")
    
    def _predict_fn(self, texts: List[str]) -> np.ndarray:
        """
        Prediction function for text inputs
        """
        try:
            # Handle different model types
            if hasattr(self.model, 'predict_proba'):
                # Scikit-learn style
                predictions = self.model.predict_proba(texts)
            elif hasattr(self.model, 'predict'):
                # Custom model with predict method
                predictions = self.model.predict(texts)
            else:
                # Assume it's a callable
                predictions = self.model(texts)
            
            # Ensure predictions are in the right format
            if isinstance(predictions, list):
                predictions = np.array(predictions)
            
            # Handle single class predictions
            if len(predictions.shape) == 1:
                # Convert to probabilities for binary classification
                predictions = np.column_stack([1 - predictions, predictions])
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction function: {e}")
            raise
    
    def explain_instance(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 5000,
        distance_metric: str = 'cosine',
        model_regressor: Optional[Any] = None,
        labels: Optional[List[int]] = None,
        top_labels: Optional[int] = None,
        **kwargs
    ):
        """
        Explain a single text instance
        
        Args:
            text: Input text to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples to generate for explanation
            distance_metric: Distance metric for weighting samples
            model_regressor: Regressor to use for local approximation
            labels: Labels to explain (if None, explains top prediction)
            top_labels: Number of top labels to explain
            
        Returns:
            LIME explanation object
        """
        try:
            logger.info(f"Explaining text instance with {num_samples} samples")
            
            explanation = self.explainer.explain_instance(
                text,
                self._predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                distance_metric=distance_metric,
                model_regressor=model_regressor,
                labels=labels,
                top_labels=top_labels,
                **kwargs
            )
            
            logger.info("Successfully generated LIME explanation")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain text instance: {e}")
            raise
    
    def visualize_explanation(
        self,
        explanation,
        label: Optional[int] = None,
        show_table: bool = True,
        show_in_notebook: bool = True
    ):
        """
        Visualize LIME explanation
        """
        try:
            if label is None:
                # Use the top predicted label
                label = explanation.available_labels[0]
            
            if show_in_notebook:
                # Show HTML visualization in notebook
                explanation.show_in_notebook(labels=[label], text=show_table)
            else:
                # Create matplotlib visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get feature importance
                exp_list = explanation.as_list(label=label)
                features, importances = zip(*exp_list)
                
                # Create bar plot
                colors = ['red' if imp < 0 else 'green' for imp in importances]
                bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
                
                # Customize plot
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'LIME Explanation for Label: {self.class_names[label]}')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, imp) in enumerate(zip(bars, importances)):
                    ax.text(
                        bar.get_width() + 0.01 if imp > 0 else bar.get_width() - 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{imp:.3f}',
                        ha='left' if imp > 0 else 'right',
                        va='center'
                    )
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to visualize explanation: {e}")
            raise
    
    def get_explanation_dict(self, explanation, label: Optional[int] = None) -> Dict[str, Any]:
        """
        Convert LIME explanation to dictionary format
        """
        if label is None:
            label = explanation.available_labels[0]
        
        exp_list = explanation.as_list(label=label)
        
        return {
            'label': label,
            'class_name': self.class_names[label] if label < len(self.class_names) else f'class_{label}',
            'prediction_probability': explanation.predict_proba[label],
            'features': [{'feature': feature, 'importance': importance} for feature, importance in exp_list],
            'intercept': explanation.intercept[label] if hasattr(explanation, 'intercept') else None,
            'score': explanation.score if hasattr(explanation, 'score') else None,
            'local_pred': explanation.local_pred[label] if hasattr(explanation, 'local_pred') else None
        }


class LIMEImageExplainer(BaseLIMEExplainer):
    """
    LIME explainer for image classification models
    """
    
    def __init__(
        self,
        model,
        mode: str = 'classification',
        feature_selection: str = 'auto',
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(model, mode, **kwargs)
        
        self.feature_selection = feature_selection
        self.random_state = random_state
        
        # Initialize LIME image explainer
        self.explainer = LimeImageExplainer(
            feature_selection=self.feature_selection,
            random_state=self.random_state,
            mode=self.mode
        )
        
        logger.info("Initialized LIME image explainer")
    
    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function for image inputs
        """
        try:
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(images)
            elif hasattr(self.model, 'predict'):
                predictions = self.model.predict(images)
            else:
                predictions = self.model(images)
            
            if isinstance(predictions, list):
                predictions = np.array(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in image prediction function: {e}")
            raise
    
    def explain_instance(
        self,
        image: np.ndarray,
        num_features: int = 100000,
        num_samples: int = 1000,
        batch_size: int = 10,
        segmentation_fn: Optional[Callable] = None,
        distance_metric: str = 'cosine',
        model_regressor: Optional[Any] = None,
        labels: Optional[List[int]] = None,
        top_labels: Optional[int] = None,
        hide_color: Optional[int] = None,
        **kwargs
    ):
        """
        Explain a single image instance
        """
        try:
            logger.info(f"Explaining image instance with {num_samples} samples")
            
            explanation = self.explainer.explain_instance(
                image,
                self._predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                batch_size=batch_size,
                segmentation_fn=segmentation_fn,
                distance_metric=distance_metric,
                model_regressor=model_regressor,
                labels=labels,
                top_labels=top_labels,
                hide_color=hide_color,
                **kwargs
            )
            
            logger.info("Successfully generated LIME image explanation")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain image instance: {e}")
            raise
    
    def visualize_explanation(
        self,
        explanation,
        label: Optional[int] = None,
        positive_only: bool = True,
        negative_only: bool = False,
        hide_rest: bool = False,
        num_features: int = 5,
        min_weight: float = 0.0
    ):
        """
        Visualize LIME image explanation
        """
        try:
            if label is None:
                label = explanation.top_labels[0]
            
            # Get image and mask
            temp, mask = explanation.get_image_and_mask(
                label,
                positive_only=positive_only,
                negative_only=negative_only,
                hide_rest=hide_rest,
                num_features=num_features,
                min_weight=min_weight
            )
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image
            axes[0].imshow(explanation.image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Explanation overlay
            axes[1].imshow(temp)
            axes[1].set_title(f'LIME Explanation (Label: {label})')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to visualize image explanation: {e}")
            raise


class LIMETabularExplainer(BaseLIMEExplainer):
    """
    LIME explainer for tabular data models
    """
    
    def __init__(
        self,
        model,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        mode: str = 'classification',
        categorical_features: Optional[List[int]] = None,
        categorical_names: Optional[Dict[int, List[str]]] = None,
        kernel_width: Optional[float] = None,
        discretize_continuous: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(model, mode, **kwargs)
        
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_features = categorical_features or []
        self.categorical_names = categorical_names or {}
        self.kernel_width = kernel_width
        self.discretize_continuous = discretize_continuous
        self.random_state = random_state
        
        # Initialize LIME tabular explainer
        self.explainer = LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=class_names,
            categorical_features=categorical_features,
            categorical_names=categorical_names,
            kernel_width=kernel_width,
            discretize_continuous=discretize_continuous,
            random_state=random_state,
            mode=mode
        )
        
        logger.info("Initialized LIME tabular explainer")
    
    def _predict_fn(self, instances: np.ndarray) -> np.ndarray:
        """
        Prediction function for tabular inputs
        """
        try:
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(instances)
            elif hasattr(self.model, 'predict'):
                predictions = self.model.predict(instances)
            else:
                predictions = self.model(instances)
            
            if isinstance(predictions, list):
                predictions = np.array(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in tabular prediction function: {e}")
            raise
    
    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        distance_metric: str = 'euclidean',
        model_regressor: Optional[Any] = None,
        labels: Optional[List[int]] = None,
        top_labels: Optional[int] = None,
        **kwargs
    ):
        """
        Explain a single tabular instance
        """
        try:
            logger.info(f"Explaining tabular instance with {num_samples} samples")
            
            explanation = self.explainer.explain_instance(
                instance,
                self._predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                distance_metric=distance_metric,
                model_regressor=model_regressor,
                labels=labels,
                top_labels=top_labels,
                **kwargs
            )
            
            logger.info("Successfully generated LIME tabular explanation")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain tabular instance: {e}")
            raise
    
    def visualize_explanation(
        self,
        explanation,
        label: Optional[int] = None,
        show_table: bool = True
    ):
        """
        Visualize LIME tabular explanation
        """
        try:
            if label is None:
                label = explanation.available_labels[0]
            
            # Get feature importance
            exp_list = explanation.as_list(label=label)
            features, importances = zip(*exp_list)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar plot
            colors = ['red' if imp < 0 else 'green' for imp in importances]
            bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
            
            # Customize plot
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'LIME Explanation for Label: {label}')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                ax.text(
                    bar.get_width() + 0.01 if imp > 0 else bar.get_width() - 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}',
                    ha='left' if imp > 0 else 'right',
                    va='center'
                )
            
            plt.tight_layout()
            plt.show()
            
            if show_table:
                # Show explanation as table
                df = pd.DataFrame(exp_list, columns=['Feature', 'Importance'])
                df = df.sort_values('Importance', key=abs, ascending=False)
                print("\nFeature Importance Table:")
                print(df.to_string(index=False))
                
        except Exception as e:
            logger.error(f"Failed to visualize tabular explanation: {e}")
            raise


def create_lime_explainer(
    model,
    data_type: str,
    **kwargs
) -> BaseLIMEExplainer:
    """
    Factory function to create appropriate LIME explainer
    
    Args:
        model: Model to explain
        data_type: Type of data ('text', 'image', 'tabular')
        **kwargs: Additional arguments for explainer
        
    Returns:
        Appropriate LIME explainer instance
    """
    if data_type.lower() == 'text':
        return LIMETextExplainer(model, **kwargs)
    elif data_type.lower() == 'image':
        return LIMEImageExplainer(model, **kwargs)
    elif data_type.lower() == 'tabular':
        return LIMETabularExplainer(model, **kwargs)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


# Example usage and testing
if __name__ == "__main__":
    # Example with a simple text classifier
    class DummyTextClassifier:
        def predict_proba(self, texts):
            # Dummy implementation
            return np.random.rand(len(texts), 2)
    
    # Create explainer
    model = DummyTextClassifier()
    explainer = LIMETextExplainer(model)
    
    # Explain instance
    text = "This is a great movie!"
    explanation = explainer.explain_instance(text, num_samples=100)
    
    # Visualize
    explainer.visualize_explanation(explanation, show_in_notebook=False)
    
    print("LIME explainer test completed successfully!")
