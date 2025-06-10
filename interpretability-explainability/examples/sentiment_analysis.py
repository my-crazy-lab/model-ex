"""
Comprehensive Sentiment Analysis Interpretability Example

This example demonstrates how to use multiple interpretability methods
to understand how a sentiment analysis model makes decisions.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from explainers.lime_explainer import LIMETextExplainer
from explainers.shap_explainer import SHAPTextExplainer
from explainers.captum_explainer import CaptumExplainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalysisModel:
    """
    Wrapper for sentiment analysis model to work with explainers
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # Labels mapping
        self.labels = ['negative', 'neutral', 'positive']
        
        logger.info(f"Loaded sentiment model: {model_name}")
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict sentiment probabilities for texts
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        return probabilities.numpy()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Alias for predict method (for sklearn compatibility)
        """
        return self.predict(texts)
    
    def get_prediction_with_confidence(self, text: str) -> Dict[str, Any]:
        """
        Get prediction with confidence scores
        """
        probs = self.predict([text])[0]
        predicted_class = np.argmax(probs)
        
        return {
            'text': text,
            'predicted_class': self.labels[predicted_class],
            'confidence': float(probs[predicted_class]),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(self.labels, probs)
            }
        }


class SentimentExplainer:
    """
    Comprehensive sentiment analysis explainer using multiple methods
    """
    
    def __init__(self, model: SentimentAnalysisModel):
        self.model = model
        
        # Initialize explainers
        self.lime_explainer = LIMETextExplainer(
            model=model,
            class_names=model.labels,
            mode='classification'
        )
        
        self.shap_explainer = SHAPTextExplainer(
            model=model,
            tokenizer=model.tokenizer,
            explainer_type='auto'
        )
        
        logger.info("Initialized sentiment explainers")
    
    def explain_prediction(
        self,
        text: str,
        methods: List[str] = ['lime', 'shap'],
        num_samples: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Explain sentiment prediction using multiple methods
        
        Args:
            text: Input text to explain
            methods: List of explanation methods to use
            num_samples: Number of samples for explanation
            **kwargs: Additional arguments for explainers
            
        Returns:
            Dictionary containing explanations from all methods
        """
        logger.info(f"Explaining prediction for text: '{text[:50]}...'")
        
        # Get model prediction
        prediction = self.model.get_prediction_with_confidence(text)
        
        results = {
            'text': text,
            'prediction': prediction,
            'explanations': {}
        }
        
        # LIME explanation
        if 'lime' in methods:
            try:
                logger.info("Computing LIME explanation...")
                lime_explanation = self.lime_explainer.explain_instance(
                    text,
                    num_samples=num_samples,
                    num_features=20
                )
                
                # Convert to dictionary format
                lime_dict = self.lime_explainer.get_explanation_dict(
                    lime_explanation,
                    label=np.argmax(prediction['probabilities'].values())
                )
                
                results['explanations']['lime'] = {
                    'explanation_object': lime_explanation,
                    'explanation_dict': lime_dict
                }
                
                logger.info("LIME explanation completed")
                
            except Exception as e:
                logger.error(f"LIME explanation failed: {e}")
                results['explanations']['lime'] = {'error': str(e)}
        
        # SHAP explanation
        if 'shap' in methods:
            try:
                logger.info("Computing SHAP explanation...")
                shap_values = self.shap_explainer.explain_instance(
                    text,
                    max_evals=num_samples
                )
                
                # Convert to dictionary format
                shap_dict = self.shap_explainer.get_explanation_dict(
                    shap_values,
                    text,
                    class_idx=np.argmax(prediction['probabilities'].values())
                )
                
                results['explanations']['shap'] = {
                    'shap_values': shap_values,
                    'explanation_dict': shap_dict
                }
                
                logger.info("SHAP explanation completed")
                
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")
                results['explanations']['shap'] = {'error': str(e)}
        
        return results
    
    def visualize_explanations(
        self,
        explanation_results: Dict[str, Any],
        save_path: str = None,
        show_plots: bool = True
    ):
        """
        Visualize explanations from multiple methods
        """
        text = explanation_results['text']
        prediction = explanation_results['prediction']
        explanations = explanation_results['explanations']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Sentiment Analysis Explanations\nPredicted: {prediction['predicted_class']} (confidence: {prediction['confidence']:.3f})", fontsize=14)
        
        # Plot 1: Prediction probabilities
        ax1 = axes[0, 0]
        labels = list(prediction['probabilities'].keys())
        probs = list(prediction['probabilities'].values())
        colors = ['red', 'gray', 'green']
        
        bars = ax1.bar(labels, probs, color=colors, alpha=0.7)
        ax1.set_title('Prediction Probabilities')
        ax1.set_ylabel('Probability')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        # Plot 2: LIME explanation
        ax2 = axes[0, 1]
        if 'lime' in explanations and 'explanation_dict' in explanations['lime']:
            lime_data = explanations['lime']['explanation_dict']
            features = [f['feature'] for f in lime_data['features'][:10]]
            importances = [f['importance'] for f in lime_data['features'][:10]]
            
            colors = ['red' if imp < 0 else 'green' for imp in importances]
            bars = ax2.barh(range(len(features)), importances, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels(features)
            ax2.set_title('LIME Feature Importance')
            ax2.set_xlabel('Importance')
            
            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                ax2.text(bar.get_width() + 0.01 if imp > 0 else bar.get_width() - 0.01,
                        bar.get_y() + bar.get_height()/2, f'{imp:.3f}',
                        ha='left' if imp > 0 else 'right', va='center')
        else:
            ax2.text(0.5, 0.5, 'LIME explanation\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('LIME Feature Importance')
        
        # Plot 3: SHAP explanation
        ax3 = axes[1, 0]
        if 'shap' in explanations and 'explanation_dict' in explanations['shap']:
            shap_data = explanations['shap']['explanation_dict']
            features = [f['token'] for f in shap_data['features'][:10]]
            shap_values = [f['shap_value'] for f in shap_data['features'][:10]]
            
            colors = ['red' if val < 0 else 'green' for val in shap_values]
            bars = ax3.barh(range(len(features)), shap_values, color=colors, alpha=0.7)
            ax3.set_yticks(range(len(features)))
            ax3.set_yticklabels(features)
            ax3.set_title('SHAP Values')
            ax3.set_xlabel('SHAP Value')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, shap_values)):
                ax3.text(bar.get_width() + 0.01 if val > 0 else bar.get_width() - 0.01,
                        bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                        ha='left' if val > 0 else 'right', va='center')
        else:
            ax3.text(0.5, 0.5, 'SHAP explanation\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('SHAP Values')
        
        # Plot 4: Text with highlighting
        ax4 = axes[1, 1]
        ax4.text(0.05, 0.95, f"Original Text:\n{text}", 
                transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', wrap=True)
        ax4.set_title('Input Text')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        if show_plots:
            plt.show()
        
        return fig
    
    def compare_explanations(
        self,
        text: str,
        methods: List[str] = ['lime', 'shap'],
        num_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Compare explanations from different methods
        """
        results = self.explain_prediction(text, methods, num_samples)
        
        # Extract feature importances
        comparison_data = []
        
        # Get tokens from text
        tokens = text.split()
        
        for method in methods:
            if method in results['explanations'] and 'explanation_dict' in results['explanations'][method]:
                explanation = results['explanations'][method]['explanation_dict']
                
                if method == 'lime':
                    features = explanation['features']
                    for feature in features:
                        comparison_data.append({
                            'method': 'LIME',
                            'token': feature['feature'],
                            'importance': feature['importance']
                        })
                
                elif method == 'shap':
                    features = explanation['features']
                    for feature in features:
                        comparison_data.append({
                            'method': 'SHAP',
                            'token': feature['token'],
                            'importance': feature['shap_value']
                        })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            # Pivot for comparison
            comparison_df = df.pivot(index='token', columns='method', values='importance').fillna(0)
            
            # Add correlation if both methods available
            if len(methods) >= 2 and all(method.upper() in comparison_df.columns for method in methods[:2]):
                correlation = comparison_df.iloc[:, 0].corr(comparison_df.iloc[:, 1])
                logger.info(f"Correlation between {methods[0]} and {methods[1]}: {correlation:.3f}")
            
            return comparison_df
        
        return pd.DataFrame()


def main():
    """
    Main function to demonstrate sentiment analysis interpretability
    """
    logger.info("Starting sentiment analysis interpretability example")
    
    # Initialize model and explainer
    model = SentimentAnalysisModel()
    explainer = SentimentExplainer(model)
    
    # Example texts
    example_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "The service was terrible and the food was cold. Very disappointed.",
        "It's an okay product, nothing special but does the job.",
        "I hate this so much! Worst experience ever!",
        "Amazing quality and great customer service. Highly recommended!"
    ]
    
    # Analyze each text
    for i, text in enumerate(example_texts):
        print(f"\n{'='*60}")
        print(f"Example {i+1}: {text}")
        print('='*60)
        
        # Get explanation
        results = explainer.explain_prediction(
            text,
            methods=['lime', 'shap'],
            num_samples=500
        )
        
        # Print prediction
        prediction = results['prediction']
        print(f"Predicted sentiment: {prediction['predicted_class']}")
        print(f"Confidence: {prediction['confidence']:.3f}")
        print(f"All probabilities: {prediction['probabilities']}")
        
        # Print LIME explanation
        if 'lime' in results['explanations'] and 'explanation_dict' in results['explanations']['lime']:
            lime_data = results['explanations']['lime']['explanation_dict']
            print(f"\nLIME Top Features:")
            for feature in lime_data['features'][:5]:
                print(f"  {feature['feature']}: {feature['importance']:.3f}")
        
        # Print SHAP explanation
        if 'shap' in results['explanations'] and 'explanation_dict' in results['explanations']['shap']:
            shap_data = results['explanations']['shap']['explanation_dict']
            print(f"\nSHAP Top Features:")
            for feature in shap_data['features'][:5]:
                print(f"  {feature['token']}: {feature['shap_value']:.3f}")
        
        # Visualize explanations
        fig = explainer.visualize_explanations(
            results,
            save_path=f"sentiment_explanation_{i+1}.png",
            show_plots=False
        )
        plt.close(fig)
        
        # Compare methods
        comparison_df = explainer.compare_explanations(text)
        if not comparison_df.empty:
            print(f"\nMethod Comparison:")
            print(comparison_df.head())
    
    logger.info("Sentiment analysis interpretability example completed")


if __name__ == "__main__":
    main()
