"""
Captum (PyTorch Model Interpretability) Implementation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Captum imports
from captum.attr import (
    IntegratedGradients,
    Saliency,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    GuidedGradCam,
    LayerConductance,
    LayerGradientXActivation,
    LayerIntegratedGradients,
    NeuronConductance,
    NeuronGradient,
    Occlusion,
    ShapleyValueSampling,
    FeaturePermutation,
    LRP
)
from captum.attr._utils.visualization import visualize_image_attr, visualize_image_attr_multiple
from captum.attr._utils.common import _format_tensor_into_tuples

logger = logging.getLogger(__name__)


class CaptumExplainer:
    """
    Comprehensive Captum-based explainer for PyTorch models
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize attribution methods
        self.attribution_methods = self._initialize_attribution_methods()
        
        logger.info(f"Initialized Captum explainer with {len(self.attribution_methods)} methods")
    
    def _initialize_attribution_methods(self) -> Dict[str, Any]:
        """
        Initialize all available attribution methods
        """
        methods = {
            # Gradient-based methods
            'integrated_gradients': IntegratedGradients(self.model),
            'saliency': Saliency(self.model),
            'gradient_shap': GradientShap(self.model),
            'deeplift': DeepLift(self.model),
            'deeplift_shap': DeepLiftShap(self.model),
            
            # Perturbation-based methods
            'occlusion': Occlusion(self.model),
            'feature_permutation': FeaturePermutation(self.model),
            'shapley_value_sampling': ShapleyValueSampling(self.model),
            
            # Layer-wise methods (will be initialized when needed)
            'layer_conductance': None,
            'layer_gradient_x_activation': None,
            'layer_integrated_gradients': None,
            
            # Neuron-wise methods (will be initialized when needed)
            'neuron_conductance': None,
            'neuron_gradient': None,
        }
        
        # Add LRP if available
        try:
            methods['lrp'] = LRP(self.model)
        except Exception as e:
            logger.warning(f"LRP not available: {e}")
        
        return methods
    
    def explain_instance(
        self,
        inputs: torch.Tensor,
        target: Optional[Union[int, torch.Tensor]] = None,
        method: str = 'integrated_gradients',
        baselines: Optional[torch.Tensor] = None,
        additional_forward_args: Optional[Tuple] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Explain a single instance using specified attribution method
        
        Args:
            inputs: Input tensor to explain
            target: Target class index or tensor
            method: Attribution method to use
            baselines: Baseline inputs for comparison
            additional_forward_args: Additional arguments for forward pass
            **kwargs: Method-specific arguments
            
        Returns:
            Attribution tensor
        """
        try:
            # Ensure inputs are on correct device
            inputs = inputs.to(self.device)
            if baselines is not None:
                baselines = baselines.to(self.device)
            
            # Get attribution method
            attr_method = self.attribution_methods.get(method)
            if attr_method is None:
                raise ValueError(f"Unknown attribution method: {method}")
            
            logger.info(f"Computing {method} attributions for input shape {inputs.shape}")
            
            # Compute attributions based on method type
            if method in ['integrated_gradients', 'gradient_shap', 'deeplift', 'deeplift_shap']:
                # Methods that require baselines
                if baselines is None:
                    baselines = torch.zeros_like(inputs)
                
                attributions = attr_method.attribute(
                    inputs,
                    baselines=baselines,
                    target=target,
                    additional_forward_args=additional_forward_args,
                    **kwargs
                )
            
            elif method in ['saliency']:
                # Gradient-based methods
                attributions = attr_method.attribute(
                    inputs,
                    target=target,
                    additional_forward_args=additional_forward_args,
                    **kwargs
                )
            
            elif method in ['occlusion']:
                # Occlusion-based method
                sliding_window_shapes = kwargs.get('sliding_window_shapes', (1,) * (inputs.dim() - 1))
                strides = kwargs.get('strides', None)
                
                attributions = attr_method.attribute(
                    inputs,
                    sliding_window_shapes=sliding_window_shapes,
                    strides=strides,
                    target=target,
                    additional_forward_args=additional_forward_args,
                    **{k: v for k, v in kwargs.items() if k not in ['sliding_window_shapes', 'strides']}
                )
            
            elif method in ['feature_permutation']:
                # Feature permutation method
                attributions = attr_method.attribute(
                    inputs,
                    target=target,
                    additional_forward_args=additional_forward_args,
                    **kwargs
                )
            
            elif method in ['shapley_value_sampling']:
                # Shapley value sampling
                attributions = attr_method.attribute(
                    inputs,
                    target=target,
                    additional_forward_args=additional_forward_args,
                    **kwargs
                )
            
            elif method in ['lrp']:
                # Layer-wise relevance propagation
                attributions = attr_method.attribute(
                    inputs,
                    target=target,
                    additional_forward_args=additional_forward_args,
                    **kwargs
                )
            
            else:
                raise ValueError(f"Attribution method {method} not implemented")
            
            logger.info(f"Successfully computed {method} attributions")
            return attributions
            
        except Exception as e:
            logger.error(f"Failed to compute attributions with {method}: {e}")
            raise
    
    def explain_layer(
        self,
        inputs: torch.Tensor,
        layer: nn.Module,
        target: Optional[Union[int, torch.Tensor]] = None,
        method: str = 'layer_integrated_gradients',
        baselines: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Explain layer activations
        
        Args:
            inputs: Input tensor
            layer: Layer to analyze
            target: Target class
            method: Layer attribution method
            baselines: Baseline inputs
            **kwargs: Method-specific arguments
            
        Returns:
            Layer attribution tensor
        """
        try:
            inputs = inputs.to(self.device)
            if baselines is not None:
                baselines = baselines.to(self.device)
            
            logger.info(f"Computing {method} for layer {layer}")
            
            # Initialize layer method if needed
            if method == 'layer_integrated_gradients':
                attr_method = LayerIntegratedGradients(self.model, layer)
                if baselines is None:
                    baselines = torch.zeros_like(inputs)
                attributions = attr_method.attribute(
                    inputs,
                    baselines=baselines,
                    target=target,
                    **kwargs
                )
            
            elif method == 'layer_conductance':
                attr_method = LayerConductance(self.model, layer)
                if baselines is None:
                    baselines = torch.zeros_like(inputs)
                attributions = attr_method.attribute(
                    inputs,
                    baselines=baselines,
                    target=target,
                    **kwargs
                )
            
            elif method == 'layer_gradient_x_activation':
                attr_method = LayerGradientXActivation(self.model, layer)
                attributions = attr_method.attribute(
                    inputs,
                    target=target,
                    **kwargs
                )
            
            else:
                raise ValueError(f"Unknown layer method: {method}")
            
            logger.info(f"Successfully computed {method} for layer")
            return attributions
            
        except Exception as e:
            logger.error(f"Failed to compute layer attributions: {e}")
            raise
    
    def explain_neuron(
        self,
        inputs: torch.Tensor,
        layer: nn.Module,
        neuron_idx: Union[int, Tuple[int, ...]],
        method: str = 'neuron_gradient',
        **kwargs
    ) -> torch.Tensor:
        """
        Explain individual neuron activations
        
        Args:
            inputs: Input tensor
            layer: Layer containing the neuron
            neuron_idx: Index of the neuron to analyze
            method: Neuron attribution method
            **kwargs: Method-specific arguments
            
        Returns:
            Neuron attribution tensor
        """
        try:
            inputs = inputs.to(self.device)
            
            logger.info(f"Computing {method} for neuron {neuron_idx} in layer {layer}")
            
            if method == 'neuron_gradient':
                attr_method = NeuronGradient(self.model, layer)
                attributions = attr_method.attribute(
                    inputs,
                    neuron_idx,
                    **kwargs
                )
            
            elif method == 'neuron_conductance':
                attr_method = NeuronConductance(self.model, layer)
                baselines = kwargs.get('baselines', torch.zeros_like(inputs))
                attributions = attr_method.attribute(
                    inputs,
                    neuron_idx,
                    baselines=baselines,
                    **{k: v for k, v in kwargs.items() if k != 'baselines'}
                )
            
            else:
                raise ValueError(f"Unknown neuron method: {method}")
            
            logger.info(f"Successfully computed {method} for neuron")
            return attributions
            
        except Exception as e:
            logger.error(f"Failed to compute neuron attributions: {e}")
            raise
    
    def compare_methods(
        self,
        inputs: torch.Tensor,
        target: Optional[Union[int, torch.Tensor]] = None,
        methods: Optional[List[str]] = None,
        baselines: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compare multiple attribution methods on the same input
        
        Args:
            inputs: Input tensor to explain
            target: Target class
            methods: List of methods to compare
            baselines: Baseline inputs
            **kwargs: Method-specific arguments
            
        Returns:
            Dictionary mapping method names to attributions
        """
        if methods is None:
            methods = ['integrated_gradients', 'saliency', 'gradient_shap', 'deeplift']
        
        results = {}
        
        for method in methods:
            try:
                logger.info(f"Computing attributions with {method}")
                attributions = self.explain_instance(
                    inputs,
                    target=target,
                    method=method,
                    baselines=baselines,
                    **kwargs
                )
                results[method] = attributions
                
            except Exception as e:
                logger.warning(f"Failed to compute {method}: {e}")
                results[method] = None
        
        return results
    
    def visualize_attributions(
        self,
        attributions: torch.Tensor,
        original_image: torch.Tensor,
        method: str = 'heat_map',
        sign: str = 'absolute_value',
        plt_fig_axis: Optional[Tuple] = None,
        outlier_perc: int = 2,
        cmap: Optional[str] = None,
        alpha_overlay: float = 0.4,
        show_colorbar: bool = True,
        title: Optional[str] = None,
        **kwargs
    ):
        """
        Visualize attributions for image data
        
        Args:
            attributions: Attribution tensor
            original_image: Original image tensor
            method: Visualization method
            sign: How to handle attribution signs
            plt_fig_axis: Matplotlib figure and axis
            outlier_perc: Percentile for outlier removal
            cmap: Colormap for visualization
            alpha_overlay: Alpha for overlay
            show_colorbar: Whether to show colorbar
            title: Plot title
            **kwargs: Additional visualization arguments
        """
        try:
            # Convert tensors to numpy
            if isinstance(attributions, torch.Tensor):
                attributions = attributions.detach().cpu().numpy()
            if isinstance(original_image, torch.Tensor):
                original_image = original_image.detach().cpu().numpy()
            
            # Ensure correct dimensions
            if attributions.ndim == 4:
                attributions = attributions.squeeze(0)
            if original_image.ndim == 4:
                original_image = original_image.squeeze(0)
            
            # Transpose if needed (C, H, W) -> (H, W, C)
            if attributions.shape[0] in [1, 3]:
                attributions = np.transpose(attributions, (1, 2, 0))
            if original_image.shape[0] in [1, 3]:
                original_image = np.transpose(original_image, (1, 2, 0))
            
            # Visualize
            fig, axis = visualize_image_attr(
                attributions,
                original_image,
                method=method,
                sign=sign,
                plt_fig_axis=plt_fig_axis,
                outlier_perc=outlier_perc,
                cmap=cmap,
                alpha_overlay=alpha_overlay,
                show_colorbar=show_colorbar,
                title=title,
                **kwargs
            )
            
            return fig, axis
            
        except Exception as e:
            logger.error(f"Failed to visualize attributions: {e}")
            raise
    
    def visualize_multiple_attributions(
        self,
        attributions_dict: Dict[str, torch.Tensor],
        original_image: torch.Tensor,
        methods: List[str] = ['heat_map'],
        signs: List[str] = ['absolute_value'],
        titles: Optional[List[str]] = None,
        fig_size: Tuple[int, int] = (12, 8),
        **kwargs
    ):
        """
        Visualize multiple attribution methods side by side
        
        Args:
            attributions_dict: Dictionary of method names to attributions
            original_image: Original image tensor
            methods: Visualization methods for each attribution
            signs: Sign handling for each attribution
            titles: Titles for each subplot
            fig_size: Figure size
            **kwargs: Additional visualization arguments
        """
        try:
            # Prepare data
            attributions_list = []
            method_names = []
            
            for name, attr in attributions_dict.items():
                if attr is not None:
                    if isinstance(attr, torch.Tensor):
                        attr = attr.detach().cpu().numpy()
                    attributions_list.append(attr)
                    method_names.append(name)
            
            if not attributions_list:
                raise ValueError("No valid attributions to visualize")
            
            # Convert original image
            if isinstance(original_image, torch.Tensor):
                original_image = original_image.detach().cpu().numpy()
            
            # Ensure correct dimensions
            if original_image.ndim == 4:
                original_image = original_image.squeeze(0)
            if original_image.shape[0] in [1, 3]:
                original_image = np.transpose(original_image, (1, 2, 0))
            
            # Process attributions
            processed_attributions = []
            for attr in attributions_list:
                if attr.ndim == 4:
                    attr = attr.squeeze(0)
                if attr.shape[0] in [1, 3]:
                    attr = np.transpose(attr, (1, 2, 0))
                processed_attributions.append(attr)
            
            # Extend methods and signs if needed
            if len(methods) == 1:
                methods = methods * len(processed_attributions)
            if len(signs) == 1:
                signs = signs * len(processed_attributions)
            
            # Use default titles if not provided
            if titles is None:
                titles = method_names
            
            # Visualize
            fig, axes = visualize_image_attr_multiple(
                processed_attributions,
                original_image,
                methods=methods,
                signs=signs,
                titles=titles,
                fig_size=fig_size,
                **kwargs
            )
            
            return fig, axes
            
        except Exception as e:
            logger.error(f"Failed to visualize multiple attributions: {e}")
            raise
    
    def get_attribution_summary(
        self,
        attributions: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get summary statistics of attributions
        
        Args:
            attributions: Attribution tensor
            feature_names: Names of features
            top_k: Number of top features to return
            
        Returns:
            Summary dictionary
        """
        try:
            # Convert to numpy
            if isinstance(attributions, torch.Tensor):
                attributions = attributions.detach().cpu().numpy()
            
            # Flatten attributions
            flat_attr = attributions.flatten()
            
            # Get statistics
            summary = {
                'mean': float(np.mean(flat_attr)),
                'std': float(np.std(flat_attr)),
                'min': float(np.min(flat_attr)),
                'max': float(np.max(flat_attr)),
                'sum': float(np.sum(flat_attr)),
                'positive_sum': float(np.sum(flat_attr[flat_attr > 0])),
                'negative_sum': float(np.sum(flat_attr[flat_attr < 0])),
                'num_positive': int(np.sum(flat_attr > 0)),
                'num_negative': int(np.sum(flat_attr < 0)),
                'num_zero': int(np.sum(flat_attr == 0))
            }
            
            # Get top features by absolute value
            abs_attr = np.abs(flat_attr)
            top_indices = np.argsort(abs_attr)[-top_k:][::-1]
            
            if feature_names is not None and len(feature_names) == len(flat_attr):
                top_features = [
                    {
                        'feature': feature_names[idx],
                        'attribution': float(flat_attr[idx]),
                        'abs_attribution': float(abs_attr[idx]),
                        'rank': i + 1
                    }
                    for i, idx in enumerate(top_indices)
                ]
            else:
                top_features = [
                    {
                        'feature_idx': int(idx),
                        'attribution': float(flat_attr[idx]),
                        'abs_attribution': float(abs_attr[idx]),
                        'rank': i + 1
                    }
                    for i, idx in enumerate(top_indices)
                ]
            
            summary['top_features'] = top_features
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get attribution summary: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Example with a simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Create model and explainer
    model = SimpleCNN()
    explainer = CaptumExplainer(model)
    
    # Create dummy input
    inputs = torch.randn(1, 3, 32, 32)
    
    # Explain instance
    attributions = explainer.explain_instance(inputs, method='integrated_gradients')
    
    # Get summary
    summary = explainer.get_attribution_summary(attributions)
    
    print("Captum explainer test completed successfully!")
    print(f"Attribution summary: {summary}")
