"""
Classification Head Implementations for Feature-Based Fine-Tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class LinearClassifier(nn.Module):
    """
    Simple linear classifier for feature-based fine-tuning
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        task: str = 'classification',
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task = task
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Linear layer
        self.linear = nn.Linear(input_dim, num_classes, bias=bias)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize linear layer weights"""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through linear classifier
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Apply dropout
        features = self.dropout(features)
        
        # Linear transformation
        logits = self.linear(features)
        
        return logits


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier for feature-based fine-tuning
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[list] = None,
        task: str = 'classification',
        dropout: float = 0.1,
        activation: str = 'relu',
        batch_norm: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task = task
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'gelu':
                layers.append(nn.GELU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Create sequential model
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize MLP weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP classifier
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        return self.layers(features)


class AttentionClassifier(nn.Module):
    """
    Attention-based classifier for sequence features
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        task: str = 'classification',
        num_heads: int = 8,
        dropout: float = 0.1,
        use_self_attention: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task = task
        self.use_self_attention = use_self_attention
        
        # Self-attention layer
        if use_self_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Attention pooling
        self.attention_weights = nn.Linear(input_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.classifier = nn.Linear(input_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize attention classifier weights"""
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention classifier
        
        Args:
            features: Input features [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Handle 2D input (convert to sequence of length 1)
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        batch_size, seq_len, input_dim = features.shape
        
        # Apply self-attention if enabled
        if self.use_self_attention and seq_len > 1:
            attended_features, _ = self.self_attention(features, features, features)
            features = attended_features
        
        # Attention pooling
        attention_scores = self.attention_weights(features)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Weighted sum
        pooled_features = torch.sum(features * attention_weights, dim=1)  # [batch_size, input_dim]
        
        # Apply dropout
        pooled_features = self.dropout(pooled_features)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits


class ResidualClassifier(nn.Module):
    """
    Classifier with residual connections for deep feature transformation
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        task: str = 'classification',
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task = task
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        # Input projection if needed
        self.input_projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize residual classifier weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual classifier
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Input projection
        x = self.input_projection(features)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class ResidualBlock(nn.Module):
    """
    Residual block for deep feature transformation
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        residual = x
        x = self.layers(x)
        x = x + residual  # Residual connection
        x = self.layer_norm(x)
        return F.relu(x)


class AdaptiveClassifier(nn.Module):
    """
    Adaptive classifier that can handle variable input dimensions
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        task: str = 'classification',
        adaptation_method: str = 'linear',
        target_dim: Optional[int] = None,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task = task
        self.adaptation_method = adaptation_method
        
        if target_dim is None:
            target_dim = min(input_dim, 512)  # Default target dimension
        
        # Adaptation layer
        if adaptation_method == 'linear':
            self.adaptation = nn.Linear(input_dim, target_dim)
        elif adaptation_method == 'pca':
            # PCA-like adaptation (learned)
            self.adaptation = nn.Sequential(
                nn.Linear(input_dim, target_dim, bias=False),
                nn.LayerNorm(target_dim)
            )
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.classifier = nn.Linear(target_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize adaptive classifier weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adaptive classifier
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Adapt features
        adapted_features = self.adaptation(features)
        
        # Apply dropout
        adapted_features = self.dropout(adapted_features)
        
        # Classification
        logits = self.classifier(adapted_features)
        
        return logits


def create_classifier(
    classifier_type: str,
    input_dim: int,
    num_classes: int,
    task: str = 'classification',
    **config
) -> nn.Module:
    """
    Factory function to create classifiers
    
    Args:
        classifier_type: Type of classifier
        input_dim: Input feature dimension
        num_classes: Number of output classes
        task: Task type
        **config: Additional configuration
        
    Returns:
        Classifier module
    """
    classifier_map = {
        'linear': LinearClassifier,
        'mlp': MLPClassifier,
        'attention': AttentionClassifier,
        'residual': ResidualClassifier,
        'adaptive': AdaptiveClassifier
    }
    
    if classifier_type not in classifier_map:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    classifier_class = classifier_map[classifier_type]
    return classifier_class(
        input_dim=input_dim,
        num_classes=num_classes,
        task=task,
        **config
    )


# Example usage and testing
if __name__ == "__main__":
    # Test different classifiers
    input_dim = 768
    num_classes = 3
    batch_size = 4
    
    # Create dummy features
    features = torch.randn(batch_size, input_dim)
    
    # Test linear classifier
    linear_clf = LinearClassifier(input_dim, num_classes)
    linear_output = linear_clf(features)
    print(f"Linear classifier output shape: {linear_output.shape}")
    
    # Test MLP classifier
    mlp_clf = MLPClassifier(input_dim, num_classes, hidden_dims=[512, 256])
    mlp_output = mlp_clf(features)
    print(f"MLP classifier output shape: {mlp_output.shape}")
    
    # Test attention classifier
    attention_clf = AttentionClassifier(input_dim, num_classes)
    attention_output = attention_clf(features)
    print(f"Attention classifier output shape: {attention_output.shape}")
    
    # Test with sequence features
    seq_features = torch.randn(batch_size, 10, input_dim)
    attention_seq_output = attention_clf(seq_features)
    print(f"Attention classifier (sequence) output shape: {attention_seq_output.shape}")
    
    # Test residual classifier
    residual_clf = ResidualClassifier(input_dim, num_classes, num_layers=3)
    residual_output = residual_clf(features)
    print(f"Residual classifier output shape: {residual_output.shape}")
    
    # Test adaptive classifier
    adaptive_clf = AdaptiveClassifier(input_dim, num_classes, target_dim=256)
    adaptive_output = adaptive_clf(features)
    print(f"Adaptive classifier output shape: {adaptive_output.shape}")
    
    print("All classifier tests completed successfully!")
