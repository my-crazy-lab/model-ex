"""
Task-Specific Heads for Full Fine-Tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math


class ClassificationHead(nn.Module):
    """
    Classification head for sequence classification tasks
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
        activation: str = 'tanh',
        use_pooler: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_pooler = use_pooler
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        # Classification layer
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classification head weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head
        
        Args:
            pooled_output: Pooled output from backbone [batch_size, hidden_size]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Apply dropout
        output = self.dropout(pooled_output)
        
        # Apply activation
        output = self.activation(output)
        
        # Classification
        logits = self.classifier(output)
        
        return logits


class TokenClassificationHead(nn.Module):
    """
    Token classification head for NER and other token-level tasks
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        use_crf: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.use_crf = use_crf
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification layer
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # CRF layer for structured prediction
        if use_crf:
            try:
                from torchcrf import CRF
                self.crf = CRF(num_labels, batch_first=True)
            except ImportError:
                raise ImportError("torchcrf is required for CRF. Install with: pip install pytorch-crf")
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize token classification head weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through token classification head
        
        Args:
            sequence_output: Sequence output from backbone [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for CRF training [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, num_labels] or CRF predictions
        """
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Classification
        logits = self.classifier(sequence_output)
        
        if self.use_crf:
            if labels is not None:
                # Training: return negative log-likelihood
                mask = attention_mask.bool() if attention_mask is not None else None
                return -self.crf(logits, labels, mask=mask, reduction='mean')
            else:
                # Inference: return best path
                mask = attention_mask.bool() if attention_mask is not None else None
                return self.crf.decode(logits, mask=mask)
        
        return logits


class QuestionAnsweringHead(nn.Module):
    """
    Question answering head for extractive QA tasks
    """
    
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # QA outputs (start and end positions)
        self.qa_outputs = nn.Linear(hidden_size, 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize QA head weights"""
        nn.init.xavier_uniform_(self.qa_outputs.weight)
        if self.qa_outputs.bias is not None:
            nn.init.zeros_(self.qa_outputs.bias)
    
    def forward(self, sequence_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through QA head
        
        Args:
            sequence_output: Sequence output from backbone [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple of (start_logits, end_logits) each [batch_size, seq_len]
        """
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Get start and end logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits


class GenerationHead(nn.Module):
    """
    Generation head for text generation tasks
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        dropout: float = 0.1,
        tie_weights: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize generation head weights"""
        nn.init.xavier_uniform_(self.lm_head.weight)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generation head
        
        Args:
            hidden_states: Hidden states from backbone [batch_size, seq_len, hidden_size]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Language modeling
        logits = self.lm_head(hidden_states)
        
        return logits


class MultiTaskHead(nn.Module):
    """
    Multi-task head for handling multiple tasks simultaneously
    """
    
    def __init__(
        self,
        hidden_size: int,
        task_configs: Dict[str, Dict[str, Any]],
        shared_layers: int = 1,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.task_configs = task_configs
        self.shared_layers = shared_layers
        
        # Shared layers
        shared_modules = []
        for i in range(shared_layers):
            shared_modules.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.shared_layers = nn.Sequential(*shared_modules) if shared_modules else nn.Identity()
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            task_type = config['type']
            
            if task_type == 'classification':
                head = ClassificationHead(
                    hidden_size=hidden_size,
                    num_classes=config['num_classes'],
                    dropout=dropout
                )
            elif task_type == 'token_classification':
                head = TokenClassificationHead(
                    hidden_size=hidden_size,
                    num_labels=config['num_labels'],
                    dropout=dropout
                )
            elif task_type == 'question_answering':
                head = QuestionAnsweringHead(
                    hidden_size=hidden_size,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            self.task_heads[task_name] = head
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_name: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through multi-task head
        
        Args:
            hidden_states: Hidden states from backbone
            task_name: Name of the task to execute
            
        Returns:
            Task-specific outputs
        """
        # Shared processing
        shared_output = self.shared_layers(hidden_states)
        
        # Task-specific processing
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")
        
        return self.task_heads[task_name](shared_output, **kwargs)


class AttentionPoolingHead(nn.Module):
    """
    Attention-based pooling head for sequence classification
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_attention_heads = num_attention_heads
        
        # Multi-head attention for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize attention pooling head weights"""
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through attention pooling head
        
        Args:
            sequence_output: Sequence output [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        batch_size = sequence_output.size(0)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)
        
        # Attention pooling
        pooled_output, attention_weights = self.attention(
            query=query,
            key=sequence_output,
            value=sequence_output,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Squeeze sequence dimension
        pooled_output = pooled_output.squeeze(1)
        
        # Classification
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits


class HierarchicalHead(nn.Module):
    """
    Hierarchical classification head for multi-level classification
    """
    
    def __init__(
        self,
        hidden_size: int,
        hierarchy_config: Dict[str, Any],
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.hierarchy_config = hierarchy_config
        
        # Create classifiers for each level
        self.level_classifiers = nn.ModuleDict()
        
        for level_name, level_config in hierarchy_config.items():
            self.level_classifiers[level_name] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, level_config['num_classes'])
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize hierarchical head weights"""
        for classifier in self.level_classifiers.values():
            for module in classifier:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, pooled_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical head
        
        Args:
            pooled_output: Pooled output [batch_size, hidden_size]
            
        Returns:
            Dictionary of logits for each level
        """
        outputs = {}
        
        for level_name, classifier in self.level_classifiers.items():
            outputs[level_name] = classifier(pooled_output)
        
        return outputs


# Factory function for creating task heads
def create_task_head(
    task_type: str,
    hidden_size: int,
    **config
) -> nn.Module:
    """
    Factory function to create task-specific heads
    
    Args:
        task_type: Type of task head
        hidden_size: Hidden size from backbone
        **config: Additional configuration
        
    Returns:
        Task head module
    """
    head_map = {
        'classification': ClassificationHead,
        'token_classification': TokenClassificationHead,
        'question_answering': QuestionAnsweringHead,
        'generation': GenerationHead,
        'multi_task': MultiTaskHead,
        'attention_pooling': AttentionPoolingHead,
        'hierarchical': HierarchicalHead
    }
    
    if task_type not in head_map:
        raise ValueError(f"Unknown task type: {task_type}")
    
    head_class = head_map[task_type]
    return head_class(hidden_size=hidden_size, **config)


# Example usage and testing
if __name__ == "__main__":
    # Test classification head
    hidden_size = 768
    batch_size = 4
    seq_len = 128
    
    # Test classification head
    cls_head = ClassificationHead(hidden_size, num_classes=3)
    pooled_output = torch.randn(batch_size, hidden_size)
    cls_logits = cls_head(pooled_output)
    print(f"Classification logits shape: {cls_logits.shape}")
    
    # Test token classification head
    token_head = TokenClassificationHead(hidden_size, num_labels=9)
    sequence_output = torch.randn(batch_size, seq_len, hidden_size)
    token_logits = token_head(sequence_output)
    print(f"Token classification logits shape: {token_logits.shape}")
    
    # Test QA head
    qa_head = QuestionAnsweringHead(hidden_size)
    start_logits, end_logits = qa_head(sequence_output)
    print(f"QA start logits shape: {start_logits.shape}")
    print(f"QA end logits shape: {end_logits.shape}")
    
    # Test generation head
    gen_head = GenerationHead(hidden_size, vocab_size=30522)
    gen_logits = gen_head(sequence_output)
    print(f"Generation logits shape: {gen_logits.shape}")
    
    # Test attention pooling head
    att_head = AttentionPoolingHead(hidden_size, num_classes=3)
    attention_mask = torch.ones(batch_size, seq_len)
    att_logits = att_head(sequence_output, attention_mask)
    print(f"Attention pooling logits shape: {att_logits.shape}")
    
    print("All task heads test completed successfully!")
