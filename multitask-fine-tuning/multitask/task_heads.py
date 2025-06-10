"""
Task-specific heads for multitask learning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Classification head for text classification tasks"""
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout_rate: float = 0.1,
        use_pooler: bool = True
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.use_pooler = use_pooler
        
        if use_pooler:
            # Use [CLS] token representation
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(hidden_size, num_labels)
        else:
            # Use mean pooling
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_pooler:
            # Use [CLS] token (first token)
            pooled_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        else:
            # Mean pooling over sequence length
            if attention_mask is not None:
                # Mask out padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = hidden_states.mean(dim=1)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class QuestionAnsweringHead(nn.Module):
    """Question answering head for extractive QA tasks"""
    
    def __init__(
        self,
        hidden_size: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start and end positions
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        sequence_output = self.dropout(hidden_states)
        logits = self.qa_outputs(sequence_output)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch_size, seq_len]
        end_logits = end_logits.squeeze(-1)      # [batch_size, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Set logits for padding tokens to large negative value
            start_logits = start_logits + (1 - attention_mask) * -10000.0
            end_logits = end_logits + (1 - attention_mask) * -10000.0
        
        return {
            "start_logits": start_logits,
            "end_logits": end_logits
        }


class TokenClassificationHead(nn.Module):
    """Token classification head for NER, POS tagging, etc."""
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        sequence_output = self.dropout(hidden_states)
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        return logits


class GenerationHead(nn.Module):
    """Generation head for text generation tasks (summarization, etc.)"""
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout_rate)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        return logits


class TaskEmbedding(nn.Module):
    """Task embedding layer for task-aware representations"""
    
    def __init__(
        self,
        num_tasks: int,
        embedding_dim: int,
        hidden_size: int
    ):
        super().__init__()
        
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        
        # Task embeddings
        self.task_embeddings = nn.Embedding(num_tasks, embedding_dim)
        
        # Projection to hidden size
        self.projection = nn.Linear(embedding_dim, hidden_size)
        
        # Initialize embeddings
        nn.init.normal_(self.task_embeddings.weight, std=0.02)
    
    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_ids: [batch_size] tensor of task IDs
            
        Returns:
            Task embeddings: [batch_size, hidden_size]
        """
        task_embeds = self.task_embeddings(task_ids)  # [batch_size, embedding_dim]
        projected_embeds = self.projection(task_embeds)  # [batch_size, hidden_size]
        
        return projected_embeds


class TaskHeads(nn.Module):
    """Container for all task-specific heads"""
    
    def __init__(
        self,
        hidden_size: int,
        task_configs: Dict[str, Any],
        vocab_size: Optional[int] = None,
        use_task_embeddings: bool = False,
        task_embedding_dim: int = 64
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.task_configs = task_configs
        self.use_task_embeddings = use_task_embeddings
        
        # Task heads
        self.heads = nn.ModuleDict()
        
        # Task name to ID mapping
        self.task_name_to_id = {name: idx for idx, name in enumerate(task_configs.keys())}
        self.task_id_to_name = {idx: name for name, idx in self.task_name_to_id.items()}
        
        # Initialize task heads
        for task_name, task_config in task_configs.items():
            self.heads[task_name] = self._create_task_head(task_config, vocab_size)
        
        # Task embeddings
        if use_task_embeddings:
            self.task_embedding = TaskEmbedding(
                num_tasks=len(task_configs),
                embedding_dim=task_embedding_dim,
                hidden_size=hidden_size
            )
        
        logger.info(f"TaskHeads initialized with {len(task_configs)} tasks")
    
    def _create_task_head(self, task_config: Any, vocab_size: Optional[int]) -> nn.Module:
        """Create task-specific head based on task type"""
        task_type = task_config.task_type
        
        if task_type == "classification":
            return ClassificationHead(
                hidden_size=self.hidden_size,
                num_labels=task_config.num_labels,
                dropout_rate=getattr(task_config, 'dropout_rate', 0.1),
                use_pooler=getattr(task_config, 'use_pooler', True)
            )
        
        elif task_type == "question_answering":
            return QuestionAnsweringHead(
                hidden_size=self.hidden_size,
                dropout_rate=getattr(task_config, 'dropout_rate', 0.1)
            )
        
        elif task_type == "token_classification":
            return TokenClassificationHead(
                hidden_size=self.hidden_size,
                num_labels=task_config.num_labels,
                dropout_rate=getattr(task_config, 'dropout_rate', 0.1)
            )
        
        elif task_type == "generation":
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for generation tasks")
            
            return GenerationHead(
                hidden_size=self.hidden_size,
                vocab_size=vocab_size,
                dropout_rate=getattr(task_config, 'dropout_rate', 0.1)
            )
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_name: str,
        attention_mask: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through task-specific head
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            task_name: Name of the task
            attention_mask: [batch_size, seq_len] attention mask
            task_ids: [batch_size] task IDs for task embeddings
            
        Returns:
            Task-specific outputs
        """
        if task_name not in self.heads:
            raise ValueError(f"Unknown task: {task_name}")
        
        # Add task embeddings if enabled
        if self.use_task_embeddings and task_ids is not None:
            task_embeds = self.task_embedding(task_ids)  # [batch_size, hidden_size]
            
            # Add task embeddings to hidden states
            # Broadcast task embeddings to all sequence positions
            task_embeds = task_embeds.unsqueeze(1)  # [batch_size, 1, hidden_size]
            hidden_states = hidden_states + task_embeds
        
        # Forward through task-specific head
        head = self.heads[task_name]
        
        if isinstance(head, (ClassificationHead, TokenClassificationHead, GenerationHead)):
            return head(hidden_states, attention_mask)
        elif isinstance(head, QuestionAnsweringHead):
            return head(hidden_states, attention_mask)
        else:
            return head(hidden_states)
    
    def get_task_id(self, task_name: str) -> int:
        """Get task ID from task name"""
        return self.task_name_to_id[task_name]
    
    def get_task_name(self, task_id: int) -> str:
        """Get task name from task ID"""
        return self.task_id_to_name[task_id]
    
    def get_task_names(self) -> List[str]:
        """Get all task names"""
        return list(self.task_configs.keys())
    
    def add_task_head(self, task_name: str, task_config: Any, vocab_size: Optional[int] = None):
        """Add a new task head"""
        if task_name in self.heads:
            logger.warning(f"Task {task_name} already exists, replacing...")
        
        # Create new head
        self.heads[task_name] = self._create_task_head(task_config, vocab_size)
        
        # Update task mappings
        if task_name not in self.task_name_to_id:
            new_id = len(self.task_name_to_id)
            self.task_name_to_id[task_name] = new_id
            self.task_id_to_name[new_id] = task_name
        
        # Update task configs
        self.task_configs[task_name] = task_config
        
        # Update task embeddings if needed
        if self.use_task_embeddings:
            old_num_tasks = self.task_embedding.num_tasks
            new_num_tasks = len(self.task_configs)
            
            if new_num_tasks > old_num_tasks:
                # Expand task embeddings
                old_embeddings = self.task_embedding.task_embeddings.weight.data
                self.task_embedding.task_embeddings = nn.Embedding(
                    new_num_tasks, 
                    self.task_embedding.embedding_dim
                )
                
                # Copy old embeddings
                self.task_embedding.task_embeddings.weight.data[:old_num_tasks] = old_embeddings
                
                # Initialize new embeddings
                nn.init.normal_(
                    self.task_embedding.task_embeddings.weight.data[old_num_tasks:], 
                    std=0.02
                )
                
                self.task_embedding.num_tasks = new_num_tasks
        
        logger.info(f"Added task head for {task_name}")
    
    def remove_task_head(self, task_name: str):
        """Remove a task head"""
        if task_name not in self.heads:
            raise ValueError(f"Task {task_name} not found")
        
        # Remove head
        del self.heads[task_name]
        
        # Remove from configs
        del self.task_configs[task_name]
        
        # Update task mappings
        del self.task_name_to_id[task_name]
        self.task_id_to_name = {idx: name for name, idx in self.task_name_to_id.items()}
        
        logger.info(f"Removed task head for {task_name}")
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters for each task head"""
        param_counts = {}
        
        for task_name, head in self.heads.items():
            param_counts[task_name] = sum(p.numel() for p in head.parameters())
        
        if self.use_task_embeddings:
            param_counts["task_embeddings"] = sum(p.numel() for p in self.task_embedding.parameters())
        
        return param_counts
