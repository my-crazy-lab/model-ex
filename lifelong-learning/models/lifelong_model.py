"""
Main lifelong learning model implementation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from transformers import AutoModel, AutoModelForSequenceClassification

from ..config.lifelong_config import LifelongConfig, LifelongTechnique
from ..techniques.ewc import ElasticWeightConsolidation
from ..techniques.rehearsal import ExperienceReplay
from ..techniques.regularization import L2Regularization, SynapticIntelligence
from .task_heads import TaskSpecificHeads

logger = logging.getLogger(__name__)


class LifelongModel(nn.Module):
    """
    Main lifelong learning model that integrates various continual learning techniques
    """
    
    def __init__(
        self,
        base_model_name: str,
        lifelong_config: LifelongConfig,
        num_labels_per_task: Optional[Dict[int, int]] = None
    ):
        super().__init__()
        
        self.base_model_name = base_model_name
        self.lifelong_config = lifelong_config
        self.num_labels_per_task = num_labels_per_task or {}
        
        # Load base model
        self.base_model = self._load_base_model()
        
        # Task-specific components
        self.task_heads = TaskSpecificHeads(
            hidden_size=self.base_model.config.hidden_size,
            num_labels_per_task=num_labels_per_task
        )
        
        # Lifelong learning techniques
        self.techniques = {}
        self._initialize_techniques()
        
        # Task tracking
        self.current_task_id = None
        self.completed_tasks = []
        self.task_order = []
        
        # Training state
        self.training_step = 0
        
        logger.info(f"LifelongModel initialized with technique: {lifelong_config.technique}")
    
    def _load_base_model(self):
        """Load the base transformer model"""
        try:
            # Try to load as a classification model first
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=2  # Dummy, will be overridden by task heads
            )
            # Remove the classifier head as we'll use task-specific heads
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
            elif hasattr(model, 'cls'):
                model.cls = nn.Identity()
        except:
            # Fall back to base model
            model = AutoModel.from_pretrained(self.base_model_name)
        
        return model
    
    def _initialize_techniques(self):
        """Initialize lifelong learning techniques"""
        config = self.lifelong_config
        
        if config.technique == LifelongTechnique.EWC or LifelongTechnique.EWC in config.combined_techniques:
            self.techniques['ewc'] = ElasticWeightConsolidation(
                model=self.base_model,
                ewc_lambda=config.ewc_lambda,
                gamma=config.ewc_gamma,
                online_ewc=config.online_ewc,
                fisher_estimation_samples=config.fisher_estimation_samples
            )
        
        if config.technique == LifelongTechnique.REHEARSAL or LifelongTechnique.REHEARSAL in config.combined_techniques:
            self.techniques['rehearsal'] = ExperienceReplay(
                memory_size=config.memory_size,
                sampling_strategy=config.memory_strategy.value,
                replay_frequency=config.replay_frequency,
                balanced_replay=config.balanced_replay,
                replay_batch_size=config.replay_batch_size
            )
        
        if config.technique == LifelongTechnique.L2_REGULARIZATION or LifelongTechnique.L2_REGULARIZATION in config.combined_techniques:
            self.techniques['l2_reg'] = L2Regularization(
                model=self.base_model,
                l2_lambda=config.l2_lambda,
                selective=config.selective_l2
            )
        
        if config.technique == LifelongTechnique.SYNAPTIC_INTELLIGENCE or LifelongTechnique.SYNAPTIC_INTELLIGENCE in config.combined_techniques:
            self.techniques['si'] = SynapticIntelligence(
                model=self.base_model,
                si_c=config.si_c,
                xi=config.si_xi
            )
    
    def add_task(self, task_id: int, num_labels: int):
        """Add a new task to the model"""
        if task_id not in self.num_labels_per_task:
            self.num_labels_per_task[task_id] = num_labels
            self.task_heads.add_task_head(task_id, num_labels)
            
            logger.info(f"Added task {task_id} with {num_labels} labels")
    
    def set_current_task(self, task_id: int):
        """Set the current task for training"""
        self.current_task_id = task_id
        
        if task_id not in self.task_order:
            self.task_order.append(task_id)
        
        logger.debug(f"Current task set to {task_id}")
    
    def complete_task(self, task_id: int, dataloader=None):
        """Mark task as completed and update techniques"""
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        
        # Update techniques that need task completion info
        if 'ewc' in self.techniques and dataloader is not None:
            self.techniques['ewc'].register_task(
                task_id=task_id,
                dataloader=dataloader,
                device=next(self.parameters()).device
            )
        
        if 'si' in self.techniques:
            self.techniques['si'].consolidate_task(task_id)
        
        logger.info(f"Task {task_id} marked as completed")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_id: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the lifelong model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task_id: Task identifier
            labels: Ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing logits, loss, and other outputs
        """
        # Use current task if not specified
        if task_id is None:
            task_id = self.current_task_id
        
        if task_id is None:
            raise ValueError("Task ID must be specified or set as current task")
        
        # Forward through base model
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get hidden states
        if hasattr(base_outputs, 'last_hidden_state'):
            hidden_states = base_outputs.last_hidden_state
        elif hasattr(base_outputs, 'hidden_states'):
            hidden_states = base_outputs.hidden_states[-1]
        else:
            hidden_states = base_outputs[0]
        
        # Forward through task-specific head
        task_outputs = self.task_heads(hidden_states, task_id)
        
        outputs = {
            'logits': task_outputs['logits'],
            'hidden_states': hidden_states
        }
        
        # Compute loss if labels provided
        if labels is not None:
            task_loss = nn.CrossEntropyLoss()(task_outputs['logits'], labels)
            
            # Add regularization losses
            total_loss = task_loss
            loss_components = {'task_loss': task_loss}
            
            # EWC loss
            if 'ewc' in self.techniques:
                ewc_loss = self.techniques['ewc'].compute_ewc_loss(task_id)
                total_loss += ewc_loss
                loss_components['ewc_loss'] = ewc_loss
            
            # L2 regularization loss
            if 'l2_reg' in self.techniques:
                l2_loss = self.techniques['l2_reg'].compute_l2_loss()
                total_loss += l2_loss
                loss_components['l2_loss'] = l2_loss
            
            # Synaptic Intelligence loss
            if 'si' in self.techniques:
                si_loss = self.techniques['si'].compute_si_loss()
                total_loss += si_loss
                loss_components['si_loss'] = si_loss
            
            outputs['loss'] = total_loss
            outputs['loss_components'] = loss_components
        
        return outputs
    
    def compute_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_id: Optional[int] = None,
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        Compute prediction uncertainty using Monte Carlo dropout
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task_id: Task identifier
            num_samples: Number of MC samples
            
        Returns:
            Uncertainty scores for each example
        """
        self.train()  # Enable dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_id=task_id
                )
                
                probs = torch.softmax(outputs['logits'], dim=-1)
                predictions.append(probs)
        
        # Compute uncertainty as entropy of mean prediction
        mean_probs = torch.stack(predictions).mean(dim=0)
        uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        self.eval()  # Disable dropout
        
        return uncertainty
    
    def get_task_performance(self, task_id: int, dataloader, device: torch.device) -> Dict[str, float]:
        """Evaluate performance on specific task"""
        self.eval()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.forward(task_id=task_id, **batch)
                
                # Compute accuracy
                predictions = torch.argmax(outputs['logits'], dim=-1)
                correct = (predictions == batch['labels']).sum().item()
                
                total_correct += correct
                total_samples += batch['labels'].size(0)
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'num_samples': total_samples
        }
    
    def evaluate_all_tasks(self, task_dataloaders: Dict[int, Any], device: torch.device) -> Dict[str, Any]:
        """Evaluate performance on all completed tasks"""
        results = {}
        
        for task_id in self.completed_tasks:
            if task_id in task_dataloaders:
                task_results = self.get_task_performance(
                    task_id=task_id,
                    dataloader=task_dataloaders[task_id],
                    device=device
                )
                results[f'task_{task_id}'] = task_results
        
        # Compute average metrics
        if results:
            avg_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
            avg_loss = sum(r['loss'] for r in results.values()) / len(results)
            
            results['average'] = {
                'accuracy': avg_accuracy,
                'loss': avg_loss,
                'num_tasks': len(results)
            }
        
        return results
    
    def get_forgetting_metrics(
        self,
        task_dataloaders: Dict[int, Any],
        device: torch.device,
        baseline_performance: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """Compute forgetting metrics"""
        if baseline_performance is None:
            # Use perfect performance as baseline
            baseline_performance = {task_id: 1.0 for task_id in self.completed_tasks}
        
        current_results = self.evaluate_all_tasks(task_dataloaders, device)
        
        forgetting_scores = {}
        total_forgetting = 0.0
        
        for task_id in self.completed_tasks:
            if f'task_{task_id}' in current_results and task_id in baseline_performance:
                current_acc = current_results[f'task_{task_id}']['accuracy']
                baseline_acc = baseline_performance[task_id]
                
                forgetting = max(0, baseline_acc - current_acc)
                forgetting_scores[f'task_{task_id}_forgetting'] = forgetting
                total_forgetting += forgetting
        
        if forgetting_scores:
            forgetting_scores['average_forgetting'] = total_forgetting / len(forgetting_scores)
        
        return forgetting_scores
    
    def save_model(self, save_path: str):
        """Save the lifelong model"""
        import os
        import json
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        model_state = {
            'base_model_state_dict': self.base_model.state_dict(),
            'task_heads_state_dict': self.task_heads.state_dict(),
            'num_labels_per_task': self.num_labels_per_task,
            'completed_tasks': self.completed_tasks,
            'task_order': self.task_order,
            'current_task_id': self.current_task_id
        }
        
        torch.save(model_state, os.path.join(save_path, 'model.pt'))
        
        # Save configuration
        with open(os.path.join(save_path, 'lifelong_config.json'), 'w') as f:
            json.dump(self.lifelong_config.to_dict(), f, indent=2)
        
        # Save technique states
        for name, technique in self.techniques.items():
            if hasattr(technique, 'save_state'):
                technique.save_state(os.path.join(save_path, f'{name}_state.pt'))
        
        logger.info(f"Lifelong model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load the lifelong model"""
        import os
        import json
        
        # Load model state
        model_state = torch.load(os.path.join(load_path, 'model.pt'), map_location='cpu')
        
        self.base_model.load_state_dict(model_state['base_model_state_dict'])
        self.task_heads.load_state_dict(model_state['task_heads_state_dict'])
        self.num_labels_per_task = model_state['num_labels_per_task']
        self.completed_tasks = model_state['completed_tasks']
        self.task_order = model_state['task_order']
        self.current_task_id = model_state['current_task_id']
        
        # Load technique states
        for name, technique in self.techniques.items():
            technique_path = os.path.join(load_path, f'{name}_state.pt')
            if os.path.exists(technique_path) and hasattr(technique, 'load_state'):
                technique.load_state(technique_path)
        
        logger.info(f"Lifelong model loaded from {load_path}")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        stats = {
            'num_completed_tasks': len(self.completed_tasks),
            'current_task_id': self.current_task_id,
            'task_order': self.task_order,
            'num_labels_per_task': self.num_labels_per_task,
            'training_step': self.training_step
        }
        
        # Add technique-specific statistics
        for name, technique in self.techniques.items():
            if hasattr(technique, 'get_statistics'):
                stats[f'{name}_stats'] = technique.get_statistics()
            elif hasattr(technique, 'get_ewc_statistics'):
                stats[f'{name}_stats'] = technique.get_ewc_statistics()
        
        # Model size statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        stats['model_size'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_efficiency': trainable_params / total_params * 100
        }
        
        return stats
