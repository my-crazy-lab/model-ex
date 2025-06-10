"""
Elastic Weight Consolidation (EWC) implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class FisherInformationMatrix:
    """Computes and manages Fisher Information Matrix for EWC"""
    
    def __init__(
        self,
        model: nn.Module,
        estimation_method: str = "diagonal",
        num_samples: int = 1000
    ):
        self.model = model
        self.estimation_method = estimation_method
        self.num_samples = num_samples
        self.fisher_dict = {}
        
    def compute_fisher_information(
        self,
        dataloader,
        task_id: int,
        device: torch.device = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix for given task
        
        Args:
            dataloader: DataLoader for the task
            task_id: Task identifier
            device: Device to compute on
            
        Returns:
            Dictionary mapping parameter names to Fisher information
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        self.model.eval()
        fisher_dict = {}
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
        
        logger.info(f"Computing Fisher information for task {task_id}...")
        
        num_samples_processed = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if num_samples_processed >= self.num_samples:
                break
            
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = [item.to(device) if isinstance(item, torch.Tensor) else item 
                        for item in batch]
            
            # Forward pass
            self.model.zero_grad()
            
            if isinstance(batch, dict):
                outputs = self.model(**batch)
            else:
                outputs = self.model(*batch)
            
            # Get logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Sample from model's prediction (for Fisher information)
            probs = F.softmax(logits, dim=-1)
            sampled_labels = torch.multinomial(probs, 1).squeeze()
            
            # Compute loss with sampled labels
            loss = F.cross_entropy(logits, sampled_labels)
            
            # Backward pass
            loss.backward()
            
            # Accumulate Fisher information
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if self.estimation_method == "diagonal":
                        # Diagonal Fisher (most common)
                        fisher_dict[name] += param.grad.data ** 2
                    elif self.estimation_method == "full":
                        # Full Fisher matrix (memory intensive)
                        # For simplicity, we'll use diagonal approximation
                        fisher_dict[name] += param.grad.data ** 2
            
            num_samples_processed += batch[0].size(0) if isinstance(batch, (list, tuple)) else batch['input_ids'].size(0)
        
        # Normalize by number of samples
        for name in fisher_dict:
            fisher_dict[name] /= num_samples_processed
        
        # Store Fisher information for this task
        self.fisher_dict[task_id] = fisher_dict
        
        logger.info(f"Fisher information computed for task {task_id} using {num_samples_processed} samples")
        
        return fisher_dict
    
    def get_fisher_information(self, task_id: int) -> Dict[str, torch.Tensor]:
        """Get Fisher information for specific task"""
        return self.fisher_dict.get(task_id, {})
    
    def get_consolidated_fisher(self, task_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Get consolidated Fisher information across multiple tasks"""
        if not task_ids:
            return {}
        
        consolidated = {}
        
        # Get parameter names from first task
        first_task_fisher = self.fisher_dict.get(task_ids[0], {})
        
        for name in first_task_fisher:
            consolidated[name] = torch.zeros_like(first_task_fisher[name])
            
            # Sum Fisher information across tasks
            for task_id in task_ids:
                if task_id in self.fisher_dict and name in self.fisher_dict[task_id]:
                    consolidated[name] += self.fisher_dict[task_id][name]
        
        return consolidated


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        gamma: float = 1.0,
        online_ewc: bool = False,
        fisher_estimation_samples: int = 1000,
        fisher_estimation_method: str = "diagonal"
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma  # Decay factor for online EWC
        self.online_ewc = online_ewc
        
        # Fisher information matrix computer
        self.fisher_computer = FisherInformationMatrix(
            model=model,
            estimation_method=fisher_estimation_method,
            num_samples=fisher_estimation_samples
        )
        
        # Store optimal parameters for each task
        self.optimal_params = {}
        
        # Store consolidated Fisher information
        self.consolidated_fisher = {}
        
        # Task tracking
        self.current_task = 0
        self.completed_tasks = []
        
        logger.info("EWC initialized")
    
    def register_task(
        self,
        task_id: int,
        dataloader,
        device: torch.device = None
    ):
        """
        Register a new task and compute Fisher information
        
        Args:
            task_id: Task identifier
            dataloader: DataLoader for computing Fisher information
            device: Device to compute on
        """
        logger.info(f"Registering task {task_id} for EWC...")
        
        # Store current optimal parameters
        self.optimal_params[task_id] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[task_id][name] = param.data.clone()
        
        # Compute Fisher information for this task
        fisher_info = self.fisher_computer.compute_fisher_information(
            dataloader, task_id, device
        )
        
        # Update consolidated Fisher information
        if self.online_ewc:
            self._update_consolidated_fisher_online(task_id, fisher_info)
        else:
            self._update_consolidated_fisher_offline(task_id, fisher_info)
        
        # Update task tracking
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        
        self.current_task = task_id
        
        logger.info(f"Task {task_id} registered successfully")
    
    def _update_consolidated_fisher_online(
        self,
        task_id: int,
        fisher_info: Dict[str, torch.Tensor]
    ):
        """Update Fisher information using online EWC"""
        if not self.consolidated_fisher:
            # First task
            self.consolidated_fisher = copy.deepcopy(fisher_info)
        else:
            # Update with decay
            for name in fisher_info:
                if name in self.consolidated_fisher:
                    self.consolidated_fisher[name] = (
                        self.gamma * self.consolidated_fisher[name] + 
                        fisher_info[name]
                    )
                else:
                    self.consolidated_fisher[name] = fisher_info[name]
    
    def _update_consolidated_fisher_offline(
        self,
        task_id: int,
        fisher_info: Dict[str, torch.Tensor]
    ):
        """Update Fisher information using offline EWC"""
        if not self.consolidated_fisher:
            # First task
            self.consolidated_fisher = copy.deepcopy(fisher_info)
        else:
            # Sum Fisher information across all tasks
            for name in fisher_info:
                if name in self.consolidated_fisher:
                    self.consolidated_fisher[name] += fisher_info[name]
                else:
                    self.consolidated_fisher[name] = fisher_info[name]
    
    def compute_ewc_loss(self, current_task_id: Optional[int] = None) -> torch.Tensor:
        """
        Compute EWC regularization loss
        
        Args:
            current_task_id: Current task ID (for task-specific EWC)
            
        Returns:
            EWC loss tensor
        """
        if not self.completed_tasks or not self.consolidated_fisher:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self.consolidated_fisher:
                continue
            
            # Get Fisher information
            fisher = self.consolidated_fisher[name]
            
            # Get optimal parameters (use most recent task if not specified)
            if current_task_id is not None and current_task_id in self.optimal_params:
                optimal_param = self.optimal_params[current_task_id][name]
            elif self.completed_tasks:
                # Use parameters from the most recent completed task
                recent_task = self.completed_tasks[-1]
                optimal_param = self.optimal_params[recent_task][name]
            else:
                continue
            
            # Compute EWC penalty: F * (θ - θ*)^2
            penalty = fisher * (param - optimal_param) ** 2
            ewc_loss += penalty.sum()
        
        return self.ewc_lambda * ewc_loss
    
    def get_total_loss(
        self,
        task_loss: torch.Tensor,
        current_task_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute total loss including EWC regularization
        
        Args:
            task_loss: Loss from current task
            current_task_id: Current task ID
            
        Returns:
            Total loss (task loss + EWC loss)
        """
        ewc_loss = self.compute_ewc_loss(current_task_id)
        total_loss = task_loss + ewc_loss
        
        return total_loss
    
    def get_ewc_statistics(self) -> Dict[str, float]:
        """Get EWC statistics for monitoring"""
        stats = {
            "num_completed_tasks": len(self.completed_tasks),
            "current_task": self.current_task,
            "ewc_lambda": self.ewc_lambda,
            "online_ewc": self.online_ewc
        }
        
        if self.consolidated_fisher:
            # Compute Fisher information statistics
            total_fisher = 0.0
            num_params = 0
            
            for name, fisher in self.consolidated_fisher.items():
                total_fisher += fisher.sum().item()
                num_params += fisher.numel()
            
            stats.update({
                "total_fisher_info": total_fisher,
                "avg_fisher_info": total_fisher / max(num_params, 1),
                "num_fisher_params": num_params
            })
        
        return stats
    
    def save_state(self, filepath: str):
        """Save EWC state"""
        state = {
            "optimal_params": self.optimal_params,
            "consolidated_fisher": self.consolidated_fisher,
            "completed_tasks": self.completed_tasks,
            "current_task": self.current_task,
            "ewc_lambda": self.ewc_lambda,
            "gamma": self.gamma,
            "online_ewc": self.online_ewc
        }
        
        torch.save(state, filepath)
        logger.info(f"EWC state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load EWC state"""
        state = torch.load(filepath, map_location="cpu")
        
        self.optimal_params = state["optimal_params"]
        self.consolidated_fisher = state["consolidated_fisher"]
        self.completed_tasks = state["completed_tasks"]
        self.current_task = state["current_task"]
        self.ewc_lambda = state["ewc_lambda"]
        self.gamma = state["gamma"]
        self.online_ewc = state["online_ewc"]
        
        logger.info(f"EWC state loaded from {filepath}")
    
    def reset(self):
        """Reset EWC state"""
        self.optimal_params.clear()
        self.consolidated_fisher.clear()
        self.completed_tasks.clear()
        self.current_task = 0
        
        logger.info("EWC state reset")
