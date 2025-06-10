"""
Distillation loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..config.distillation_config import DistillationConfig, LossType

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """Comprehensive distillation loss functions"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        
        # Initialize loss functions
        self.distillation_loss_fn = self._get_loss_function(config.distillation_loss_type)
        self.feature_loss_fn = self._get_loss_function(config.feature_loss_type)
        self.attention_loss_fn = self._get_loss_function(config.attention_loss_type)
        
        # Task loss (standard cross-entropy)
        self.task_loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        logger.info("DistillationLoss initialized")
    
    def _get_loss_function(self, loss_type: LossType):
        """Get loss function based on type"""
        if loss_type == LossType.KL_DIVERGENCE:
            return self._kl_divergence_loss
        elif loss_type == LossType.MSE:
            return F.mse_loss
        elif loss_type == LossType.COSINE_SIMILARITY:
            return self._cosine_similarity_loss
        elif loss_type == LossType.L1:
            return F.l1_loss
        elif loss_type == LossType.HUBER:
            return F.huber_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def _kl_divergence_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """KL divergence loss for logit distillation"""
        if temperature is None:
            temperature = self.config.temperature
        
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence loss
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # Scale by temperature squared (as in original paper)
        return kl_loss * (temperature ** 2)
    
    def _cosine_similarity_loss(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        """Cosine similarity loss for feature matching"""
        # Normalize features
        student_norm = F.normalize(student_features, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
        
        # Cosine similarity
        cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=-1)
        
        # Convert to loss (1 - similarity)
        return 1.0 - cosine_sim.mean()
    
    def compute_logit_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = None
    ) -> torch.Tensor:
        """Compute logit distillation loss"""
        return self.distillation_loss_fn(student_logits, teacher_logits, temperature)
    
    def compute_feature_distillation_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        layer_mapping: Optional[Dict[int, int]] = None
    ) -> torch.Tensor:
        """Compute feature distillation loss"""
        if len(student_features) != len(teacher_features):
            if layer_mapping is None:
                raise ValueError("Layer mapping required when feature dimensions differ")
        
        total_loss = 0.0
        num_layers = 0
        
        for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
            # Project student features if dimensions don't match
            if student_feat.size(-1) != teacher_feat.size(-1):
                # Simple linear projection
                projection = nn.Linear(student_feat.size(-1), teacher_feat.size(-1)).to(student_feat.device)
                student_feat = projection(student_feat)
            
            # Normalize features if configured
            if self.config.normalize_features:
                student_feat = F.normalize(student_feat, p=2, dim=-1)
                teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)
            
            # Compute feature loss
            layer_loss = self.feature_loss_fn(student_feat, teacher_feat)
            total_loss += layer_loss
            num_layers += 1
        
        return total_loss / max(num_layers, 1)
    
    def compute_attention_distillation_loss(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention distillation loss"""
        total_loss = 0.0
        num_layers = 0
        
        for student_attn, teacher_attn in zip(student_attentions, teacher_attentions):
            # Handle different number of attention heads
            if student_attn.size(1) != teacher_attn.size(1):
                # Average teacher attention heads to match student
                teacher_attn = teacher_attn.mean(dim=1, keepdim=True).expand(-1, student_attn.size(1), -1, -1)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Expand mask for attention dimensions
                mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
                mask = mask.expand_as(student_attn)
                
                # Mask out padded positions
                student_attn = student_attn.masked_fill(~mask, 0.0)
                teacher_attn = teacher_attn.masked_fill(~mask, 0.0)
            
            # Compute attention loss
            layer_loss = self.attention_loss_fn(student_attn, teacher_attn)
            total_loss += layer_loss
            num_layers += 1
        
        return total_loss / max(num_layers, 1)
    
    def compute_combined_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        labels: torch.Tensor,
        temperature: float = None,
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Compute combined distillation loss"""
        losses = {}
        
        # Get current temperature
        if temperature is None:
            temperature = self.config.get_temperature_at_step(step)
        
        # Task loss (hard targets)
        if labels is not None:
            task_loss = self.task_loss_fn(student_outputs['logits'], labels)
            losses['task_loss'] = task_loss
        
        # Logit distillation loss (soft targets)
        if 'logits' in teacher_outputs:
            distillation_loss = self.compute_logit_distillation_loss(
                student_outputs['logits'],
                teacher_outputs['logits'],
                temperature
            )
            losses['distillation_loss'] = distillation_loss
        
        # Feature distillation loss
        if ('hidden_states' in student_outputs and 'hidden_states' in teacher_outputs and
            self.config.distillation_type in ['feature', 'combined']):
            
            feature_loss = self.compute_feature_distillation_loss(
                student_outputs['hidden_states'],
                teacher_outputs['hidden_states']
            )
            losses['feature_loss'] = feature_loss
        
        # Attention distillation loss
        if ('attentions' in student_outputs and 'attentions' in teacher_outputs and
            self.config.distillation_type in ['attention', 'combined']):
            
            attention_loss = self.compute_attention_distillation_loss(
                student_outputs['attentions'],
                teacher_outputs['attentions'],
                student_outputs.get('attention_mask')
            )
            losses['attention_loss'] = attention_loss
        
        # Combine losses with weights
        loss_weights = self.config.get_loss_weights()
        total_loss = 0.0
        
        if 'task_loss' in losses:
            total_loss += loss_weights.get('task', 0.0) * losses['task_loss']
        
        if 'distillation_loss' in losses:
            total_loss += loss_weights.get('distillation', 0.0) * losses['distillation_loss']
        
        if 'feature_loss' in losses:
            total_loss += loss_weights.get('feature', 0.0) * losses['feature_loss']
        
        if 'attention_loss' in losses:
            total_loss += loss_weights.get('attention', 0.0) * losses['attention_loss']
        
        losses['total_loss'] = total_loss
        losses['temperature'] = torch.tensor(temperature)
        
        return losses
    
    def compute_progressive_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        labels: torch.Tensor,
        stage: int,
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Compute progressive distillation loss"""
        # Adjust loss weights based on stage
        original_alpha = self.config.alpha
        original_beta = self.config.beta
        
        # Progressive weighting: start with more task loss, gradually increase distillation
        stage_progress = stage / self.config.progressive_stages
        self.config.alpha = original_alpha * stage_progress + 0.1 * (1 - stage_progress)
        self.config.beta = 1.0 - self.config.alpha
        
        # Compute combined loss
        losses = self.compute_combined_loss(student_outputs, teacher_outputs, labels, step=step)
        
        # Restore original weights
        self.config.alpha = original_alpha
        self.config.beta = original_beta
        
        losses['stage'] = torch.tensor(stage)
        losses['stage_progress'] = torch.tensor(stage_progress)
        
        return losses
    
    def adaptive_temperature_update(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        current_temperature: float
    ) -> float:
        """Adaptively update temperature based on prediction confidence"""
        if not self.config.adaptive_temperature:
            return current_temperature
        
        # Compute prediction confidence
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Entropy as measure of uncertainty
        student_entropy = -torch.sum(student_probs * torch.log(student_probs + 1e-8), dim=-1).mean()
        teacher_entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-8), dim=-1).mean()
        
        # Adjust temperature based on entropy difference
        entropy_diff = teacher_entropy - student_entropy
        temperature_adjustment = self.config.temperature_adaptation_rate * entropy_diff.item()
        
        # Update temperature with bounds
        new_temperature = current_temperature + temperature_adjustment
        new_temperature = max(1.0, min(10.0, new_temperature))  # Clamp between 1 and 10
        
        return new_temperature
    
    def get_loss_statistics(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get loss statistics for logging"""
        stats = {}
        
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                stats[loss_name] = loss_value.item()
            else:
                stats[loss_name] = loss_value
        
        # Compute loss ratios
        if 'total_loss' in stats and stats['total_loss'] > 0:
            for loss_name in ['task_loss', 'distillation_loss', 'feature_loss', 'attention_loss']:
                if loss_name in stats:
                    stats[f'{loss_name}_ratio'] = stats[loss_name] / stats['total_loss']
        
        return stats
