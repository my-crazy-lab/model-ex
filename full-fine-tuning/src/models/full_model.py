"""
Full Fine-Tuning Model Implementation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertForSequenceClassification, RobertaForSequenceClassification,
    BertForTokenClassification, RobertaForTokenClassification,
    BertForQuestionAnswering, RobertaForQuestionAnswering,
    T5ForConditionalGeneration, GPT2LMHeadModel
)
from .task_heads import (
    ClassificationHead, TokenClassificationHead, 
    QuestionAnsweringHead, GenerationHead
)

logger = logging.getLogger(__name__)


class FullFinetuningModel(nn.Module):
    """
    Full fine-tuning model with all parameters trainable
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        task_head: Optional[nn.Module] = None,
        task: str = 'classification',
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.backbone = backbone
        self.task_head = task_head
        self.task = task
        self.freeze_backbone = freeze_backbone
        self.gradient_checkpointing = gradient_checkpointing
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing and hasattr(backbone, 'gradient_checkpointing_enable'):
            backbone.gradient_checkpointing_enable()
        
        # Set backbone training mode
        if not freeze_backbone:
            self._unfreeze_backbone()
        else:
            self._freeze_backbone()
        
        # Track parameter counts
        self._update_param_tracking()
        
        logger.info(f"Initialized FullFinetuningModel for task: {task}")
        logger.info(f"Backbone frozen: {freeze_backbone}")
        logger.info(f"Gradient checkpointing: {gradient_checkpointing}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        task: str = 'classification',
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        freeze_backbone: bool = False,
        use_task_specific_model: bool = True,
        **kwargs
    ) -> 'FullFinetuningModel':
        """
        Create full fine-tuning model from pre-trained backbone
        
        Args:
            model_name_or_path: Pre-trained model name or path
            task: Task type ('classification', 'token_classification', 'question_answering', 'generation')
            num_classes: Number of classes for classification
            num_labels: Number of labels for token classification
            freeze_backbone: Whether to freeze backbone parameters
            use_task_specific_model: Whether to use task-specific model from transformers
            
        Returns:
            FullFinetuningModel instance
        """
        try:
            # Use task-specific models from transformers if available
            if use_task_specific_model:
                model = cls._load_task_specific_model(
                    model_name_or_path, task, num_classes, num_labels, **kwargs
                )
                
                return cls(
                    backbone=model,
                    task_head=None,  # Task head is integrated
                    task=task,
                    freeze_backbone=freeze_backbone,
                    **kwargs
                )
            
            # Load generic backbone and add custom task head
            backbone = AutoModel.from_pretrained(model_name_or_path, **kwargs)
            config = AutoConfig.from_pretrained(model_name_or_path)
            
            # Create task-specific head
            task_head = cls._create_task_head(
                task=task,
                hidden_size=config.hidden_size,
                num_classes=num_classes,
                num_labels=num_labels,
                **kwargs
            )
            
            return cls(
                backbone=backbone,
                task_head=task_head,
                task=task,
                freeze_backbone=freeze_backbone,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Failed to create model from {model_name_or_path}: {e}")
            raise
    
    @staticmethod
    def _load_task_specific_model(
        model_name_or_path: str,
        task: str,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """Load task-specific model from transformers"""
        
        if task == 'classification':
            if 'bert' in model_name_or_path.lower():
                return BertForSequenceClassification.from_pretrained(
                    model_name_or_path, num_labels=num_classes or 2, **kwargs
                )
            elif 'roberta' in model_name_or_path.lower():
                return RobertaForSequenceClassification.from_pretrained(
                    model_name_or_path, num_labels=num_classes or 2, **kwargs
                )
            else:
                # Fallback to AutoModel
                from transformers import AutoModelForSequenceClassification
                return AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path, num_labels=num_classes or 2, **kwargs
                )
        
        elif task == 'token_classification':
            if 'bert' in model_name_or_path.lower():
                return BertForTokenClassification.from_pretrained(
                    model_name_or_path, num_labels=num_labels or 9, **kwargs
                )
            elif 'roberta' in model_name_or_path.lower():
                return RobertaForTokenClassification.from_pretrained(
                    model_name_or_path, num_labels=num_labels or 9, **kwargs
                )
            else:
                from transformers import AutoModelForTokenClassification
                return AutoModelForTokenClassification.from_pretrained(
                    model_name_or_path, num_labels=num_labels or 9, **kwargs
                )
        
        elif task == 'question_answering':
            if 'bert' in model_name_or_path.lower():
                return BertForQuestionAnswering.from_pretrained(
                    model_name_or_path, **kwargs
                )
            elif 'roberta' in model_name_or_path.lower():
                return RobertaForQuestionAnswering.from_pretrained(
                    model_name_or_path, **kwargs
                )
            else:
                from transformers import AutoModelForQuestionAnswering
                return AutoModelForQuestionAnswering.from_pretrained(
                    model_name_or_path, **kwargs
                )
        
        elif task == 'generation':
            if 't5' in model_name_or_path.lower():
                return T5ForConditionalGeneration.from_pretrained(
                    model_name_or_path, **kwargs
                )
            elif 'gpt' in model_name_or_path.lower():
                return GPT2LMHeadModel.from_pretrained(
                    model_name_or_path, **kwargs
                )
            else:
                from transformers import AutoModelForCausalLM
                return AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, **kwargs
                )
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    @staticmethod
    def _create_task_head(
        task: str,
        hidden_size: int,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """Create task-specific head"""
        
        if task == 'classification':
            return ClassificationHead(
                hidden_size=hidden_size,
                num_classes=num_classes or 2,
                **kwargs
            )
        elif task == 'token_classification':
            return TokenClassificationHead(
                hidden_size=hidden_size,
                num_labels=num_labels or 9,
                **kwargs
            )
        elif task == 'question_answering':
            return QuestionAnsweringHead(
                hidden_size=hidden_size,
                **kwargs
            )
        elif task == 'generation':
            return GenerationHead(
                hidden_size=hidden_size,
                vocab_size=kwargs.get('vocab_size', 30522),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        logger.info("Backbone parameters frozen")
    
    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        logger.info("Backbone parameters unfrozen")
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specific layers by name"""
        frozen_count = 0
        for name, param in self.backbone.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                frozen_count += 1
        
        self._update_param_tracking()
        logger.info(f"Frozen {frozen_count} parameters in layers: {layer_names}")
    
    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers by name"""
        unfrozen_count = 0
        for name, param in self.backbone.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                unfrozen_count += 1
        
        self._update_param_tracking()
        logger.info(f"Unfrozen {unfrozen_count} parameters in layers: {layer_names}")
    
    def gradual_unfreeze(self, num_layers: int = 1):
        """Gradually unfreeze layers from top to bottom"""
        # Get layer names (assuming transformer architecture)
        layer_names = []
        for name, _ in self.backbone.named_parameters():
            if 'layer' in name and 'encoder' in name:
                layer_num = int(name.split('layer.')[1].split('.')[0])
                layer_names.append(f'layer.{layer_num}')
        
        # Sort layers in descending order (top layers first)
        unique_layers = sorted(set(layer_names), key=lambda x: int(x.split('.')[1]), reverse=True)
        
        # Unfreeze top num_layers
        layers_to_unfreeze = unique_layers[:num_layers]
        self.unfreeze_layers(layers_to_unfreeze)
        
        logger.info(f"Gradually unfroze {num_layers} top layers: {layers_to_unfreeze}")
    
    def _update_param_tracking(self):
        """Update tracking of frozen and trainable parameters"""
        self._frozen_params = set()
        self._trainable_params = set()
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                self._trainable_params.add(name)
            else:
                self._frozen_params.add(name)
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count total, trainable, and frozen parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Labels for classification/generation
            start_positions: Start positions for QA
            end_positions: End positions for QA
            
        Returns:
            Dictionary containing outputs
        """
        # If using integrated task-specific model
        if self.task_head is None:
            # Pass all inputs to the integrated model
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            
            if token_type_ids is not None:
                model_inputs['token_type_ids'] = token_type_ids
            
            if labels is not None:
                model_inputs['labels'] = labels
            
            if start_positions is not None:
                model_inputs['start_positions'] = start_positions
            
            if end_positions is not None:
                model_inputs['end_positions'] = end_positions
            
            # Add any additional kwargs
            model_inputs.update(kwargs)
            
            return self.backbone(**model_inputs)
        
        # Custom backbone + task head approach
        backbone_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        if token_type_ids is not None:
            backbone_inputs['token_type_ids'] = token_type_ids
        
        # Forward through backbone
        backbone_outputs = self.backbone(**backbone_inputs)
        
        # Forward through task head
        if self.task == 'classification':
            # Use pooled output for classification
            pooled_output = backbone_outputs.pooler_output
            logits = self.task_head(pooled_output)
            
            outputs = {'logits': logits}
            
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                outputs['loss'] = loss
        
        elif self.task == 'token_classification':
            # Use sequence output for token classification
            sequence_output = backbone_outputs.last_hidden_state
            logits = self.task_head(sequence_output)
            
            outputs = {'logits': logits}
            
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                # Flatten for loss computation
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fn.ignore_index).type_as(labels)
                )
                loss = loss_fn(active_logits, active_labels)
                outputs['loss'] = loss
        
        elif self.task == 'question_answering':
            # Use sequence output for QA
            sequence_output = backbone_outputs.last_hidden_state
            start_logits, end_logits = self.task_head(sequence_output)
            
            outputs = {
                'start_logits': start_logits,
                'end_logits': end_logits
            }
            
            if start_positions is not None and end_positions is not None:
                loss_fn = nn.CrossEntropyLoss()
                start_loss = loss_fn(start_logits, start_positions)
                end_loss = loss_fn(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
                outputs['loss'] = loss
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return outputs
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters"""
        return [param for param in self.parameters() if param.requires_grad]
    
    def get_frozen_parameters(self) -> List[nn.Parameter]:
        """Get list of frozen parameters"""
        return [param for param in self.parameters() if not param.requires_grad]
    
    def print_parameter_status(self):
        """Print detailed parameter status"""
        param_counts = self._count_parameters()
        
        print(f"Parameter Status:")
        print(f"  Total parameters: {param_counts['total']:,}")
        print(f"  Trainable parameters: {param_counts['trainable']:,}")
        print(f"  Frozen parameters: {param_counts['frozen']:,}")
        print(f"  Trainable ratio: {param_counts['trainable']/param_counts['total']:.2%}")
        
        # Print layer-wise status
        print(f"\nLayer-wise parameter status:")
        for name, param in self.named_parameters():
            status = "✓ Trainable" if param.requires_grad else "✗ Frozen"
            print(f"  {name}: {param.numel():,} parameters - {status}")
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        if hasattr(self.backbone, 'save_pretrained'):
            self.backbone.save_pretrained(save_directory)
        else:
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save configuration
        config = {
            'task': self.task,
            'freeze_backbone': self.freeze_backbone,
            'gradient_checkpointing': self.gradient_checkpointing,
            'parameter_counts': self._count_parameters()
        }
        
        with open(os.path.join(save_directory, "training_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {save_directory}")


# Example usage and testing
if __name__ == "__main__":
    # Test with BERT for classification
    model = FullFinetuningModel.from_pretrained(
        'bert-base-uncased',
        task='classification',
        num_classes=3,
        freeze_backbone=False
    )
    
    # Print parameter status
    model.print_parameter_status()
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)
    labels = torch.randint(0, 3, (2,))
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    print(f"Outputs keys: {outputs.keys()}")
    if 'logits' in outputs:
        print(f"Logits shape: {outputs['logits'].shape}")
    if 'loss' in outputs:
        print(f"Loss: {outputs['loss'].item():.4f}")
    
    print("FullFinetuningModel test completed successfully!")
