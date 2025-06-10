"""
Reward model implementations for RLHF
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    PreTrainedModel, PreTrainedTokenizer
)

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Base reward model for RLHF
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        num_labels: int = 1,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        
        # Reward head
        self.dropout = nn.Dropout(dropout_rate)
        self.reward_head = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize reward head
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)
        
        # Tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"RewardModel initialized with {model_name_or_path}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through reward model
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing rewards and hidden states
        """
        # Forward through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get sequence representation
        # Use last token representation for causal models
        if attention_mask is not None:
            # Find last non-padding token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            sequence_representation = outputs.last_hidden_state[
                torch.arange(batch_size, device=input_ids.device), sequence_lengths
            ]
        else:
            # Use last token if no attention mask
            sequence_representation = outputs.last_hidden_state[:, -1, :]
        
        # Apply dropout and compute reward
        sequence_representation = self.dropout(sequence_representation)
        rewards = self.reward_head(sequence_representation)
        
        return {
            "rewards": rewards,
            "hidden_states": outputs.last_hidden_state,
            "sequence_representation": sequence_representation
        }
    
    def score(
        self,
        texts: Union[str, List[str]],
        return_tensors: bool = False
    ) -> Union[float, List[float], torch.Tensor]:
        """
        Score text(s) with the reward model
        
        Args:
            texts: Text or list of texts to score
            return_tensors: Whether to return tensors or Python values
            
        Returns:
            Reward score(s)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to model device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(**inputs)
            rewards = outputs["rewards"].squeeze(-1)  # Remove last dimension if single label
        
        if return_tensors:
            return rewards
        else:
            rewards_list = rewards.cpu().tolist()
            return rewards_list[0] if len(rewards_list) == 1 else rewards_list
    
    def compare(
        self,
        text_a: str,
        text_b: str
    ) -> Dict[str, Any]:
        """
        Compare two texts and return preference
        
        Args:
            text_a: First text
            text_b: Second text
            
        Returns:
            Comparison results
        """
        scores = self.score([text_a, text_b])
        score_a, score_b = scores[0], scores[1]
        
        preference = "a" if score_a > score_b else "b"
        confidence = abs(score_a - score_b)
        
        return {
            "preference": preference,
            "confidence": confidence,
            "score_a": score_a,
            "score_b": score_b,
            "score_diff": score_a - score_b
        }
    
    def save_pretrained(self, save_directory: str):
        """Save the reward model"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save backbone
        self.backbone.save_pretrained(save_directory)
        
        # Save reward head
        torch.save(
            self.reward_head.state_dict(),
            os.path.join(save_directory, "reward_head.pt")
        )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save config
        import json
        config_dict = {
            "model_name_or_path": self.model_name_or_path,
            "num_labels": self.num_labels,
            "hidden_size": self.config.hidden_size
        }
        
        with open(os.path.join(save_directory, "reward_model_config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Reward model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ) -> "RewardModel":
        """Load reward model from pretrained path"""
        import os
        import json
        
        # Load config
        config_path = os.path.join(model_path, "reward_model_config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create model
        model = cls(
            model_name_or_path=config_dict["model_name_or_path"],
            tokenizer=tokenizer,
            num_labels=config_dict["num_labels"]
        )
        
        # Load backbone
        model.backbone = AutoModel.from_pretrained(model_path)
        
        # Load reward head
        reward_head_path = os.path.join(model_path, "reward_head.pt")
        if os.path.exists(reward_head_path):
            model.reward_head.load_state_dict(torch.load(reward_head_path, map_location="cpu"))
        
        # Load tokenizer if not provided
        if tokenizer is None:
            model.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"Reward model loaded from {model_path}")
        return model


class PreferenceRewardModel(RewardModel):
    """
    Reward model trained on preference data
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Preference-specific parameters
        self.preference_margin = 0.1
        self.temperature = 1.0
    
    def compute_preference_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        margin: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute preference loss for training
        
        Args:
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses
            margin: Margin for preference loss
            
        Returns:
            Preference loss
        """
        if margin is None:
            margin = self.preference_margin
        
        # Preference loss: chosen should have higher reward than rejected
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards - margin))
        return loss.mean()
    
    def forward_preference(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for preference training
        
        Args:
            chosen_input_ids: Input IDs for chosen responses
            chosen_attention_mask: Attention mask for chosen responses
            rejected_input_ids: Input IDs for rejected responses
            rejected_attention_mask: Attention mask for rejected responses
            
        Returns:
            Dictionary containing rewards and loss
        """
        # Forward pass for chosen responses
        chosen_outputs = self.forward(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        chosen_rewards = chosen_outputs["rewards"].squeeze(-1)
        
        # Forward pass for rejected responses
        rejected_outputs = self.forward(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        rejected_rewards = rejected_outputs["rewards"].squeeze(-1)
        
        # Compute preference loss
        loss = self.compute_preference_loss(chosen_rewards, rejected_rewards)
        
        return {
            "loss": loss,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "reward_difference": chosen_rewards - rejected_rewards
        }


class ConstitutionalRewardModel(RewardModel):
    """
    Reward model for Constitutional AI
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        constitution: List[str],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs
    ):
        super().__init__(model_name_or_path, tokenizer, **kwargs)
        
        self.constitution = constitution
        self.principle_weights = torch.ones(len(constitution))
    
    def evaluate_constitutional_principles(
        self,
        text: str,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Evaluate text against constitutional principles
        
        Args:
            text: Text to evaluate
            return_details: Whether to return detailed scores
            
        Returns:
            Constitutional score or detailed evaluation
        """
        principle_scores = []
        
        for i, principle in enumerate(self.constitution):
            # Create evaluation prompt
            eval_prompt = f"Principle: {principle}\nText: {text}\nDoes this text follow the principle? Score (0-1):"
            
            # Score against principle (simplified - in practice would use more sophisticated evaluation)
            score = self.score(eval_prompt)
            principle_scores.append(score)
        
        principle_scores = torch.tensor(principle_scores)
        
        # Weighted average
        constitutional_score = torch.sum(principle_scores * self.principle_weights) / torch.sum(self.principle_weights)
        
        if return_details:
            return {
                "constitutional_score": constitutional_score.item(),
                "principle_scores": principle_scores.tolist(),
                "principles": self.constitution
            }
        else:
            return constitutional_score.item()
    
    def self_critique(
        self,
        text: str,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Perform self-critique and improvement
        
        Args:
            text: Text to critique and improve
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            Critique and improvement results
        """
        current_text = text
        critique_history = []
        
        for iteration in range(max_iterations):
            # Evaluate current text
            eval_result = self.evaluate_constitutional_principles(current_text, return_details=True)
            
            # Generate critique
            critique_prompt = f"Critique this text based on constitutional principles:\n{current_text}\nCritique:"
            critique = self.score(critique_prompt)  # In practice, would generate text
            
            # Generate improvement
            improve_prompt = f"Improve this text:\n{current_text}\nCritique: {critique}\nImproved text:"
            improved_text = current_text  # In practice, would generate improved text
            
            critique_history.append({
                "iteration": iteration,
                "text": current_text,
                "constitutional_score": eval_result["constitutional_score"],
                "critique": critique,
                "improved_text": improved_text
            })
            
            current_text = improved_text
            
            # Stop if score is high enough
            if eval_result["constitutional_score"] > 0.9:
                break
        
        return {
            "original_text": text,
            "final_text": current_text,
            "critique_history": critique_history,
            "improvement_iterations": len(critique_history)
        }


class MultiObjectiveRewardModel(nn.Module):
    """
    Multi-objective reward model combining multiple reward signals
    """
    
    def __init__(
        self,
        reward_models: Dict[str, RewardModel],
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.reward_models = nn.ModuleDict(reward_models)
        
        # Set weights
        if weights is None:
            weights = {name: 1.0 / len(reward_models) for name in reward_models}
        
        self.weights = weights
        
        logger.info(f"MultiObjectiveRewardModel initialized with {len(reward_models)} objectives")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all reward models
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Combined rewards and individual objective scores
        """
        objective_rewards = {}
        total_reward = 0.0
        
        for name, model in self.reward_models.items():
            outputs = model.forward(input_ids, attention_mask, **kwargs)
            objective_rewards[name] = outputs["rewards"]
            
            # Add weighted contribution to total reward
            weight = self.weights.get(name, 1.0)
            total_reward += weight * outputs["rewards"]
        
        return {
            "rewards": total_reward,
            "objective_rewards": objective_rewards,
            "weights": self.weights
        }
    
    def score(
        self,
        texts: Union[str, List[str]],
        return_breakdown: bool = False
    ) -> Union[float, List[float], Dict[str, Any]]:
        """
        Score text(s) with multi-objective reward model
        
        Args:
            texts: Text or list of texts to score
            return_breakdown: Whether to return breakdown by objective
            
        Returns:
            Total reward score(s) or detailed breakdown
        """
        if isinstance(texts, str):
            texts = [texts]
        
        objective_scores = {}
        total_scores = []
        
        for name, model in self.reward_models.items():
            scores = model.score(texts)
            if not isinstance(scores, list):
                scores = [scores]
            objective_scores[name] = scores
        
        # Compute weighted total scores
        for i in range(len(texts)):
            total_score = sum(
                self.weights.get(name, 1.0) * objective_scores[name][i]
                for name in objective_scores
            )
            total_scores.append(total_score)
        
        if return_breakdown:
            return {
                "total_scores": total_scores[0] if len(total_scores) == 1 else total_scores,
                "objective_scores": {
                    name: scores[0] if len(scores) == 1 else scores
                    for name, scores in objective_scores.items()
                },
                "weights": self.weights
            }
        else:
            return total_scores[0] if len(total_scores) == 1 else total_scores
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update objective weights"""
        self.weights.update(new_weights)
        logger.info(f"Updated reward weights: {self.weights}")
    
    def add_objective(self, name: str, model: RewardModel, weight: float = 1.0):
        """Add new objective to multi-objective model"""
        self.reward_models[name] = model
        self.weights[name] = weight
        logger.info(f"Added objective '{name}' with weight {weight}")
    
    def remove_objective(self, name: str):
        """Remove objective from multi-objective model"""
        if name in self.reward_models:
            del self.reward_models[name]
            del self.weights[name]
            logger.info(f"Removed objective '{name}'")
        else:
            logger.warning(f"Objective '{name}' not found")
