"""
Proximal Policy Optimization (PPO) implementation for RLHF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO trainer for RLHF
    """
    
    def __init__(
        self,
        policy_model,
        value_model,
        reward_model,
        tokenizer,
        config
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # PPO hyperparameters
        self.clip_range = config.clip_range
        self.clip_range_vf = config.clip_range_vf or config.clip_range
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.max_grad_norm = config.max_grad_norm
        
        # KL penalty
        self.kl_coef = config.init_kl_coef
        self.target_kl = config.target_kl
        self.adaptive_kl = config.adaptive_kl
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("PPOTrainer initialized")
    
    def generate_responses(
        self,
        prompts: List[str],
        max_length: int = 512,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate responses for given prompts
        
        Args:
            prompts: List of prompt strings
            max_length: Maximum generation length
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated responses and metadata
        """
        # Tokenize prompts
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length // 2
        ).to(self.device)
        
        prompt_length = prompt_inputs["input_ids"].shape[1]
        
        # Generate responses
        self.policy_model.eval()
        with torch.no_grad():
            generation_config = self.config.get_generation_config()
            generation_config.update(generation_kwargs)
            generation_config["max_length"] = prompt_length + max_length // 2
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            generation_config["eos_token_id"] = self.tokenizer.eos_token_id
            
            generated_outputs = self.policy_model.generate(
                **prompt_inputs,
                **generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Extract generated sequences
        generated_sequences = generated_outputs.sequences
        
        # Separate prompts and responses
        response_sequences = generated_sequences[:, prompt_length:]
        
        # Decode responses
        responses = self.tokenizer.batch_decode(
            response_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Compute log probabilities
        log_probs = self._compute_log_probs(
            generated_sequences,
            generated_outputs.scores,
            prompt_length
        )
        
        return {
            "prompts": prompts,
            "responses": responses,
            "prompt_input_ids": prompt_inputs["input_ids"],
            "response_input_ids": response_sequences,
            "full_input_ids": generated_sequences,
            "log_probs": log_probs,
            "prompt_length": prompt_length
        }
    
    def _compute_log_probs(
        self,
        sequences: torch.Tensor,
        scores: Tuple[torch.Tensor],
        prompt_length: int
    ) -> torch.Tensor:
        """
        Compute log probabilities for generated sequences
        
        Args:
            sequences: Generated sequences
            scores: Generation scores
            prompt_length: Length of prompt
            
        Returns:
            Log probabilities for response tokens
        """
        # Convert scores to log probabilities
        log_probs_list = []
        
        for i, score in enumerate(scores):
            log_probs = F.log_softmax(score, dim=-1)
            
            # Get log prob for actual token
            token_ids = sequences[:, prompt_length + i]
            token_log_probs = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
            log_probs_list.append(token_log_probs)
        
        # Stack log probabilities
        if log_probs_list:
            return torch.stack(log_probs_list, dim=1)  # [batch_size, seq_len]
        else:
            return torch.zeros((sequences.shape[0], 0), device=sequences.device)
    
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards for prompt-response pairs
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            Reward scores
        """
        # Combine prompts and responses
        full_texts = [f"{prompt} {response}" for prompt, response in zip(prompts, responses)]
        
        # Get rewards from reward model
        rewards = self.reward_model.score(full_texts, return_tensors=True)
        
        return rewards.to(self.device)
    
    def compute_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute value estimates
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Value estimates
        """
        self.value_model.eval()
        with torch.no_grad():
            values = self.value_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        return values.squeeze(-1)  # Remove last dimension
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 1.0,
        lam: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: Reward values
            values: Value estimates
            gamma: Discount factor
            lam: GAE lambda parameter
            
        Returns:
            Advantages and returns
        """
        batch_size = rewards.shape[0]
        
        # For episodic tasks, we treat each response as a single-step episode
        # So advantages are simply rewards - values
        advantages = rewards - values
        returns = rewards
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO policy loss
        
        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities
            advantages: Advantage estimates
            attention_mask: Attention mask for valid tokens
            
        Returns:
            Policy loss components
        """
        # Compute probability ratios
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            ratio = ratio * attention_mask
            advantages = advantages * attention_mask
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        
        # PPO loss (negative because we want to maximize)
        policy_loss = -torch.min(surr1, surr2)
        
        # Average over valid tokens
        if attention_mask is not None:
            policy_loss = policy_loss.sum() / attention_mask.sum()
        else:
            policy_loss = policy_loss.mean()
        
        # Compute additional metrics
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            clipped_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()
        
        return {
            "policy_loss": policy_loss,
            "approx_kl": approx_kl,
            "clipped_fraction": clipped_fraction
        }
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value function loss
        
        Args:
            values: Current value estimates
            old_values: Old value estimates
            returns: Target returns
            
        Returns:
            Value loss
        """
        # Clipped value loss
        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -self.clip_range_vf,
            self.clip_range_vf
        )
        
        value_loss1 = (values - returns) ** 2
        value_loss2 = (value_pred_clipped - returns) ** 2
        
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        
        return value_loss
    
    def compute_entropy_loss(
        self,
        log_probs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute entropy loss for exploration
        
        Args:
            log_probs: Log probabilities
            attention_mask: Attention mask
            
        Returns:
            Entropy loss
        """
        # Entropy = -sum(p * log(p))
        entropy = -log_probs.exp() * log_probs
        
        if attention_mask is not None:
            entropy = entropy * attention_mask
            entropy_loss = entropy.sum() / attention_mask.sum()
        else:
            entropy_loss = entropy.mean()
        
        return entropy_loss
    
    def ppo_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Perform one PPO update step
        
        Args:
            batch: Batch of training data
            
        Returns:
            Training metrics
        """
        # Extract batch data
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        old_log_probs = batch["log_probs"].to(self.device)
        old_values = batch["values"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        
        # Forward pass through policy model
        self.policy_model.train()
        policy_outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        # Compute current log probabilities
        log_probs = F.log_softmax(policy_outputs.logits, dim=-1)
        log_probs = log_probs.gather(2, input_ids.unsqueeze(2)).squeeze(2)
        
        # Forward pass through value model
        self.value_model.train()
        values = self.value_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).squeeze(-1)
        
        # Compute losses
        policy_loss_dict = self.compute_policy_loss(
            log_probs, old_log_probs, advantages, attention_mask
        )
        
        value_loss = self.compute_value_loss(values, old_values, returns)
        entropy_loss = self.compute_entropy_loss(log_probs, attention_mask)
        
        # Total loss
        total_loss = (
            policy_loss_dict["policy_loss"] +
            self.vf_coef * value_loss -
            self.ent_coef * entropy_loss
        )
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.value_model.parameters(),
                self.max_grad_norm
            )
        
        # Optimizer step
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        # Update KL coefficient if adaptive
        if self.adaptive_kl:
            approx_kl = policy_loss_dict["approx_kl"].item()
            if approx_kl > 2.0 * self.target_kl:
                self.kl_coef *= 1.5
            elif approx_kl < 0.5 * self.target_kl:
                self.kl_coef /= 1.5
        
        # Return metrics
        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss_dict["policy_loss"].item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl": policy_loss_dict["approx_kl"].item(),
            "clipped_fraction": policy_loss_dict["clipped_fraction"].item(),
            "kl_coef": self.kl_coef
        }
    
    def train_step(
        self,
        prompts: List[str],
        num_epochs: int = 4
    ) -> Dict[str, Any]:
        """
        Complete PPO training step
        
        Args:
            prompts: List of training prompts
            num_epochs: Number of PPO epochs
            
        Returns:
            Training results and metrics
        """
        # Generate responses
        generation_results = self.generate_responses(prompts)
        
        # Compute rewards
        rewards = self.compute_rewards(
            generation_results["prompts"],
            generation_results["responses"]
        )
        
        # Compute values
        values = self.compute_values(generation_results["full_input_ids"])
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values)
        
        # Prepare training batch
        batch = {
            "input_ids": generation_results["full_input_ids"],
            "log_probs": generation_results["log_probs"],
            "values": values,
            "advantages": advantages,
            "returns": returns,
            "rewards": rewards
        }
        
        # PPO training epochs
        epoch_metrics = []
        for epoch in range(num_epochs):
            metrics = self.ppo_step(batch)
            metrics["epoch"] = epoch
            epoch_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            if key != "epoch":
                avg_metrics[f"avg_{key}"] = np.mean([m[key] for m in epoch_metrics])
        
        return {
            "generation_results": generation_results,
            "rewards": rewards,
            "advantages": advantages,
            "epoch_metrics": epoch_metrics,
            "avg_metrics": avg_metrics
        }
