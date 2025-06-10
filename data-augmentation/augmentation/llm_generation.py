"""
LLM-based synthetic data generation
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional, Union
import openai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from ..config.generation_config import GenerationConfig, GenerationModel

logger = logging.getLogger(__name__)


class LLMGenerator:
    """LLM-based synthetic data generator"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
        # Initialize model based on configuration
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the generation model"""
        if isinstance(self.config.model_name, GenerationModel):
            model_name = self.config.model_name.value
        else:
            model_name = self.config.model_name
        
        if "gpt" in model_name.lower() or "claude" in model_name.lower():
            # API-based models
            self._initialize_api_model()
        else:
            # Local transformer models
            self._initialize_local_model(model_name)
    
    def _initialize_api_model(self):
        """Initialize API-based model (GPT, Claude)"""
        if isinstance(self.config.model_name, GenerationModel):
            model_name = self.config.model_name.value
        else:
            model_name = self.config.model_name
        
        if "gpt" in model_name.lower():
            # OpenAI GPT models
            if self.config.api_key:
                openai.api_key = self.config.api_key
            if self.config.api_base:
                openai.api_base = self.config.api_base
        
        logger.info(f"Initialized API model: {model_name}")
    
    def _initialize_local_model(self, model_name: str):
        """Initialize local transformer model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            logger.info(f"Initialized local model: {model_name}")
        
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            raise
    
    def generate_with_gpt(self, prompt: str, num_samples: int = 1) -> List[str]:
        """Generate text using OpenAI GPT models"""
        if isinstance(self.config.model_name, GenerationModel):
            model_name = self.config.model_name.value
        else:
            model_name = self.config.model_name
        
        generations = []
        params = self.config.get_model_params()
        
        for _ in range(num_samples):
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self.config.system_prompt or "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    **params
                )
                
                generated_text = response.choices[0].message.content.strip()
                generations.append(generated_text)
                
                # Add delay to respect rate limits
                if self.config.retry_delay > 0:
                    time.sleep(self.config.retry_delay)
            
            except Exception as e:
                logger.error(f"Error generating with GPT: {e}")
                if len(generations) == 0:
                    raise
                break
        
        return generations
    
    def generate_with_t5(self, prompt: str, num_samples: int = 1) -> List[str]:
        """Generate text using T5 models"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("T5 model not initialized")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        generations = []
        params = self.config.get_model_params()
        
        # Generate samples
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_return_sequences=num_samples,
                do_sample=True,
                **params
            )
        
        # Decode outputs
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            generations.append(generated_text.strip())
        
        return generations
    
    def generate(
        self,
        prompt: str,
        num_samples: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate synthetic text
        
        Args:
            prompt: Input prompt for generation
            num_samples: Number of samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        if num_samples is None:
            num_samples = self.config.num_samples
        
        if isinstance(self.config.model_name, GenerationModel):
            model_name = self.config.model_name.value
        else:
            model_name = self.config.model_name
        
        # Route to appropriate generation method
        if "gpt" in model_name.lower():
            generations = self.generate_with_gpt(prompt, num_samples)
        elif "claude" in model_name.lower():
            generations = self.generate_with_claude(prompt, num_samples)
        elif "t5" in model_name.lower():
            generations = self.generate_with_t5(prompt, num_samples)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Post-process generations
        generations = self._post_process_generations(generations)
        
        return generations
    
    def generate_with_claude(self, prompt: str, num_samples: int = 1) -> List[str]:
        """Generate text using Anthropic Claude models"""
        # This would require the Anthropic API client
        # For now, return placeholder
        logger.warning("Claude generation not implemented yet")
        return [f"Generated text {i+1}" for i in range(num_samples)]
    
    def _post_process_generations(self, generations: List[str]) -> List[str]:
        """Post-process generated texts"""
        processed = []
        
        for text in generations:
            # Clean up text
            text = text.strip()
            
            # Apply length constraints
            if len(text) < self.config.min_length:
                continue
            
            if len(text) > self.config.max_length:
                text = text[:self.config.max_length]
            
            # Remove duplicates if configured
            if self.config.filter_duplicates:
                if text not in processed:
                    processed.append(text)
            else:
                processed.append(text)
        
        return processed
    
    def generate_batch(
        self,
        prompts: List[str],
        num_samples_per_prompt: int = 1
    ) -> List[List[str]]:
        """
        Generate text for multiple prompts
        
        Args:
            prompts: List of input prompts
            num_samples_per_prompt: Number of samples per prompt
            
        Returns:
            List of lists, each containing generations for a prompt
        """
        all_generations = []
        
        # Process in batches
        for i in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[i:i + self.config.batch_size]
            batch_generations = []
            
            for prompt in batch_prompts:
                generations = self.generate(prompt, num_samples_per_prompt)
                batch_generations.append(generations)
            
            all_generations.extend(batch_generations)
            
            # Add delay between batches
            if self.config.retry_delay > 0:
                time.sleep(self.config.retry_delay)
        
        return all_generations
    
    def generate_with_template(
        self,
        template: str,
        variables: Dict[str, List[str]],
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate text using templates with variables
        
        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary mapping variable names to possible values
            num_samples: Number of samples to generate
            
        Returns:
            List of generated texts
        """
        import itertools
        import random
        
        # Generate all possible combinations
        var_names = list(variables.keys())
        var_values = list(variables.values())
        
        combinations = list(itertools.product(*var_values))
        
        # Sample combinations if too many
        if len(combinations) > num_samples:
            combinations = random.sample(combinations, num_samples)
        
        generated_texts = []
        
        for combination in combinations:
            # Fill template
            filled_template = template
            for var_name, var_value in zip(var_names, combination):
                filled_template = filled_template.replace(f"{{{var_name}}}", var_value)
            
            # Generate based on filled template
            generations = self.generate(filled_template, 1)
            generated_texts.extend(generations)
        
        return generated_texts[:num_samples]
    
    def estimate_cost(self, prompts: List[str]) -> float:
        """Estimate generation cost"""
        total_cost = 0.0
        
        for prompt in prompts:
            # Estimate tokens (rough approximation)
            estimated_tokens = len(prompt.split()) * 1.3  # Account for tokenization
            estimated_tokens += self.config.max_tokens  # Output tokens
            
            # Get cost per token (this would need to be updated with current pricing)
            cost_per_token = self._get_cost_per_token()
            total_cost += estimated_tokens * cost_per_token
        
        return total_cost
    
    def _get_cost_per_token(self) -> float:
        """Get cost per token for the current model"""
        if isinstance(self.config.model_name, GenerationModel):
            model_name = self.config.model_name.value
        else:
            model_name = self.config.model_name
        
        # Rough cost estimates (as of 2024, per 1K tokens)
        costs = {
            "gpt-3.5-turbo": 0.002 / 1000,
            "gpt-4": 0.03 / 1000,
            "gpt-4-turbo-preview": 0.01 / 1000,
        }
        
        return costs.get(model_name, 0.0)


class PromptTemplate:
    """Template for structured prompt generation"""
    
    def __init__(
        self,
        template: str,
        variables: Dict[str, List[str]],
        examples: Optional[List[Dict[str, str]]] = None
    ):
        self.template = template
        self.variables = variables
        self.examples = examples or []
    
    def generate_prompts(self, num_prompts: int = 10) -> List[str]:
        """Generate prompts using the template"""
        import random
        
        prompts = []
        
        for _ in range(num_prompts):
            # Sample variables
            filled_template = self.template
            for var_name, var_options in self.variables.items():
                var_value = random.choice(var_options)
                filled_template = filled_template.replace(f"{{{var_name}}}", var_value)
            
            # Add examples if provided
            if self.examples:
                example_text = "\n".join([
                    f"Example: {ex.get('input', '')} -> {ex.get('output', '')}"
                    for ex in self.examples[:3]  # Limit to 3 examples
                ])
                filled_template = f"{example_text}\n\n{filled_template}"
            
            prompts.append(filled_template)
        
        return prompts
