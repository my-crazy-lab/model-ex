"""
Complete chatbot RLHF example
"""

import argparse
import logging
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from typing import List, Dict, Any

from config import RLHFConfig
from models import RewardModel, PreferenceRewardModel
from training import RLHFTrainer
from algorithms import PPOTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_preference_dataset(size: int = 100) -> Dataset:
    """Create sample preference dataset for demonstration"""
    
    logger.info("Creating sample preference dataset...")
    
    # Sample prompts for chatbot training
    prompts = [
        "How can I improve my productivity at work?",
        "What are some healthy breakfast ideas?",
        "Explain quantum computing in simple terms.",
        "How do I learn a new programming language?",
        "What's the best way to manage stress?",
        "Can you help me plan a weekend trip?",
        "How do I start investing in stocks?",
        "What are the benefits of meditation?",
        "How can I improve my communication skills?",
        "What should I know about climate change?"
    ] * (size // 10)
    
    # Generate chosen and rejected responses
    chosen_responses = [
        "Here are some evidence-based strategies to boost your productivity: 1) Use time-blocking to schedule focused work periods, 2) Eliminate distractions by turning off notifications, 3) Take regular breaks using the Pomodoro technique, 4) Prioritize tasks using the Eisenhower matrix. These methods have been shown to significantly improve work efficiency.",
        "Here are some nutritious and quick breakfast options: 1) Greek yogurt with berries and nuts, 2) Overnight oats with fruits, 3) Avocado toast with eggs, 4) Smoothie bowls with protein powder, 5) Whole grain cereal with milk. These provide balanced nutrition to start your day right.",
        "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, unlike classical bits that are either 0 or 1. This allows quantum computers to process many possibilities at once, potentially solving certain problems exponentially faster than classical computers. Think of it like exploring a maze - classical computers try one path at a time, while quantum computers can explore all paths simultaneously.",
        "To learn a new programming language effectively: 1) Start with the basics and syntax, 2) Practice with small projects, 3) Read others' code, 4) Join online communities, 5) Build progressively complex projects. Consistency is key - dedicate 30-60 minutes daily rather than long irregular sessions.",
        "Effective stress management techniques include: 1) Deep breathing exercises, 2) Regular physical exercise, 3) Adequate sleep (7-9 hours), 4) Mindfulness meditation, 5) Time management, 6) Social support. If stress persists, consider speaking with a mental health professional."
    ] * (size // 5)
    
    rejected_responses = [
        "Just work harder and you'll be more productive. Stop being lazy.",
        "Eat whatever you want for breakfast, it doesn't really matter.",
        "Quantum computing is too complicated to explain. It's just really fast computers.",
        "Programming is hard. You probably shouldn't bother learning it.",
        "Stress is just in your head. Just ignore it and it will go away."
    ] * (size // 5)
    
    # Create dataset
    data = []
    for i in range(size):
        data.append({
            "prompt": prompts[i],
            "chosen": chosen_responses[i % len(chosen_responses)],
            "rejected": rejected_responses[i % len(rejected_responses)]
        })
    
    return Dataset.from_list(data)


def create_sample_prompts(size: int = 50) -> List[str]:
    """Create sample prompts for RL training"""
    
    prompts = [
        "How can I be more creative?",
        "What's the meaning of life?",
        "How do I make friends as an adult?",
        "What are some good books to read?",
        "How can I save money effectively?",
        "What should I cook for dinner tonight?",
        "How do I deal with difficult people?",
        "What are the latest technology trends?",
        "How can I improve my memory?",
        "What's the best way to learn a new skill?"
    ] * (size // 10)
    
    return prompts[:size]


def train_reward_model(
    preference_dataset: Dataset,
    model_name: str = "gpt2",
    output_dir: str = "./reward_model"
) -> str:
    """Train reward model on preference data"""
    
    logger.info("🎯 Training reward model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create reward model
    reward_model = PreferenceRewardModel(
        model_name_or_path=model_name,
        tokenizer=tokenizer,
        num_labels=1
    )
    
    # Prepare training data
    def preprocess_preference_data(examples):
        # Combine prompt with responses
        chosen_texts = [f"{prompt} {chosen}" for prompt, chosen in zip(examples["prompt"], examples["chosen"])]
        rejected_texts = [f"{prompt} {rejected}" for prompt, rejected in zip(examples["prompt"], examples["rejected"])]
        
        # Tokenize
        chosen_inputs = tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        rejected_inputs = tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_inputs["input_ids"],
            "chosen_attention_mask": chosen_inputs["attention_mask"],
            "rejected_input_ids": rejected_inputs["input_ids"],
            "rejected_attention_mask": rejected_inputs["attention_mask"]
        }
    
    # Preprocess dataset
    processed_dataset = preference_dataset.map(
        preprocess_preference_data,
        batched=True,
        desc="Preprocessing preference data"
    )
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)
    
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)
    
    # Training loop
    num_epochs = 3
    batch_size = 4
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        total_loss = 0.0
        num_batches = 0
        
        # Create batches
        for i in range(0, len(processed_dataset), batch_size):
            batch_end = min(i + batch_size, len(processed_dataset))
            batch = processed_dataset[i:batch_end]
            
            # Move to device
            chosen_input_ids = torch.stack([torch.tensor(x) for x in batch["chosen_input_ids"]]).to(device)
            chosen_attention_mask = torch.stack([torch.tensor(x) for x in batch["chosen_attention_mask"]]).to(device)
            rejected_input_ids = torch.stack([torch.tensor(x) for x in batch["rejected_input_ids"]]).to(device)
            rejected_attention_mask = torch.stack([torch.tensor(x) for x in batch["rejected_attention_mask"]]).to(device)
            
            # Forward pass
            outputs = reward_model.forward_preference(
                chosen_input_ids=chosen_input_ids,
                chosen_attention_mask=chosen_attention_mask,
                rejected_input_ids=rejected_input_ids,
                rejected_attention_mask=rejected_attention_mask
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Average loss: {avg_loss:.4f}")
    
    # Save reward model
    os.makedirs(output_dir, exist_ok=True)
    reward_model.save_pretrained(output_dir)
    
    logger.info(f"✅ Reward model saved to {output_dir}")
    return output_dir


def run_ppo_training(
    model_name: str,
    reward_model_path: str,
    prompts: List[str],
    config: RLHFConfig,
    output_dir: str = "./ppo_model"
) -> str:
    """Run PPO training with reward model"""
    
    logger.info("🚀 Starting PPO training...")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Policy model (the model we're training)
    policy_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Value model (copy of policy model for value estimation)
    value_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Reward model
    reward_model = PreferenceRewardModel.from_pretrained(reward_model_path, tokenizer=tokenizer)
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config
    )
    
    # Training loop
    num_training_steps = 10
    batch_size = config.batch_size
    
    for step in range(num_training_steps):
        logger.info(f"Training step {step + 1}/{num_training_steps}")
        
        # Sample prompts for this step
        step_prompts = prompts[step * batch_size:(step + 1) * batch_size]
        if len(step_prompts) < batch_size:
            # Repeat prompts if needed
            step_prompts = (step_prompts * (batch_size // len(step_prompts) + 1))[:batch_size]
        
        # PPO training step
        results = ppo_trainer.train_step(step_prompts, num_epochs=config.ppo_epochs)
        
        # Log metrics
        metrics = results["avg_metrics"]
        logger.info(f"Step {step + 1} metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Log sample generations
        if step % 5 == 0:
            logger.info("Sample generations:")
            for i, (prompt, response) in enumerate(zip(results["generation_results"]["prompts"][:3], 
                                                     results["generation_results"]["responses"][:3])):
                reward = results["rewards"][i].item()
                logger.info(f"  Prompt: {prompt}")
                logger.info(f"  Response: {response}")
                logger.info(f"  Reward: {reward:.3f}")
                logger.info("")
    
    # Save trained model
    os.makedirs(output_dir, exist_ok=True)
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"✅ PPO trained model saved to {output_dir}")
    return output_dir


def evaluate_model(
    model_path: str,
    test_prompts: List[str],
    reward_model_path: str
) -> Dict[str, Any]:
    """Evaluate trained model"""
    
    logger.info("📊 Evaluating trained model...")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    reward_model = PreferenceRewardModel.from_pretrained(reward_model_path, tokenizer=tokenizer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    reward_model.to(device)
    
    # Generate responses
    model.eval()
    generated_responses = []
    rewards = []
    
    for prompt in test_prompts:
        # Tokenize prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Get reward
        full_text = f"{prompt} {response}"
        reward = reward_model.score(full_text)
        
        generated_responses.append(response)
        rewards.append(reward)
    
    # Compute metrics
    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)
    
    results = {
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "min_reward": min_reward,
        "num_samples": len(test_prompts),
        "sample_generations": [
            {"prompt": prompt, "response": response, "reward": reward}
            for prompt, response, reward in zip(test_prompts[:5], generated_responses[:5], rewards[:5])
        ]
    }
    
    logger.info(f"Evaluation results:")
    logger.info(f"  Average reward: {avg_reward:.3f}")
    logger.info(f"  Max reward: {max_reward:.3f}")
    logger.info(f"  Min reward: {min_reward:.3f}")
    
    return results


def main():
    """Main chatbot RLHF example"""
    
    parser = argparse.ArgumentParser(description="Chatbot RLHF Example")
    parser.add_argument("--model_name", default="gpt2", help="Base model name")
    parser.add_argument("--preference_data_size", type=int, default=100, help="Size of preference dataset")
    parser.add_argument("--num_prompts", type=int, default=50, help="Number of training prompts")
    parser.add_argument("--output_dir", default="./rlhf_results", help="Output directory")
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "dpo"], help="RLHF algorithm")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO epochs per step")
    parser.add_argument("--skip_reward_training", action="store_true", help="Skip reward model training")
    
    args = parser.parse_args()
    
    logger.info("🎯 Starting Chatbot RLHF example...")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create preference dataset
    preference_dataset = create_sample_preference_dataset(args.preference_data_size)
    logger.info(f"Created preference dataset with {len(preference_dataset)} examples")
    
    # Step 2: Train reward model (if not skipping)
    reward_model_path = os.path.join(args.output_dir, "reward_model")
    
    if not args.skip_reward_training:
        reward_model_path = train_reward_model(
            preference_dataset=preference_dataset,
            model_name=args.model_name,
            output_dir=reward_model_path
        )
    else:
        logger.info(f"Skipping reward model training, using existing model at {reward_model_path}")
    
    # Step 3: Create training prompts
    training_prompts = create_sample_prompts(args.num_prompts)
    logger.info(f"Created {len(training_prompts)} training prompts")
    
    # Step 4: Setup RLHF configuration
    rlhf_config = RLHFConfig(
        model_name_or_path=args.model_name,
        algorithm=args.algorithm,
        reward_model_path=reward_model_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        max_length=256,
        output_dir=args.output_dir
    )
    
    # Step 5: Run RLHF training
    if args.algorithm == "ppo":
        trained_model_path = run_ppo_training(
            model_name=args.model_name,
            reward_model_path=reward_model_path,
            prompts=training_prompts,
            config=rlhf_config,
            output_dir=os.path.join(args.output_dir, "ppo_model")
        )
    else:
        # DPO training would go here
        logger.info("DPO training not implemented in this example")
        return
    
    # Step 6: Evaluate trained model
    test_prompts = [
        "How can I be happier in life?",
        "What's the best way to learn new things?",
        "How do I build better relationships?",
        "What should I do when I feel overwhelmed?",
        "How can I make a positive impact on the world?"
    ]
    
    evaluation_results = evaluate_model(
        model_path=trained_model_path,
        test_prompts=test_prompts,
        reward_model_path=reward_model_path
    )
    
    # Save results
    results_summary = {
        "config": rlhf_config.to_dict(),
        "training_prompts_count": len(training_prompts),
        "preference_data_size": len(preference_dataset),
        "evaluation_results": evaluation_results
    }
    
    with open(os.path.join(args.output_dir, "rlhf_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("🎯 CHATBOT RLHF RESULTS")
    logger.info("="*60)
    logger.info(f"Base model: {args.model_name}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Training prompts: {len(training_prompts)}")
    logger.info(f"Preference data: {len(preference_dataset)} examples")
    logger.info(f"Average reward: {evaluation_results['avg_reward']:.3f}")
    logger.info(f"Reward range: {evaluation_results['min_reward']:.3f} - {evaluation_results['max_reward']:.3f}")
    
    logger.info("\nSample generations:")
    for i, sample in enumerate(evaluation_results["sample_generations"]):
        logger.info(f"\n{i+1}. Prompt: {sample['prompt']}")
        logger.info(f"   Response: {sample['response']}")
        logger.info(f"   Reward: {sample['reward']:.3f}")
    
    logger.info("="*60)
    logger.info("✅ Chatbot RLHF example completed!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
