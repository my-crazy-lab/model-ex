"""
Text Generation example using LoRA/PEFT
"""

import os
import logging
from datasets import Dataset

from config import ModelConfig, PEFTConfig, TrainingConfig, GENERATION_CONFIG
from data import TextGenerationPreprocessor
from training import PEFTTrainer
from inference import InferencePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset():
    """Create a sample dataset for text generation"""
    
    # Sample data for instruction following
    data = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "Describe the benefits of renewable energy sources.",
        "Explain how photosynthesis works in plants.",
        "Write a recipe for chocolate chip cookies.",
        "Describe the process of how rain is formed.",
        "Explain the importance of biodiversity in ecosystems.",
        "Write a brief summary of the water cycle.",
        "Describe the basic principles of democracy.",
        "Explain how vaccines work to prevent diseases.",
    ] * 100  # Repeat for more training data
    
    return Dataset.from_dict({"text": data})


def main():
    """Main function for text generation example"""
    
    # Configuration
    model_config = ModelConfig(
        model_name_or_path="microsoft/DialoGPT-small",  # Small model for testing
        max_length=256,
        use_quantization=False,  # Set to True for quantized training
    )
    
    peft_config = GENERATION_CONFIG
    peft_config.r = 8  # Smaller rank for faster training
    peft_config.lora_alpha = 16
    
    training_config = TrainingConfig(
        output_dir="./results/text_generation",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Set to "wandb" for W&B logging
    )
    
    logger.info("Starting text generation example")
    
    # Create sample dataset
    logger.info("Creating sample dataset...")
    dataset = create_sample_dataset()
    
    # Split dataset
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Setup trainer
    trainer = PEFTTrainer(
        model_config=model_config,
        peft_config=peft_config,
        training_config=training_config,
        task_type="generation"
    )
    
    # Setup preprocessor
    model_wrapper = trainer.setup_model()
    tokenizer = model_wrapper.get_tokenizer()
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    preprocessor = TextGenerationPreprocessor(
        tokenizer=tokenizer,
        text_column="text",
        max_length=model_config.max_length,
    )
    
    # Train model
    logger.info("Starting training...")
    train_result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocessor=preprocessor
    )
    
    logger.info(f"Training completed: {train_result}")
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate(eval_dataset)
    logger.info(f"Evaluation results: {eval_results}")
    
    # Test inference
    logger.info("Testing inference...")
    
    # Load the trained model for inference
    inference_pipeline = InferencePipeline(
        model_path=training_config.output_dir,
        model_config=model_config,
        peft_config=peft_config,
    )
    
    # Test prompts
    test_prompts = [
        "Explain the concept of",
        "Write a short story about",
        "Describe the benefits of",
        "The process of photosynthesis",
    ]
    
    # Generate text
    logger.info("Generating text...")
    for prompt in test_prompts:
        generated = inference_pipeline.generate_text(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
        )
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated: {generated}")
        logger.info("-" * 50)
    
    # Test with different generation parameters
    logger.info("Testing different generation parameters...")
    
    prompt = "The importance of education"
    
    # Greedy decoding
    greedy_output = inference_pipeline.generate_text(
        prompt,
        max_new_tokens=30,
        do_sample=False,
    )
    
    # Sampling with high temperature
    creative_output = inference_pipeline.generate_text(
        prompt,
        max_new_tokens=30,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
    )
    
    # Sampling with low temperature
    conservative_output = inference_pipeline.generate_text(
        prompt,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )
    
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Greedy: {greedy_output}")
    logger.info(f"Creative (temp=1.0): {creative_output}")
    logger.info(f"Conservative (temp=0.3): {conservative_output}")
    
    # Generate multiple sequences
    logger.info("Generating multiple sequences...")
    
    multiple_outputs = inference_pipeline.generate_text(
        "The future of artificial intelligence",
        max_new_tokens=25,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=3,
    )
    
    logger.info("Multiple generations:")
    for i, output in enumerate(multiple_outputs):
        logger.info(f"Generation {i+1}: {output}")
    
    # Test batch generation
    logger.info("Testing batch generation...")
    
    batch_prompts = [
        "Climate change is",
        "Technology has changed",
        "The benefits of exercise",
    ]
    
    batch_outputs = inference_pipeline.generate_text(
        batch_prompts,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
    )
    
    for prompt, output in zip(batch_prompts, batch_outputs):
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Output: {output}")
        logger.info("-" * 30)
    
    logger.info("Text generation example completed!")


if __name__ == "__main__":
    main()
