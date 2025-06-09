"""
Text Classification example using LoRA/PEFT
"""

import os
import logging
from datasets import load_dataset

from config import ModelConfig, PEFTConfig, TrainingConfig, CLASSIFICATION_CONFIG
from data import load_dataset_from_hub, TextClassificationPreprocessor
from training import PEFTTrainer
from evaluation import ModelEvaluator
from inference import InferencePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function for text classification example"""
    
    # Configuration
    model_config = ModelConfig(
        model_name_or_path="bert-base-uncased",
        num_labels=2,
        max_length=512,
        use_quantization=False,  # Set to True for quantized training
    )
    
    peft_config = CLASSIFICATION_CONFIG
    
    training_config = TrainingConfig(
        output_dir="./results/text_classification",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-4,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to=None,  # Set to "wandb" for W&B logging
    )
    
    logger.info("Starting text classification example")
    
    # Load dataset
    logger.info("Loading IMDB dataset...")
    dataset = load_dataset_from_hub("imdb")
    
    # Take a smaller subset for quick testing
    train_dataset = dataset["train"].select(range(1000))
    eval_dataset = dataset["test"].select(range(500))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Setup trainer
    trainer = PEFTTrainer(
        model_config=model_config,
        peft_config=peft_config,
        training_config=training_config,
        task_type="classification"
    )
    
    # Setup preprocessor
    model_wrapper = trainer.setup_model()
    tokenizer = model_wrapper.get_tokenizer()
    
    preprocessor = TextClassificationPreprocessor(
        tokenizer=tokenizer,
        text_column="text",
        label_column="label",
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
    
    # Test samples
    test_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible movie, waste of time. Very disappointing.",
        "The plot was okay but the acting was great.",
        "Not the best movie I've seen, but not the worst either."
    ]
    
    # Get predictions
    predictions = inference_pipeline.classify_text(
        test_texts,
        return_all_scores=True
    )
    
    logger.info("Inference results:")
    for text, pred in zip(test_texts, predictions):
        logger.info(f"Text: {text}")
        logger.info(f"Prediction: {pred}")
        logger.info("-" * 50)
    
    # Detailed evaluation
    logger.info("Performing detailed evaluation...")
    evaluator = ModelEvaluator(
        model_wrapper=model_wrapper,
        task_type="classification"
    )
    
    detailed_results = evaluator.evaluate_dataset(
        eval_dataset,
        preprocessor=preprocessor,
        save_predictions=True,
        output_dir=os.path.join(training_config.output_dir, "evaluation")
    )
    
    logger.info(f"Detailed evaluation: {detailed_results['metrics']}")
    
    # Model comparison (compare with base model)
    logger.info("Comparing with base model...")
    
    # Create base model evaluator (without PEFT)
    base_model_config = ModelConfig(
        model_name_or_path="bert-base-uncased",
        num_labels=2,
    )
    
    # Note: In practice, you would train a base model or load a pre-trained one
    # For this example, we'll skip the actual comparison
    
    logger.info("Text classification example completed!")


if __name__ == "__main__":
    main()
