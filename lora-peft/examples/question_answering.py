"""
Question Answering example using LoRA/PEFT
"""

import os
import logging
from datasets import load_dataset

from config import ModelConfig, PEFTConfig, TrainingConfig, QA_CONFIG
from data import load_dataset_from_hub, QuestionAnsweringPreprocessor
from training import PEFTTrainer
from evaluation import ModelEvaluator
from inference import InferencePipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function for question answering example"""
    
    # Configuration
    model_config = ModelConfig(
        model_name_or_path="bert-base-uncased",
        max_length=384,  # Shorter for QA
        use_quantization=False,
    )
    
    peft_config = QA_CONFIG
    peft_config.r = 16
    peft_config.lora_alpha = 32
    
    training_config = TrainingConfig(
        output_dir="./results/question_answering",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=3e-4,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
    )
    
    logger.info("Starting question answering example")
    
    # Load SQuAD dataset
    logger.info("Loading SQuAD dataset...")
    dataset = load_dataset_from_hub("squad")
    
    # Take smaller subsets for quick testing
    train_dataset = dataset["train"].select(range(1000))
    eval_dataset = dataset["validation"].select(range(200))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Setup trainer
    trainer = PEFTTrainer(
        model_config=model_config,
        peft_config=peft_config,
        training_config=training_config,
        task_type="question_answering"
    )
    
    # Setup preprocessor
    model_wrapper = trainer.setup_model()
    tokenizer = model_wrapper.get_tokenizer()
    
    preprocessor = QuestionAnsweringPreprocessor(
        tokenizer=tokenizer,
        question_column="question",
        context_column="context",
        answer_column="answers",
        max_length=model_config.max_length,
        doc_stride=128,
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
    
    # Test questions and contexts
    test_qa_pairs = [
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital and largest city is Paris, which is located in the north-central part of the country. Paris is known for its art, fashion, and culture."
        },
        {
            "question": "When was the Eiffel Tower built?",
            "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists and intellectuals."
        },
        {
            "question": "What is machine learning?",
            "context": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
        },
        {
            "question": "How many legs does a spider have?",
            "context": "Spiders are arachnids, not insects. They have eight legs, unlike insects which have six legs. Spiders also have two body segments: the cephalothorax and the abdomen. Most spiders have eight eyes, though some have fewer."
        }
    ]
    
    # Get answers
    logger.info("Getting answers...")
    for i, qa_pair in enumerate(test_qa_pairs):
        try:
            answer = inference_pipeline.answer_question(
                questions=qa_pair["question"],
                contexts=qa_pair["context"],
                max_answer_length=50,
            )
            
            logger.info(f"Question {i+1}: {qa_pair['question']}")
            logger.info(f"Context: {qa_pair['context'][:100]}...")
            logger.info(f"Answer: {answer['answer']}")
            logger.info(f"Confidence: {answer['score']:.4f}")
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}")
    
    # Test batch question answering
    logger.info("Testing batch question answering...")
    
    questions = [qa["question"] for qa in test_qa_pairs]
    contexts = [qa["context"] for qa in test_qa_pairs]
    
    try:
        batch_answers = inference_pipeline.answer_question(
            questions=questions,
            contexts=contexts,
        )
        
        logger.info("Batch answers:")
        for i, (qa_pair, answer) in enumerate(zip(test_qa_pairs, batch_answers)):
            logger.info(f"Q{i+1}: {qa_pair['question']}")
            logger.info(f"A{i+1}: {answer['answer']} (score: {answer['score']:.4f})")
            logger.info("-" * 30)
            
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
    
    # Detailed evaluation on SQuAD
    logger.info("Performing detailed evaluation on SQuAD...")
    
    evaluator = ModelEvaluator(
        model_wrapper=model_wrapper,
        task_type="question_answering"
    )
    
    # Note: For QA evaluation, we would need to implement specific QA metrics
    # like exact match and F1 score. For now, we'll use the basic evaluation.
    
    try:
        detailed_results = evaluator.evaluate_dataset(
            eval_dataset.select(range(50)),  # Small subset for quick evaluation
            preprocessor=preprocessor,
            save_predictions=True,
            output_dir=os.path.join(training_config.output_dir, "evaluation")
        )
        
        logger.info(f"Detailed evaluation completed")
        
    except Exception as e:
        logger.error(f"Error in detailed evaluation: {e}")
    
    # Test with custom context
    logger.info("Testing with custom context...")
    
    custom_context = """
    The transformer architecture was introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017.
    It revolutionized natural language processing by using self-attention mechanisms instead of recurrent or convolutional layers.
    The transformer consists of an encoder and decoder, each made up of multiple layers.
    Each layer contains multi-head self-attention and position-wise feed-forward networks.
    Popular models like BERT, GPT, and T5 are all based on the transformer architecture.
    """
    
    custom_questions = [
        "When was the transformer architecture introduced?",
        "Who introduced the transformer architecture?",
        "What does the transformer use instead of recurrent layers?",
        "What are some popular models based on transformers?",
        "What components make up each transformer layer?"
    ]
    
    logger.info("Custom Q&A session:")
    for question in custom_questions:
        try:
            answer = inference_pipeline.answer_question(
                questions=question,
                contexts=custom_context,
            )
            
            logger.info(f"Q: {question}")
            logger.info(f"A: {answer['answer']} (confidence: {answer['score']:.4f})")
            logger.info("-" * 40)
            
        except Exception as e:
            logger.error(f"Error answering question '{question}': {e}")
    
    logger.info("Question answering example completed!")


if __name__ == "__main__":
    main()
