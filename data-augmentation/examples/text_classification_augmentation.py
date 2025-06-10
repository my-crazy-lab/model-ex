"""
Complete example: Text classification with data augmentation
"""

import argparse
import logging
import json
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

from config import AugmentationConfig, GenerationConfig, QualityConfig
from augmentation import TextAugmenter, LLMGenerator
from quality import QualityMetrics
from training import AugmentedTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_dataset(dataset_name: str = "imdb", sample_size: int = 1000):
    """Load and sample dataset for demonstration"""
    
    logger.info(f"Loading {dataset_name} dataset...")
    
    if dataset_name == "imdb":
        dataset = load_dataset("imdb")
        
        # Sample data for quick demonstration
        train_data = dataset["train"].select(range(min(sample_size, len(dataset["train"]))))
        test_data = dataset["test"].select(range(min(sample_size // 4, len(dataset["test"]))))
        
        return {
            "train_texts": train_data["text"],
            "train_labels": train_data["label"],
            "test_texts": test_data["text"],
            "test_labels": test_data["label"]
        }
    
    else:
        # Create dummy dataset
        logger.info("Creating dummy dataset...")
        texts = [
            "This movie is great!",
            "I love this film.",
            "Amazing acting and plot.",
            "Terrible movie, waste of time.",
            "Boring and poorly made.",
            "Worst film ever made."
        ] * (sample_size // 6)
        
        labels = [1, 1, 1, 0, 0, 0] * (sample_size // 6)
        
        # Split into train/test
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        return {
            "train_texts": train_texts,
            "train_labels": train_labels,
            "test_texts": test_texts,
            "test_labels": test_labels
        }


def augment_with_rules(texts, labels, config):
    """Augment data using rule-based methods"""
    
    logger.info("Applying rule-based augmentation...")
    
    augmenter = TextAugmenter(config)
    
    # Augment dataset
    augmented_data = augmenter.augment_dataset(texts, labels)
    
    logger.info(f"Original dataset size: {len(texts)}")
    logger.info(f"Augmented dataset size: {len(augmented_data['texts'])}")
    
    return augmented_data["texts"], augmented_data["labels"]


def augment_with_llm(texts, labels, config, num_samples_per_class=50):
    """Augment data using LLM generation"""
    
    logger.info("Applying LLM-based augmentation...")
    
    generator = LLMGenerator(config)
    
    # Create prompts for each class
    class_examples = {}
    for text, label in zip(texts, labels):
        if label not in class_examples:
            class_examples[label] = []
        class_examples[label].append(text)
    
    augmented_texts = []
    augmented_labels = []
    
    for label, examples in class_examples.items():
        # Create prompt with examples
        sentiment = "positive" if label == 1 else "negative"
        example_texts = examples[:3]  # Use first 3 as examples
        
        prompt = f"""Generate {sentiment} movie reviews similar to these examples:
        
Examples:
{chr(10).join([f"- {ex}" for ex in example_texts])}

Generate a new {sentiment} movie review:"""
        
        try:
            # Generate samples
            generated = generator.generate(prompt, num_samples_per_class)
            
            augmented_texts.extend(generated)
            augmented_labels.extend([label] * len(generated))
            
            logger.info(f"Generated {len(generated)} samples for class {label}")
        
        except Exception as e:
            logger.error(f"Error generating for class {label}: {e}")
    
    # Combine with original data
    all_texts = list(texts) + augmented_texts
    all_labels = list(labels) + augmented_labels
    
    logger.info(f"Original dataset size: {len(texts)}")
    logger.info(f"Total dataset size after LLM augmentation: {len(all_texts)}")
    
    return all_texts, all_labels


def assess_and_filter_quality(texts, labels, quality_config):
    """Assess and filter data by quality"""
    
    logger.info("Assessing data quality...")
    
    quality_metrics = QualityMetrics()
    
    # Filter by quality
    filtered_texts, quality_scores = quality_metrics.filter_by_quality(
        texts,
        min_fluency=quality_config.fluency_threshold,
        min_coherence=0.5,
        min_grammaticality=0.5,
        min_overall=0.6
    )
    
    # Filter corresponding labels
    filtered_labels = []
    original_indices = []
    
    for i, text in enumerate(texts):
        if text in filtered_texts:
            filtered_labels.append(labels[i])
            original_indices.append(i)
    
    logger.info(f"Original size: {len(texts)}")
    logger.info(f"Filtered size: {len(filtered_texts)}")
    logger.info(f"Quality filter ratio: {len(filtered_texts)/len(texts):.2%}")
    
    # Print quality statistics
    if quality_scores:
        avg_fluency = sum(s["fluency"] for s in quality_scores) / len(quality_scores)
        avg_coherence = sum(s["coherence"] for s in quality_scores) / len(quality_scores)
        avg_overall = sum(s["overall"] for s in quality_scores) / len(quality_scores)
        
        logger.info(f"Average quality scores:")
        logger.info(f"  Fluency: {avg_fluency:.3f}")
        logger.info(f"  Coherence: {avg_coherence:.3f}")
        logger.info(f"  Overall: {avg_overall:.3f}")
    
    return filtered_texts, filtered_labels


def train_and_evaluate(train_texts, train_labels, test_texts, test_labels, model_name="distilbert-base-uncased"):
    """Train and evaluate model"""
    
    logger.info("Training and evaluating model...")
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "labels": train_labels
    })
    
    test_dataset = Dataset.from_dict({
        "text": test_texts,
        "labels": test_labels
    })
    
    # Simple evaluation using sklearn (placeholder)
    # In practice, you'd use transformers Trainer
    
    # For demonstration, use simple baseline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    # Vectorize texts
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # Train classifier
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, train_labels)
    
    # Evaluate
    train_pred = classifier.predict(X_train)
    test_pred = classifier.predict(X_test)
    
    train_acc = accuracy_score(train_labels, train_pred)
    test_acc = accuracy_score(test_labels, test_pred)
    
    logger.info(f"Training accuracy: {train_acc:.3f}")
    logger.info(f"Test accuracy: {test_acc:.3f}")
    
    # Detailed classification report
    report = classification_report(test_labels, test_pred, output_dict=True)
    
    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "classification_report": report
    }


def main():
    """Main augmentation example"""
    
    parser = argparse.ArgumentParser(description="Text Classification Augmentation Example")
    parser.add_argument("--dataset", default="dummy", choices=["imdb", "dummy"], help="Dataset to use")
    parser.add_argument("--sample_size", type=int, default=500, help="Sample size for demonstration")
    parser.add_argument("--augmentation_method", default="rules", choices=["rules", "llm", "both"], help="Augmentation method")
    parser.add_argument("--output_dir", default="./augmentation_results", help="Output directory")
    parser.add_argument("--use_quality_filter", action="store_true", help="Apply quality filtering")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting text classification augmentation example...")
    
    # Load dataset
    data = load_sample_dataset(args.dataset, args.sample_size)
    
    # Original performance
    logger.info("📊 Evaluating baseline (no augmentation)...")
    baseline_results = train_and_evaluate(
        data["train_texts"], data["train_labels"],
        data["test_texts"], data["test_labels"]
    )
    
    # Setup configurations
    aug_config = AugmentationConfig(
        synonym_replacement_prob=0.1,
        random_insertion_prob=0.1,
        random_deletion_prob=0.1,
        random_swap_prob=0.1,
        num_augmentations_per_sample=2
    )
    
    gen_config = GenerationConfig(
        model_name="gpt-3.5-turbo",  # Change to local model if no API key
        temperature=0.8,
        max_tokens=100,
        num_samples=3
    )
    
    quality_config = QualityConfig(
        fluency_threshold=0.7,
        diversity_threshold=0.5
    )
    
    # Apply augmentation
    if args.augmentation_method in ["rules", "both"]:
        logger.info("🔄 Applying rule-based augmentation...")
        aug_texts, aug_labels = augment_with_rules(
            data["train_texts"], data["train_labels"], aug_config
        )
    else:
        aug_texts, aug_labels = data["train_texts"], data["train_labels"]
    
    if args.augmentation_method in ["llm", "both"]:
        logger.info("🤖 Applying LLM-based augmentation...")
        try:
            aug_texts, aug_labels = augment_with_llm(
                aug_texts, aug_labels, gen_config, num_samples_per_class=20
            )
        except Exception as e:
            logger.error(f"LLM augmentation failed: {e}")
            logger.info("Continuing with rule-based augmentation only...")
    
    # Quality filtering
    if args.use_quality_filter:
        logger.info("🔍 Applying quality filtering...")
        aug_texts, aug_labels = assess_and_filter_quality(
            aug_texts, aug_labels, quality_config
        )
    
    # Evaluate augmented model
    logger.info("📈 Evaluating augmented model...")
    augmented_results = train_and_evaluate(
        aug_texts, aug_labels,
        data["test_texts"], data["test_labels"]
    )
    
    # Compare results
    logger.info("\n" + "="*50)
    logger.info("📊 RESULTS COMPARISON")
    logger.info("="*50)
    logger.info(f"Baseline accuracy: {baseline_results['test_accuracy']:.3f}")
    logger.info(f"Augmented accuracy: {augmented_results['test_accuracy']:.3f}")
    
    improvement = augmented_results['test_accuracy'] - baseline_results['test_accuracy']
    logger.info(f"Improvement: {improvement:+.3f} ({improvement/baseline_results['test_accuracy']*100:+.1f}%)")
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        "baseline": baseline_results,
        "augmented": augmented_results,
        "improvement": improvement,
        "config": {
            "augmentation_method": args.augmentation_method,
            "sample_size": args.sample_size,
            "use_quality_filter": args.use_quality_filter
        }
    }
    
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save augmented data sample
    sample_data = {
        "original_texts": data["train_texts"][:10],
        "augmented_texts": aug_texts[:20],
        "original_labels": data["train_labels"][:10],
        "augmented_labels": aug_labels[:20]
    }
    
    with open(os.path.join(args.output_dir, "sample_data.json"), 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"✅ Results saved to {args.output_dir}")
    logger.info("🎉 Augmentation example completed!")


if __name__ == "__main__":
    main()
