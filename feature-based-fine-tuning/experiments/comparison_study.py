"""
Comprehensive Comparison Study: Feature-Based vs Full Fine-Tuning

This script compares feature-based fine-tuning with full fine-tuning across
multiple dimensions: training time, memory usage, data efficiency, and performance.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging
from sklearn.metrics import accuracy_score, f1_score
import gc

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.feature_based_model import FeatureBasedModel
from training.trainer import FeatureBasedTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComparisonStudy:
    """
    Comprehensive comparison between feature-based and full fine-tuning
    """
    
    def __init__(
        self,
        model_names: List[str] = ['distilbert-base-uncased', 'bert-base-uncased'],
        dataset_sizes: List[int] = [100, 500, 1000, 5000],
        num_classes: int = 3
    ):
        self.model_names = model_names
        self.dataset_sizes = dataset_sizes
        self.num_classes = num_classes
        self.results = []
        
        logger.info(f"Initialized comparison study")
        logger.info(f"Models: {model_names}")
        logger.info(f"Dataset sizes: {dataset_sizes}")
    
    def create_synthetic_dataset(self, size: int, vocab_size: int = 1000, seq_length: int = 128) -> Dataset:
        """Create synthetic dataset for comparison"""
        
        class SyntheticDataset(Dataset):
            def __init__(self, size, vocab_size, seq_length, num_classes):
                self.size = size
                self.vocab_size = vocab_size
                self.seq_length = seq_length
                self.num_classes = num_classes
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randint(0, self.vocab_size, (self.seq_length,)),
                    'attention_mask': torch.ones(self.seq_length),
                    'labels': torch.randint(0, self.num_classes, ())
                }
        
        return SyntheticDataset(size, vocab_size, seq_length, self.num_classes)
    
    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def measure_gpu_memory(self) -> float:
        """Measure GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def run_single_experiment(
        self,
        model_name: str,
        dataset_size: int,
        approach: str,  # 'feature_based' or 'full_finetuning'
        num_epochs: int = 3
    ) -> Dict[str, Any]:
        """Run a single experiment and measure metrics"""
        
        logger.info(f"Running {approach} with {model_name} on {dataset_size} samples")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Create dataset
        train_dataset = self.create_synthetic_dataset(dataset_size)
        eval_dataset = self.create_synthetic_dataset(min(200, dataset_size // 4))
        
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=16)
        
        # Create model
        freeze_backbone = (approach == 'feature_based')
        model = FeatureBasedModel.from_pretrained(
            model_name,
            num_classes=self.num_classes,
            freeze_backbone=freeze_backbone
        )
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Measure initial memory
        initial_memory = self.measure_memory_usage()
        initial_gpu_memory = self.measure_gpu_memory()
        
        # Setup training configuration
        learning_rate = 1e-3 if approach == 'feature_based' else 2e-5
        config = {
            'learning_rate': learning_rate,
            'weight_decay': 0.01,
            'optimizer': 'adamw',
            'scheduler': 'cosine'
        }
        
        # Initialize trainer
        trainer = FeatureBasedTrainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            config=config
        )
        
        # Measure training time and memory
        start_time = time.time()
        peak_memory = initial_memory
        peak_gpu_memory = initial_gpu_memory
        
        # Custom training loop with memory monitoring
        for epoch in range(num_epochs):
            trainer.current_epoch = epoch
            
            # Train epoch
            train_metrics = trainer.train_epoch()
            
            # Evaluate
            eval_metrics = trainer.evaluate()
            
            # Monitor memory
            current_memory = self.measure_memory_usage()
            current_gpu_memory = self.measure_gpu_memory()
            
            peak_memory = max(peak_memory, current_memory)
            peak_gpu_memory = max(peak_gpu_memory, current_gpu_memory)
        
        training_time = time.time() - start_time
        
        # Get final metrics
        final_eval_metrics = trainer.evaluate()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Cleanup
        del model, trainer, train_dataloader, eval_dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'model_name': model_name,
            'dataset_size': dataset_size,
            'approach': approach,
            'training_time': training_time,
            'peak_memory_mb': peak_memory - initial_memory,
            'peak_gpu_memory_mb': peak_gpu_memory - initial_gpu_memory,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'final_accuracy': final_eval_metrics.get('eval_accuracy', 0.0),
            'final_f1': final_eval_metrics.get('eval_f1', 0.0),
            'final_loss': final_eval_metrics.get('eval_loss', 0.0)
        }
    
    def run_full_comparison(self, num_epochs: int = 3) -> pd.DataFrame:
        """Run full comparison study"""
        
        logger.info("Starting full comparison study")
        
        results = []
        
        for model_name in self.model_names:
            for dataset_size in self.dataset_sizes:
                # Run feature-based approach
                try:
                    feature_result = self.run_single_experiment(
                        model_name, dataset_size, 'feature_based', num_epochs
                    )
                    results.append(feature_result)
                except Exception as e:
                    logger.error(f"Feature-based experiment failed: {e}")
                
                # Run full fine-tuning approach
                try:
                    full_result = self.run_single_experiment(
                        model_name, dataset_size, 'full_finetuning', num_epochs
                    )
                    results.append(full_result)
                except Exception as e:
                    logger.error(f"Full fine-tuning experiment failed: {e}")
        
        self.results = results
        return pd.DataFrame(results)
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze comparison results"""
        
        analysis = {}
        
        # Group by approach
        feature_based = results_df[results_df['approach'] == 'feature_based']
        full_finetuning = results_df[results_df['approach'] == 'full_finetuning']
        
        # Training time comparison
        avg_time_feature = feature_based['training_time'].mean()
        avg_time_full = full_finetuning['training_time'].mean()
        time_speedup = avg_time_full / avg_time_feature if avg_time_feature > 0 else 0
        
        # Memory usage comparison
        avg_memory_feature = feature_based['peak_memory_mb'].mean()
        avg_memory_full = full_finetuning['peak_memory_mb'].mean()
        memory_reduction = avg_memory_full / avg_memory_feature if avg_memory_feature > 0 else 0
        
        # Parameter comparison
        avg_trainable_feature = feature_based['trainable_params'].mean()
        avg_trainable_full = full_finetuning['trainable_params'].mean()
        param_reduction = avg_trainable_full / avg_trainable_feature if avg_trainable_feature > 0 else 0
        
        # Performance comparison
        avg_acc_feature = feature_based['final_accuracy'].mean()
        avg_acc_full = full_finetuning['final_accuracy'].mean()
        accuracy_diff = avg_acc_full - avg_acc_feature
        
        analysis = {
            'training_time': {
                'feature_based_avg': avg_time_feature,
                'full_finetuning_avg': avg_time_full,
                'speedup': time_speedup
            },
            'memory_usage': {
                'feature_based_avg': avg_memory_feature,
                'full_finetuning_avg': avg_memory_full,
                'reduction': memory_reduction
            },
            'parameters': {
                'feature_based_avg': avg_trainable_feature,
                'full_finetuning_avg': avg_trainable_full,
                'reduction': param_reduction
            },
            'performance': {
                'feature_based_accuracy': avg_acc_feature,
                'full_finetuning_accuracy': avg_acc_full,
                'accuracy_difference': accuracy_diff
            }
        }
        
        return analysis
    
    def plot_comparison_results(self, results_df: pd.DataFrame, save_path: str = None):
        """Plot comprehensive comparison results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Training Time vs Dataset Size
        for approach in ['feature_based', 'full_finetuning']:
            data = results_df[results_df['approach'] == approach]
            axes[0, 0].plot(data['dataset_size'], data['training_time'], 
                           marker='o', label=approach.replace('_', ' ').title())
        
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('Training Time (seconds)')
        axes[0, 0].set_title('Training Time vs Dataset Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Memory Usage vs Dataset Size
        for approach in ['feature_based', 'full_finetuning']:
            data = results_df[results_df['approach'] == approach]
            axes[0, 1].plot(data['dataset_size'], data['peak_memory_mb'], 
                           marker='s', label=approach.replace('_', ' ').title())
        
        axes[0, 1].set_xlabel('Dataset Size')
        axes[0, 1].set_ylabel('Peak Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage vs Dataset Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Accuracy vs Dataset Size
        for approach in ['feature_based', 'full_finetuning']:
            data = results_df[results_df['approach'] == approach]
            axes[0, 2].plot(data['dataset_size'], data['final_accuracy'], 
                           marker='^', label=approach.replace('_', ' ').title())
        
        axes[0, 2].set_xlabel('Dataset Size')
        axes[0, 2].set_ylabel('Final Accuracy')
        axes[0, 2].set_title('Accuracy vs Dataset Size')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Trainable Parameters Comparison
        approaches = results_df['approach'].unique()
        param_data = [results_df[results_df['approach'] == app]['trainable_params'].values 
                     for app in approaches]
        
        axes[1, 0].boxplot(param_data, labels=[app.replace('_', ' ').title() for app in approaches])
        axes[1, 0].set_ylabel('Trainable Parameters')
        axes[1, 0].set_title('Trainable Parameters Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Training Time Distribution
        time_data = [results_df[results_df['approach'] == app]['training_time'].values 
                    for app in approaches]
        
        axes[1, 1].boxplot(time_data, labels=[app.replace('_', ' ').title() for app in approaches])
        axes[1, 1].set_ylabel('Training Time (seconds)')
        axes[1, 1].set_title('Training Time Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Performance vs Efficiency Scatter
        feature_data = results_df[results_df['approach'] == 'feature_based']
        full_data = results_df[results_df['approach'] == 'full_finetuning']
        
        axes[1, 2].scatter(feature_data['training_time'], feature_data['final_accuracy'], 
                          label='Feature-Based', alpha=0.7, s=60)
        axes[1, 2].scatter(full_data['training_time'], full_data['final_accuracy'], 
                          label='Full Fine-tuning', alpha=0.7, s=60)
        
        axes[1, 2].set_xlabel('Training Time (seconds)')
        axes[1, 2].set_ylabel('Final Accuracy')
        axes[1, 2].set_title('Performance vs Efficiency')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plots saved to {save_path}")
        
        plt.show()
    
    def print_summary(self, analysis: Dict[str, Any]):
        """Print summary of comparison results"""
        
        print("\n" + "="*80)
        print("FEATURE-BASED vs FULL FINE-TUNING COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\n📊 TRAINING EFFICIENCY:")
        print(f"  Feature-Based Average Time: {analysis['training_time']['feature_based_avg']:.2f}s")
        print(f"  Full Fine-tuning Average Time: {analysis['training_time']['full_finetuning_avg']:.2f}s")
        print(f"  ⚡ Speedup: {analysis['training_time']['speedup']:.1f}x faster")
        
        print(f"\n💾 MEMORY EFFICIENCY:")
        print(f"  Feature-Based Average Memory: {analysis['memory_usage']['feature_based_avg']:.1f}MB")
        print(f"  Full Fine-tuning Average Memory: {analysis['memory_usage']['full_finetuning_avg']:.1f}MB")
        print(f"  📉 Memory Reduction: {analysis['memory_usage']['reduction']:.1f}x less")
        
        print(f"\n🔧 PARAMETER EFFICIENCY:")
        print(f"  Feature-Based Trainable Params: {analysis['parameters']['feature_based_avg']:,.0f}")
        print(f"  Full Fine-tuning Trainable Params: {analysis['parameters']['full_finetuning_avg']:,.0f}")
        print(f"  📊 Parameter Reduction: {analysis['parameters']['reduction']:.1f}x fewer")
        
        print(f"\n🎯 PERFORMANCE:")
        print(f"  Feature-Based Accuracy: {analysis['performance']['feature_based_accuracy']:.4f}")
        print(f"  Full Fine-tuning Accuracy: {analysis['performance']['full_finetuning_accuracy']:.4f}")
        print(f"  📈 Accuracy Difference: {analysis['performance']['accuracy_difference']:.4f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if analysis['performance']['accuracy_difference'] < 0.05:
            print("  ✅ Feature-based fine-tuning is recommended for most use cases")
            print("  ✅ Significant efficiency gains with minimal performance loss")
        else:
            print("  ⚠️  Consider full fine-tuning if maximum performance is critical")
            print("  ⚠️  Feature-based approach trades some accuracy for efficiency")


def main():
    """Run comprehensive comparison study"""
    
    logger.info("Starting comprehensive comparison study")
    
    # Initialize comparison study
    study = ComparisonStudy(
        model_names=['distilbert-base-uncased'],  # Start with one model for faster testing
        dataset_sizes=[100, 500, 1000],  # Smaller sizes for faster testing
        num_classes=3
    )
    
    # Run comparison
    results_df = study.run_full_comparison(num_epochs=2)  # Fewer epochs for faster testing
    
    # Save results
    results_df.to_csv('comparison_results.csv', index=False)
    logger.info("Results saved to comparison_results.csv")
    
    # Analyze results
    analysis = study.analyze_results(results_df)
    
    # Print summary
    study.print_summary(analysis)
    
    # Plot results
    study.plot_comparison_results(results_df, 'comparison_plots.png')
    
    logger.info("Comparison study completed successfully")


if __name__ == "__main__":
    main()
