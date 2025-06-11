"""
Comprehensive Performance Analysis: Full Fine-Tuning vs Feature-Based

This script provides detailed analysis of full fine-tuning performance across
multiple dimensions: accuracy, training efficiency, memory usage, and scalability.
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

from models.full_model import FullFinetuningModel
from training.trainer import FullFinetuningTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for full fine-tuning
    """
    
    def __init__(
        self,
        model_names: List[str] = ['bert-base-uncased', 'roberta-base'],
        dataset_sizes: List[int] = [500, 1000, 2000, 5000],
        num_classes: int = 3
    ):
        self.model_names = model_names
        self.dataset_sizes = dataset_sizes
        self.num_classes = num_classes
        self.results = []
        
        logger.info(f"Initialized performance analyzer")
        logger.info(f"Models: {model_names}")
        logger.info(f"Dataset sizes: {dataset_sizes}")
    
    def create_synthetic_dataset(self, size: int, vocab_size: int = 1000, seq_length: int = 128) -> Dataset:
        """Create synthetic dataset for analysis"""
        
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
    
    def run_full_finetuning_experiment(
        self,
        model_name: str,
        dataset_size: int,
        num_epochs: int = 2,
        use_advanced_features: bool = True
    ) -> Dict[str, Any]:
        """Run full fine-tuning experiment with comprehensive metrics"""
        
        logger.info(f"Running full fine-tuning: {model_name} on {dataset_size} samples")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Create dataset
        train_dataset = self.create_synthetic_dataset(dataset_size)
        eval_dataset = self.create_synthetic_dataset(min(200, dataset_size // 4))
        
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=16)
        
        # Create model with full fine-tuning
        model = FullFinetuningModel.from_pretrained(
            model_name,
            task='classification',
            num_classes=self.num_classes,
            freeze_backbone=False,  # Full fine-tuning
            gradient_checkpointing=use_advanced_features
        )
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Measure initial memory
        initial_memory = self.measure_memory_usage()
        initial_gpu_memory = self.measure_gpu_memory()
        
        # Setup advanced training configuration
        config = {
            'learning_rate': 2e-5,  # Lower LR for full fine-tuning
            'weight_decay': 0.01,
            'optimizer': 'adamw',
            'scheduler': 'linear_warmup',
            'gradient_accumulation_steps': 2 if use_advanced_features else 1,
            'mixed_precision': 'fp16' if use_advanced_features else 'no',
            'max_grad_norm': 1.0,
            'warmup_steps': 100
        }
        
        # Initialize trainer
        trainer = FullFinetuningTrainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            config=config
        )
        
        # Measure training time and memory
        start_time = time.time()
        peak_memory = initial_memory
        peak_gpu_memory = initial_gpu_memory
        
        # Training with memory monitoring
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
        
        # Measure convergence speed
        convergence_epochs = self._estimate_convergence(trainer.training_history)
        
        # Cleanup
        del model, trainer, train_dataloader, eval_dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'model_name': model_name,
            'dataset_size': dataset_size,
            'approach': 'full_finetuning',
            'training_time': training_time,
            'peak_memory_mb': peak_memory - initial_memory,
            'peak_gpu_memory_mb': peak_gpu_memory - initial_gpu_memory,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'final_accuracy': final_eval_metrics.get('eval_accuracy', 0.0),
            'final_f1': final_eval_metrics.get('eval_f1', 0.0),
            'final_loss': final_eval_metrics.get('eval_loss', 0.0),
            'convergence_epochs': convergence_epochs,
            'use_advanced_features': use_advanced_features
        }
    
    def _estimate_convergence(self, training_history: List[Dict]) -> int:
        """Estimate convergence speed from training history"""
        if not training_history:
            return 0
        
        # Look for when validation accuracy stabilizes
        eval_accuracies = [h.get('eval_accuracy', 0) for h in training_history if 'eval_accuracy' in h]
        
        if len(eval_accuracies) < 2:
            return len(training_history)
        
        # Find when improvement becomes minimal
        for i in range(1, len(eval_accuracies)):
            if i >= 3:  # Need at least 3 points
                recent_improvement = max(eval_accuracies[i-2:i+1]) - eval_accuracies[i-3]
                if recent_improvement < 0.01:  # Less than 1% improvement
                    return i
        
        return len(eval_accuracies)
    
    def run_comprehensive_analysis(self, num_epochs: int = 2) -> pd.DataFrame:
        """Run comprehensive analysis across all configurations"""
        
        logger.info("Starting comprehensive full fine-tuning analysis")
        
        results = []
        
        for model_name in self.model_names:
            for dataset_size in self.dataset_sizes:
                # Standard full fine-tuning
                try:
                    standard_result = self.run_full_finetuning_experiment(
                        model_name, dataset_size, num_epochs, use_advanced_features=False
                    )
                    results.append(standard_result)
                except Exception as e:
                    logger.error(f"Standard experiment failed: {e}")
                
                # Advanced full fine-tuning
                try:
                    advanced_result = self.run_full_finetuning_experiment(
                        model_name, dataset_size, num_epochs, use_advanced_features=True
                    )
                    advanced_result['approach'] = 'full_finetuning_advanced'
                    results.append(advanced_result)
                except Exception as e:
                    logger.error(f"Advanced experiment failed: {e}")
        
        self.results = results
        return pd.DataFrame(results)
    
    def analyze_scalability(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze scalability patterns"""
        
        scalability_analysis = {}
        
        # Training time scalability
        for approach in results_df['approach'].unique():
            approach_data = results_df[results_df['approach'] == approach]
            
            # Fit polynomial to training time vs dataset size
            if len(approach_data) > 1:
                coeffs = np.polyfit(approach_data['dataset_size'], approach_data['training_time'], 2)
                scalability_analysis[f'{approach}_time_complexity'] = coeffs
        
        # Memory scalability
        for approach in results_df['approach'].unique():
            approach_data = results_df[results_df['approach'] == approach]
            
            if len(approach_data) > 1:
                coeffs = np.polyfit(approach_data['dataset_size'], approach_data['peak_memory_mb'], 1)
                scalability_analysis[f'{approach}_memory_complexity'] = coeffs
        
        # Performance vs dataset size
        for approach in results_df['approach'].unique():
            approach_data = results_df[results_df['approach'] == approach]
            
            if len(approach_data) > 1:
                # Correlation between dataset size and accuracy
                correlation = np.corrcoef(approach_data['dataset_size'], approach_data['final_accuracy'])[0, 1]
                scalability_analysis[f'{approach}_performance_correlation'] = correlation
        
        return scalability_analysis
    
    def plot_comprehensive_analysis(self, results_df: pd.DataFrame, save_path: str = None):
        """Plot comprehensive analysis results"""
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Training Time vs Dataset Size
        for approach in results_df['approach'].unique():
            data = results_df[results_df['approach'] == approach]
            axes[0, 0].plot(data['dataset_size'], data['training_time'], 
                           marker='o', label=approach.replace('_', ' ').title())
        
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('Training Time (seconds)')
        axes[0, 0].set_title('Training Time Scalability')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Memory Usage vs Dataset Size
        for approach in results_df['approach'].unique():
            data = results_df[results_df['approach'] == approach]
            axes[0, 1].plot(data['dataset_size'], data['peak_memory_mb'], 
                           marker='s', label=approach.replace('_', ' ').title())
        
        axes[0, 1].set_xlabel('Dataset Size')
        axes[0, 1].set_ylabel('Peak Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage Scalability')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Accuracy vs Dataset Size
        for approach in results_df['approach'].unique():
            data = results_df[results_df['approach'] == approach]
            axes[0, 2].plot(data['dataset_size'], data['final_accuracy'], 
                           marker='^', label=approach.replace('_', ' ').title())
        
        axes[0, 2].set_xlabel('Dataset Size')
        axes[0, 2].set_ylabel('Final Accuracy')
        axes[0, 2].set_title('Performance vs Dataset Size')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Training Efficiency (Accuracy/Time)
        results_df['efficiency'] = results_df['final_accuracy'] / results_df['training_time']
        
        for approach in results_df['approach'].unique():
            data = results_df[results_df['approach'] == approach]
            axes[1, 0].plot(data['dataset_size'], data['efficiency'], 
                           marker='d', label=approach.replace('_', ' ').title())
        
        axes[1, 0].set_xlabel('Dataset Size')
        axes[1, 0].set_ylabel('Training Efficiency (Accuracy/Time)')
        axes[1, 0].set_title('Training Efficiency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. Convergence Speed
        for approach in results_df['approach'].unique():
            data = results_df[results_df['approach'] == approach]
            axes[1, 1].plot(data['dataset_size'], data['convergence_epochs'], 
                           marker='*', label=approach.replace('_', ' ').title())
        
        axes[1, 1].set_xlabel('Dataset Size')
        axes[1, 1].set_ylabel('Convergence Epochs')
        axes[1, 1].set_title('Convergence Speed')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 6. Model Comparison (Accuracy)
        model_comparison = results_df.groupby(['model_name', 'approach'])['final_accuracy'].mean().unstack()
        model_comparison.plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Model Comparison (Accuracy)')
        axes[1, 2].set_ylabel('Average Accuracy')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # 7. Advanced vs Standard Features
        if 'full_finetuning_advanced' in results_df['approach'].values:
            standard_data = results_df[results_df['approach'] == 'full_finetuning']
            advanced_data = results_df[results_df['approach'] == 'full_finetuning_advanced']
            
            if len(standard_data) > 0 and len(advanced_data) > 0:
                improvement = advanced_data['final_accuracy'].values - standard_data['final_accuracy'].values
                axes[2, 0].bar(range(len(improvement)), improvement)
                axes[2, 0].set_title('Advanced Features Improvement')
                axes[2, 0].set_ylabel('Accuracy Improvement')
                axes[2, 0].set_xlabel('Experiment Index')
        
        # 8. Memory Efficiency
        results_df['memory_efficiency'] = results_df['final_accuracy'] / results_df['peak_memory_mb']
        
        for approach in results_df['approach'].unique():
            data = results_df[results_df['approach'] == approach]
            axes[2, 1].plot(data['dataset_size'], data['memory_efficiency'], 
                           marker='h', label=approach.replace('_', ' ').title())
        
        axes[2, 1].set_xlabel('Dataset Size')
        axes[2, 1].set_ylabel('Memory Efficiency (Accuracy/MB)')
        axes[2, 1].set_title('Memory Efficiency')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # 9. Parameter Utilization
        results_df['param_efficiency'] = results_df['final_accuracy'] / (results_df['trainable_params'] / 1e6)
        
        for approach in results_df['approach'].unique():
            data = results_df[results_df['approach'] == approach]
            axes[2, 2].scatter(data['trainable_params'] / 1e6, data['final_accuracy'], 
                              label=approach.replace('_', ' ').title(), alpha=0.7)
        
        axes[2, 2].set_xlabel('Trainable Parameters (Millions)')
        axes[2, 2].set_ylabel('Final Accuracy')
        axes[2, 2].set_title('Parameter Efficiency')
        axes[2, 2].legend()
        axes[2, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Analysis plots saved to {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, results_df: pd.DataFrame, scalability_analysis: Dict) -> str:
        """Generate comprehensive performance report"""
        
        report = []
        report.append("="*80)
        report.append("FULL FINE-TUNING PERFORMANCE ANALYSIS REPORT")
        report.append("="*80)
        
        # Overall statistics
        report.append(f"\n📊 OVERALL STATISTICS:")
        report.append(f"  Models tested: {results_df['model_name'].nunique()}")
        report.append(f"  Dataset sizes: {sorted(results_df['dataset_size'].unique())}")
        report.append(f"  Approaches: {list(results_df['approach'].unique())}")
        report.append(f"  Total experiments: {len(results_df)}")
        
        # Performance summary
        report.append(f"\n🎯 PERFORMANCE SUMMARY:")
        for approach in results_df['approach'].unique():
            data = results_df[results_df['approach'] == approach]
            avg_accuracy = data['final_accuracy'].mean()
            std_accuracy = data['final_accuracy'].std()
            report.append(f"  {approach.replace('_', ' ').title()}:")
            report.append(f"    Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            report.append(f"    Best Accuracy: {data['final_accuracy'].max():.4f}")
            report.append(f"    Worst Accuracy: {data['final_accuracy'].min():.4f}")
        
        # Training efficiency
        report.append(f"\n⚡ TRAINING EFFICIENCY:")
        for approach in results_df['approach'].unique():
            data = results_df[results_df['approach'] == approach]
            avg_time = data['training_time'].mean()
            avg_memory = data['peak_memory_mb'].mean()
            avg_convergence = data['convergence_epochs'].mean()
            
            report.append(f"  {approach.replace('_', ' ').title()}:")
            report.append(f"    Average Training Time: {avg_time:.2f} seconds")
            report.append(f"    Average Memory Usage: {avg_memory:.1f} MB")
            report.append(f"    Average Convergence: {avg_convergence:.1f} epochs")
        
        # Scalability insights
        report.append(f"\n📈 SCALABILITY INSIGHTS:")
        for key, value in scalability_analysis.items():
            if 'correlation' in key:
                approach = key.replace('_performance_correlation', '')
                report.append(f"  {approach.replace('_', ' ').title()} Performance-Size Correlation: {value:.3f}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        
        # Find best performing approach
        best_approach = results_df.groupby('approach')['final_accuracy'].mean().idxmax()
        best_accuracy = results_df.groupby('approach')['final_accuracy'].mean().max()
        
        report.append(f"  ✅ Best Overall Approach: {best_approach.replace('_', ' ').title()}")
        report.append(f"     Average Accuracy: {best_accuracy:.4f}")
        
        # Find most efficient approach
        results_df['overall_efficiency'] = (
            results_df['final_accuracy'] / 
            (results_df['training_time'] / results_df['training_time'].max())
        )
        most_efficient = results_df.groupby('approach')['overall_efficiency'].mean().idxmax()
        
        report.append(f"  ⚡ Most Efficient Approach: {most_efficient.replace('_', ' ').title()}")
        
        # Dataset size recommendations
        large_dataset_performance = results_df[results_df['dataset_size'] >= 2000]['final_accuracy'].mean()
        small_dataset_performance = results_df[results_df['dataset_size'] < 1000]['final_accuracy'].mean()
        
        if large_dataset_performance > small_dataset_performance + 0.05:
            report.append(f"  📊 Recommendation: Use larger datasets (2000+ samples) for best results")
        else:
            report.append(f"  📊 Recommendation: Model performs well even with smaller datasets")
        
        report.append("="*80)
        
        return "\n".join(report)


def main():
    """Run comprehensive performance analysis"""
    
    logger.info("Starting comprehensive full fine-tuning performance analysis")
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(
        model_names=['bert-base-uncased'],  # Start with one model for faster testing
        dataset_sizes=[500, 1000, 2000],    # Smaller sizes for faster testing
        num_classes=3
    )
    
    # Run comprehensive analysis
    results_df = analyzer.run_comprehensive_analysis(num_epochs=2)
    
    # Save results
    results_df.to_csv('full_finetuning_analysis_results.csv', index=False)
    logger.info("Results saved to full_finetuning_analysis_results.csv")
    
    # Analyze scalability
    scalability_analysis = analyzer.analyze_scalability(results_df)
    
    # Generate report
    report = analyzer.generate_performance_report(results_df, scalability_analysis)
    print(report)
    
    # Save report
    with open('full_finetuning_performance_report.txt', 'w') as f:
        f.write(report)
    
    # Plot comprehensive analysis
    analyzer.plot_comprehensive_analysis(results_df, 'full_finetuning_analysis.png')
    
    logger.info("Comprehensive performance analysis completed successfully")


if __name__ == "__main__":
    main()
