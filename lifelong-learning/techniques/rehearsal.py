"""
Experience Replay and Memory Buffer implementation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import random
import numpy as np
import logging
from collections import defaultdict, deque
import copy

logger = logging.getLogger(__name__)


class MemoryBuffer:
    """Memory buffer for storing and sampling examples"""
    
    def __init__(
        self,
        memory_size: int,
        sampling_strategy: str = "random",
        balanced_sampling: bool = True
    ):
        self.memory_size = memory_size
        self.sampling_strategy = sampling_strategy
        self.balanced_sampling = balanced_sampling
        
        # Storage
        self.examples = []
        self.labels = []
        self.task_ids = []
        self.metadata = []  # Additional info like uncertainty, gradients
        
        # Tracking
        self.current_size = 0
        self.insertion_index = 0
        
        # For balanced sampling
        self.class_examples = defaultdict(list)
        self.task_examples = defaultdict(list)
        
        logger.info(f"Memory buffer initialized with size {memory_size}")
    
    def add_examples(
        self,
        examples: List[Any],
        labels: List[int],
        task_id: int,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add examples to memory buffer
        
        Args:
            examples: List of examples (can be tensors, dicts, etc.)
            labels: List of labels
            task_id: Task identifier
            metadata: Optional metadata for each example
        """
        if metadata is None:
            metadata = [{}] * len(examples)
        
        for example, label, meta in zip(examples, labels, metadata):
            self._add_single_example(example, label, task_id, meta)
    
    def _add_single_example(
        self,
        example: Any,
        label: int,
        task_id: int,
        metadata: Dict
    ):
        """Add single example to buffer"""
        if self.current_size < self.memory_size:
            # Buffer not full, just append
            self.examples.append(example)
            self.labels.append(label)
            self.task_ids.append(task_id)
            self.metadata.append(metadata)
            
            self.current_size += 1
            
        else:
            # Buffer full, replace based on strategy
            if self.sampling_strategy == "fifo":
                # First In First Out
                replace_idx = self.insertion_index % self.memory_size
            elif self.sampling_strategy == "reservoir":
                # Reservoir sampling
                replace_idx = random.randint(0, self.current_size)
                if replace_idx >= self.memory_size:
                    return  # Don't add this example
            else:
                # Random replacement
                replace_idx = random.randint(0, self.memory_size - 1)
            
            # Remove old example from class/task tracking
            old_label = self.labels[replace_idx]
            old_task = self.task_ids[replace_idx]
            
            if (replace_idx, old_label, old_task) in self.class_examples[old_label]:
                self.class_examples[old_label].remove((replace_idx, old_label, old_task))
            if (replace_idx, old_label, old_task) in self.task_examples[old_task]:
                self.task_examples[old_task].remove((replace_idx, old_label, old_task))
            
            # Replace with new example
            self.examples[replace_idx] = example
            self.labels[replace_idx] = label
            self.task_ids[replace_idx] = task_id
            self.metadata[replace_idx] = metadata
        
        # Update tracking
        self.class_examples[label].append((self.insertion_index % self.memory_size, label, task_id))
        self.task_examples[task_id].append((self.insertion_index % self.memory_size, label, task_id))
        
        self.insertion_index += 1
    
    def sample(
        self,
        batch_size: int,
        task_id: Optional[int] = None,
        exclude_task: Optional[int] = None
    ) -> Tuple[List[Any], List[int], List[int]]:
        """
        Sample examples from memory buffer
        
        Args:
            batch_size: Number of examples to sample
            task_id: Sample only from specific task (optional)
            exclude_task: Exclude specific task (optional)
            
        Returns:
            Tuple of (examples, labels, task_ids)
        """
        if self.current_size == 0:
            return [], [], []
        
        # Determine available indices
        available_indices = list(range(self.current_size))
        
        if task_id is not None:
            # Sample only from specific task
            task_indices = [idx for idx, label, tid in self.task_examples[task_id]]
            available_indices = [idx for idx in available_indices if idx in task_indices]
        
        if exclude_task is not None:
            # Exclude specific task
            exclude_indices = [idx for idx, label, tid in self.task_examples[exclude_task]]
            available_indices = [idx for idx in available_indices if idx not in exclude_indices]
        
        if not available_indices:
            return [], [], []
        
        # Sample indices
        if self.balanced_sampling and task_id is None:
            sampled_indices = self._balanced_sample(batch_size, available_indices)
        else:
            sampled_indices = self._random_sample(batch_size, available_indices)
        
        # Get examples
        sampled_examples = [self.examples[idx] for idx in sampled_indices]
        sampled_labels = [self.labels[idx] for idx in sampled_indices]
        sampled_task_ids = [self.task_ids[idx] for idx in sampled_indices]
        
        return sampled_examples, sampled_labels, sampled_task_ids
    
    def _random_sample(self, batch_size: int, available_indices: List[int]) -> List[int]:
        """Random sampling"""
        return random.choices(available_indices, k=min(batch_size, len(available_indices)))
    
    def _balanced_sample(self, batch_size: int, available_indices: List[int]) -> List[int]:
        """Balanced sampling across classes"""
        # Get available classes
        available_classes = set()
        for idx in available_indices:
            available_classes.add(self.labels[idx])
        
        available_classes = list(available_classes)
        
        if not available_classes:
            return []
        
        # Sample equally from each class
        samples_per_class = batch_size // len(available_classes)
        remaining_samples = batch_size % len(available_classes)
        
        sampled_indices = []
        
        for i, class_label in enumerate(available_classes):
            # Get indices for this class
            class_indices = [idx for idx in available_indices if self.labels[idx] == class_label]
            
            # Determine number of samples for this class
            num_samples = samples_per_class
            if i < remaining_samples:
                num_samples += 1
            
            # Sample from this class
            class_samples = random.choices(class_indices, k=min(num_samples, len(class_indices)))
            sampled_indices.extend(class_samples)
        
        return sampled_indices
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory buffer statistics"""
        stats = {
            "current_size": self.current_size,
            "memory_size": self.memory_size,
            "utilization": self.current_size / self.memory_size,
            "num_tasks": len(self.task_examples),
            "num_classes": len(self.class_examples)
        }
        
        # Class distribution
        class_counts = {label: len(indices) for label, indices in self.class_examples.items()}
        stats["class_distribution"] = class_counts
        
        # Task distribution
        task_counts = {task_id: len(indices) for task_id, indices in self.task_examples.items()}
        stats["task_distribution"] = task_counts
        
        return stats
    
    def clear(self):
        """Clear memory buffer"""
        self.examples.clear()
        self.labels.clear()
        self.task_ids.clear()
        self.metadata.clear()
        self.class_examples.clear()
        self.task_examples.clear()
        
        self.current_size = 0
        self.insertion_index = 0
        
        logger.info("Memory buffer cleared")


class ExperienceReplay:
    """Experience Replay for continual learning"""
    
    def __init__(
        self,
        memory_size: int,
        sampling_strategy: str = "random",
        replay_frequency: int = 1,
        balanced_replay: bool = True,
        replay_batch_size: Optional[int] = None
    ):
        self.memory_size = memory_size
        self.sampling_strategy = sampling_strategy
        self.replay_frequency = replay_frequency
        self.balanced_replay = balanced_replay
        self.replay_batch_size = replay_batch_size
        
        # Memory buffer
        self.memory_buffer = MemoryBuffer(
            memory_size=memory_size,
            sampling_strategy=sampling_strategy,
            balanced_sampling=balanced_replay
        )
        
        # Tracking
        self.step_count = 0
        self.replay_count = 0
        
        logger.info("Experience Replay initialized")
    
    def store_examples(
        self,
        examples: List[Any],
        labels: List[int],
        task_id: int,
        metadata: Optional[List[Dict]] = None
    ):
        """Store examples in memory buffer"""
        self.memory_buffer.add_examples(examples, labels, task_id, metadata)
        
        logger.debug(f"Stored {len(examples)} examples for task {task_id}")
    
    def should_replay(self) -> bool:
        """Determine if replay should happen this step"""
        self.step_count += 1
        return (self.step_count % self.replay_frequency == 0 and 
                self.memory_buffer.current_size > 0)
    
    def get_replay_batch(
        self,
        batch_size: Optional[int] = None,
        current_task_id: Optional[int] = None,
        exclude_current_task: bool = False
    ) -> Tuple[List[Any], List[int], List[int]]:
        """
        Get batch for replay
        
        Args:
            batch_size: Size of replay batch
            current_task_id: Current task ID
            exclude_current_task: Whether to exclude current task from replay
            
        Returns:
            Tuple of (examples, labels, task_ids)
        """
        if batch_size is None:
            batch_size = self.replay_batch_size or 32
        
        exclude_task = current_task_id if exclude_current_task else None
        
        examples, labels, task_ids = self.memory_buffer.sample(
            batch_size=batch_size,
            exclude_task=exclude_task
        )
        
        if examples:
            self.replay_count += 1
            logger.debug(f"Replay batch {self.replay_count}: {len(examples)} examples")
        
        return examples, labels, task_ids
    
    def get_mixed_batch(
        self,
        current_examples: List[Any],
        current_labels: List[int],
        current_task_id: int,
        replay_ratio: float = 0.5
    ) -> Tuple[List[Any], List[int], List[int]]:
        """
        Get mixed batch of current and replay examples
        
        Args:
            current_examples: Examples from current task
            current_labels: Labels from current task
            current_task_id: Current task ID
            replay_ratio: Ratio of replay examples in mixed batch
            
        Returns:
            Mixed batch of (examples, labels, task_ids)
        """
        current_batch_size = len(current_examples)
        replay_batch_size = int(current_batch_size * replay_ratio / (1 - replay_ratio))
        
        # Get replay examples
        replay_examples, replay_labels, replay_task_ids = self.get_replay_batch(
            batch_size=replay_batch_size,
            current_task_id=current_task_id,
            exclude_current_task=True
        )
        
        # Combine batches
        mixed_examples = current_examples + replay_examples
        mixed_labels = current_labels + replay_labels
        mixed_task_ids = [current_task_id] * len(current_examples) + replay_task_ids
        
        # Shuffle
        combined = list(zip(mixed_examples, mixed_labels, mixed_task_ids))
        random.shuffle(combined)
        
        if combined:
            mixed_examples, mixed_labels, mixed_task_ids = zip(*combined)
            return list(mixed_examples), list(mixed_labels), list(mixed_task_ids)
        else:
            return current_examples, current_labels, [current_task_id] * len(current_examples)
    
    def update_memory_strategy(
        self,
        examples: List[Any],
        labels: List[int],
        task_id: int,
        model: nn.Module,
        uncertainty_scores: Optional[List[float]] = None
    ):
        """
        Update memory with uncertainty-based or gradient-based selection
        
        Args:
            examples: Candidate examples
            labels: Candidate labels
            task_id: Task ID
            model: Model for computing uncertainty/gradients
            uncertainty_scores: Pre-computed uncertainty scores
        """
        if self.sampling_strategy == "uncertainty" and uncertainty_scores is not None:
            # Select most uncertain examples
            sorted_indices = sorted(
                range(len(uncertainty_scores)),
                key=lambda i: uncertainty_scores[i],
                reverse=True
            )
            
            # Take top uncertain examples
            num_to_store = min(len(examples), self.memory_size // 10)  # Store 10% of memory per update
            selected_indices = sorted_indices[:num_to_store]
            
            selected_examples = [examples[i] for i in selected_indices]
            selected_labels = [labels[i] for i in selected_indices]
            
            self.store_examples(selected_examples, selected_labels, task_id)
            
        elif self.sampling_strategy == "gradient":
            # Gradient-based selection (simplified)
            # In practice, this would compute gradient norms for each example
            # For now, use random selection
            num_to_store = min(len(examples), self.memory_size // 10)
            selected_indices = random.sample(range(len(examples)), num_to_store)
            
            selected_examples = [examples[i] for i in selected_indices]
            selected_labels = [labels[i] for i in selected_indices]
            
            self.store_examples(selected_examples, selected_labels, task_id)
            
        else:
            # Random selection
            self.store_examples(examples, labels, task_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get replay statistics"""
        buffer_stats = self.memory_buffer.get_statistics()
        
        replay_stats = {
            "step_count": self.step_count,
            "replay_count": self.replay_count,
            "replay_frequency": self.replay_frequency,
            "replay_ratio": self.replay_count / max(self.step_count, 1)
        }
        
        return {**buffer_stats, **replay_stats}
    
    def save_memory(self, filepath: str):
        """Save memory buffer"""
        memory_state = {
            "examples": self.memory_buffer.examples,
            "labels": self.memory_buffer.labels,
            "task_ids": self.memory_buffer.task_ids,
            "metadata": self.memory_buffer.metadata,
            "current_size": self.memory_buffer.current_size,
            "insertion_index": self.memory_buffer.insertion_index
        }
        
        torch.save(memory_state, filepath)
        logger.info(f"Memory buffer saved to {filepath}")
    
    def load_memory(self, filepath: str):
        """Load memory buffer"""
        memory_state = torch.load(filepath, map_location="cpu")
        
        self.memory_buffer.examples = memory_state["examples"]
        self.memory_buffer.labels = memory_state["labels"]
        self.memory_buffer.task_ids = memory_state["task_ids"]
        self.memory_buffer.metadata = memory_state["metadata"]
        self.memory_buffer.current_size = memory_state["current_size"]
        self.memory_buffer.insertion_index = memory_state["insertion_index"]
        
        # Rebuild tracking dictionaries
        self.memory_buffer.class_examples.clear()
        self.memory_buffer.task_examples.clear()
        
        for idx in range(self.memory_buffer.current_size):
            label = self.memory_buffer.labels[idx]
            task_id = self.memory_buffer.task_ids[idx]
            
            self.memory_buffer.class_examples[label].append((idx, label, task_id))
            self.memory_buffer.task_examples[task_id].append((idx, label, task_id))
        
        logger.info(f"Memory buffer loaded from {filepath}")
    
    def reset(self):
        """Reset replay system"""
        self.memory_buffer.clear()
        self.step_count = 0
        self.replay_count = 0
        
        logger.info("Experience replay reset")
