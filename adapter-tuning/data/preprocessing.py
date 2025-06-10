"""
Data preprocessing utilities for Adapter Tuning implementation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor(ABC):
    """Abstract base class for data preprocessing"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    @abstractmethod
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess a batch of examples"""
        pass
    
    def preprocess_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        batched: bool = True,
        num_proc: Optional[int] = None,
        remove_columns: Optional[List[str]] = None
    ) -> Union[Dataset, DatasetDict]:
        """Apply preprocessing to entire dataset"""
        
        def apply_preprocessing(ds):
            return ds.map(
                self.preprocess_function,
                batched=batched,
                num_proc=num_proc,
                remove_columns=remove_columns,
                desc="Preprocessing dataset"
            )
        
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: apply_preprocessing(ds)
                for split, ds in dataset.items()
            })
        else:
            return apply_preprocessing(dataset)


class TextClassificationPreprocessor(DataPreprocessor):
    """Preprocessor for text classification tasks"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "text",
        label_column: str = "label",
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True
    ):
        super().__init__(tokenizer, max_length, padding, truncation)
        self.text_column = text_column
        self.label_column = label_column
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess text classification examples"""
        # Tokenize the texts
        result = self.tokenizer(
            examples[self.text_column],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=None
        )
        
        # Add labels if present
        if self.label_column in examples:
            result["labels"] = examples[self.label_column]
        
        return result


class TokenClassificationPreprocessor(DataPreprocessor):
    """Preprocessor for token classification tasks (NER, POS tagging)"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "tokens",
        label_column: str = "ner_tags",
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        label_all_tokens: bool = False
    ):
        super().__init__(tokenizer, max_length, padding, truncation)
        self.text_column = text_column
        self.label_column = label_column
        self.label_all_tokens = label_all_tokens
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess token classification examples"""
        tokenized_inputs = self.tokenizer(
            examples[self.text_column],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors=None
        )
        
        if self.label_column in examples:
            labels = []
            for i, label in enumerate(examples[self.label_column]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(label[word_idx] if self.label_all_tokens else -100)
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            tokenized_inputs["labels"] = labels
        
        return tokenized_inputs


class QuestionAnsweringPreprocessor(DataPreprocessor):
    """Preprocessor for question answering tasks"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        question_column: str = "question",
        context_column: str = "context",
        answer_column: str = "answers",
        max_length: int = 384,
        doc_stride: int = 128,
        padding: Union[bool, str] = True,
        truncation: str = "only_second"
    ):
        super().__init__(tokenizer, max_length, padding, truncation)
        self.question_column = question_column
        self.context_column = context_column
        self.answer_column = answer_column
        self.doc_stride = doc_stride
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess question answering examples"""
        questions = [q.strip() for q in examples[self.question_column]]
        contexts = examples[self.context_column]
        
        # Tokenize questions and contexts
        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            truncation=self.truncation,
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.padding,
        )
        
        # Handle answers if present (training mode)
        if self.answer_column in examples:
            answers = examples[self.answer_column]
            start_positions = []
            end_positions = []
            
            for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)
                
                # Get the sequence ids to know which part is question vs context
                sequence_ids = tokenized_examples.sequence_ids(i)
                
                # Find the start and end of the context
                context_start = sequence_ids.index(1) if 1 in sequence_ids else None
                context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1) if 1 in sequence_ids else None
                
                # If no answers, set to CLS position
                answer = answers[i]
                if len(answer["answer_start"]) == 0:
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    # Start/end character index of the answer in the text
                    start_char = answer["answer_start"][0]
                    end_char = start_char + len(answer["text"][0])
                    
                    # Find token start and end positions
                    token_start_index = 0
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)
                    
                    token_end_index = len(offsets) - 1
                    while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)
            
            tokenized_examples["start_positions"] = start_positions
            tokenized_examples["end_positions"] = end_positions
        
        return tokenized_examples


class MultiTaskPreprocessor:
    """Preprocessor for multi-task learning"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        task_preprocessors: Dict[str, DataPreprocessor]
    ):
        self.tokenizer = tokenizer
        self.task_preprocessors = task_preprocessors
    
    def preprocess_datasets(
        self,
        datasets: Dict[str, DatasetDict]
    ) -> Dict[str, DatasetDict]:
        """Preprocess multiple datasets for different tasks"""
        processed_datasets = {}
        
        for task_name, dataset in datasets.items():
            if task_name in self.task_preprocessors:
                preprocessor = self.task_preprocessors[task_name]
                processed_datasets[task_name] = preprocessor.preprocess_dataset(dataset)
                logger.info(f"Preprocessed dataset for task: {task_name}")
            else:
                logger.warning(f"No preprocessor found for task: {task_name}")
                processed_datasets[task_name] = dataset
        
        return processed_datasets
    
    def create_mixed_dataset(
        self,
        datasets: Dict[str, Dataset],
        task_weights: Optional[Dict[str, float]] = None
    ) -> Dataset:
        """Create a mixed dataset from multiple task datasets"""
        if task_weights is None:
            task_weights = {task: 1.0 for task in datasets.keys()}
        
        # Calculate sampling probabilities
        total_weight = sum(task_weights.values())
        sampling_probs = {task: weight / total_weight for task, weight in task_weights.items()}
        
        # Sample examples from each task
        mixed_examples = []
        
        for task_name, dataset in datasets.items():
            prob = sampling_probs[task_name]
            num_samples = int(len(dataset) * prob)
            
            # Add task identifier to each example
            for i, example in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                example["task_name"] = task_name
                example["task_id"] = list(datasets.keys()).index(task_name)
                mixed_examples.append(example)
        
        # Shuffle the mixed examples
        import random
        random.shuffle(mixed_examples)
        
        return Dataset.from_list(mixed_examples)


def get_preprocessor(
    task_type: str,
    tokenizer: PreTrainedTokenizer,
    **kwargs
) -> DataPreprocessor:
    """Factory function to get appropriate preprocessor for task type"""
    
    preprocessor_map = {
        "text_classification": TextClassificationPreprocessor,
        "classification": TextClassificationPreprocessor,
        "token_classification": TokenClassificationPreprocessor,
        "ner": TokenClassificationPreprocessor,
        "question_answering": QuestionAnsweringPreprocessor,
        "qa": QuestionAnsweringPreprocessor,
    }
    
    if task_type not in preprocessor_map:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return preprocessor_map[task_type](tokenizer, **kwargs)


def create_multi_task_preprocessor(
    tokenizer: PreTrainedTokenizer,
    task_configs: Dict[str, Dict[str, Any]]
) -> MultiTaskPreprocessor:
    """Create a multi-task preprocessor from task configurations"""
    task_preprocessors = {}
    
    for task_name, config in task_configs.items():
        task_type = config.get("task_type", "text_classification")
        preprocessor_kwargs = {k: v for k, v in config.items() if k != "task_type"}
        
        task_preprocessors[task_name] = get_preprocessor(
            task_type, tokenizer, **preprocessor_kwargs
        )
    
    return MultiTaskPreprocessor(tokenizer, task_preprocessors)
