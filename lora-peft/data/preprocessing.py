"""
Data preprocessing utilities for LoRA/PEFT implementation
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


class TextGenerationPreprocessor(DataPreprocessor):
    """Preprocessor for text generation tasks"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "text",
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        add_special_tokens: bool = True
    ):
        super().__init__(tokenizer, max_length, padding, truncation)
        self.text_column = text_column
        self.add_special_tokens = add_special_tokens
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess text generation examples"""
        # Tokenize the texts
        result = self.tokenizer(
            examples[self.text_column],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens,
            return_tensors=None
        )
        
        # For causal language modeling, labels are the same as input_ids
        result["labels"] = result["input_ids"].copy()
        
        return result


class QuestionAnsweringPreprocessor(DataPreprocessor):
    """Preprocessor for question answering tasks"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        question_column: str = "question",
        context_column: str = "context",
        answer_column: str = "answers",
        max_length: int = 512,
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


class SequenceToSequencePreprocessor(DataPreprocessor):
    """Preprocessor for sequence-to-sequence tasks"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        source_column: str = "source",
        target_column: str = "target",
        max_source_length: int = 512,
        max_target_length: int = 128,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        prefix: str = ""
    ):
        super().__init__(tokenizer, max_source_length, padding, truncation)
        self.source_column = source_column
        self.target_column = target_column
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prefix = prefix
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess sequence-to-sequence examples"""
        # Add prefix to source if specified
        sources = examples[self.source_column]
        if self.prefix:
            sources = [self.prefix + source for source in sources]
        
        # Tokenize sources
        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            padding=self.padding,
            truncation=self.truncation,
        )
        
        # Tokenize targets if present
        if self.target_column in examples:
            targets = examples[self.target_column]
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    truncation=self.truncation,
                )
            
            # Replace padding token id's of the labels by -100
            if self.padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                    for label in labels["input_ids"]
                ]
            
            model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs


def get_preprocessor(
    task_type: str,
    tokenizer: PreTrainedTokenizer,
    **kwargs
) -> DataPreprocessor:
    """Factory function to get appropriate preprocessor for task type"""
    
    preprocessor_map = {
        "text_classification": TextClassificationPreprocessor,
        "text_generation": TextGenerationPreprocessor,
        "question_answering": QuestionAnsweringPreprocessor,
        "sequence_to_sequence": SequenceToSequencePreprocessor,
    }
    
    if task_type not in preprocessor_map:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return preprocessor_map[task_type](tokenizer, **kwargs)
