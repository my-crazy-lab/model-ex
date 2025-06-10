"""
Data loading utilities for Adapter Tuning implementation
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datasets import Dataset, DatasetDict, load_dataset
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Utility class for loading and managing datasets"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
    
    def load_from_hub(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs
    ) -> Union[Dataset, DatasetDict]:
        """Load dataset from Hugging Face Hub"""
        try:
            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                cache_dir=self.cache_dir,
                **kwargs
            )
            logger.info(f"Successfully loaded dataset: {dataset_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
            raise
    
    def load_from_files(
        self,
        train_file: str,
        validation_file: Optional[str] = None,
        test_file: Optional[str] = None,
        file_format: str = "auto"
    ) -> DatasetDict:
        """Load dataset from local files"""
        datasets = {}
        
        # Load training data
        datasets["train"] = self._load_single_file(train_file, file_format)
        
        # Load validation data if provided
        if validation_file:
            datasets["validation"] = self._load_single_file(validation_file, file_format)
        
        # Load test data if provided
        if test_file:
            datasets["test"] = self._load_single_file(test_file, file_format)
        
        return DatasetDict(datasets)
    
    def _load_single_file(self, file_path: str, file_format: str = "auto") -> Dataset:
        """Load a single file into a Dataset"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect format if not specified
        if file_format == "auto":
            file_format = self._detect_file_format(file_path)
        
        if file_format == "json":
            return self._load_json(file_path)
        elif file_format == "jsonl":
            return self._load_jsonl(file_path)
        elif file_format == "csv":
            return self._load_csv(file_path)
        elif file_format == "tsv":
            return self._load_tsv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def _detect_file_format(self, file_path: str) -> str:
        """Auto-detect file format based on extension"""
        extension = os.path.splitext(file_path)[1].lower()
        
        format_map = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".csv": "csv",
            ".tsv": "tsv",
            ".txt": "text"
        }
        
        return format_map.get(extension, "text")
    
    def _load_json(self, file_path: str) -> Dataset:
        """Load JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return Dataset.from_list(data)
        elif isinstance(data, dict):
            return Dataset.from_dict(data)
        else:
            raise ValueError("JSON file must contain a list or dictionary")
    
    def _load_jsonl(self, file_path: str) -> Dataset:
        """Load JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        return Dataset.from_list(data)
    
    def _load_csv(self, file_path: str) -> Dataset:
        """Load CSV file"""
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)
    
    def _load_tsv(self, file_path: str) -> Dataset:
        """Load TSV file"""
        df = pd.read_csv(file_path, sep='\t')
        return Dataset.from_pandas(df)
    
    def create_train_val_split(
        self,
        dataset: Dataset,
        val_size: float = 0.2,
        seed: int = 42
    ) -> DatasetDict:
        """Split dataset into train and validation sets"""
        split_dataset = dataset.train_test_split(
            test_size=val_size,
            seed=seed
        )
        
        return DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
    
    def load_multi_task_datasets(
        self,
        task_configs: Dict[str, Dict[str, str]]
    ) -> Dict[str, DatasetDict]:
        """
        Load multiple datasets for multi-task learning
        
        Args:
            task_configs: Dict mapping task names to dataset configs
                         e.g., {"sentiment": {"dataset": "imdb"}, 
                               "classification": {"dataset": "ag_news"}}
        
        Returns:
            Dict mapping task names to DatasetDict objects
        """
        multi_task_datasets = {}
        
        for task_name, config in task_configs.items():
            try:
                dataset = self.load_from_hub(
                    config["dataset"],
                    subset=config.get("subset"),
                    **config.get("kwargs", {})
                )
                multi_task_datasets[task_name] = dataset
                logger.info(f"Loaded dataset for task '{task_name}': {config['dataset']}")
            except Exception as e:
                logger.error(f"Failed to load dataset for task '{task_name}': {e}")
                raise
        
        return multi_task_datasets


def load_dataset_from_hub(
    dataset_name: str,
    subset: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """Convenience function to load dataset from Hugging Face Hub"""
    loader = DataLoader(cache_dir=cache_dir)
    return loader.load_from_hub(dataset_name, subset)


def load_custom_dataset(
    train_file: str,
    validation_file: Optional[str] = None,
    test_file: Optional[str] = None,
    file_format: str = "auto"
) -> DatasetDict:
    """Convenience function to load custom dataset from files"""
    loader = DataLoader()
    return loader.load_from_files(train_file, validation_file, test_file, file_format)


# Common dataset configurations for adapter tuning
ADAPTER_DATASETS = {
    # Text Classification
    "imdb": {
        "name": "imdb",
        "task": "text_classification",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 2
    },
    "ag_news": {
        "name": "ag_news",
        "task": "text_classification",
        "text_column": "text",
        "label_column": "label",
        "num_labels": 4
    },
    "sst2": {
        "name": "glue",
        "subset": "sst2",
        "task": "text_classification",
        "text_column": "sentence",
        "label_column": "label",
        "num_labels": 2
    },
    
    # Named Entity Recognition
    "conll2003": {
        "name": "conll2003",
        "task": "token_classification",
        "text_column": "tokens",
        "label_column": "ner_tags",
        "num_labels": 9
    },
    
    # Question Answering
    "squad": {
        "name": "squad",
        "task": "question_answering",
        "context_column": "context",
        "question_column": "question",
        "answer_column": "answers"
    },
    
    # Multi-task datasets
    "glue_multi": {
        "tasks": {
            "sst2": {"name": "glue", "subset": "sst2"},
            "cola": {"name": "glue", "subset": "cola"},
            "mrpc": {"name": "glue", "subset": "mrpc"},
            "qqp": {"name": "glue", "subset": "qqp"}
        }
    }
}
