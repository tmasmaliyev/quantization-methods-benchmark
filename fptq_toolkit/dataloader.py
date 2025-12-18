import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset import CalibrationDataset, collate_fn


logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory for creating calibration DataLoaders.
    
    This factory handles loading datasets from HuggingFace,
    tokenizing text, and creating properly configured DataLoaders
    for PTQ calibration.
    
    Supported datasets:
        - WikiText-2 (aliases: wikitext, wikitext-2, wikitext2)
        - Any HuggingFace dataset with a text field
    """
    
    # Aliases for WikiText-2 dataset
    WIKITEXT_ALIASES = {"wikitext-2", "wikitext2", "wikitext"}
    
    # Common text field names to try
    TEXT_FIELD_NAMES = ["text", "content", "article", "document", "sentence", "body"]
    
    @classmethod
    def create(
        cls,
        dataset_name: str,
        tokenizer,
        batch_size: int = 8,
        num_samples: Optional[int] = None,
        seq_length: int = 2048,
        device: str = "cuda",
    ) -> DataLoader:
        """Create a DataLoader for calibration.
        
        Args:
            dataset_name: Name of HuggingFace dataset to load
            tokenizer: HuggingFace tokenizer instance
            batch_size: Batch size for DataLoader
            num_samples: Maximum number of samples (None for all available)
            seq_length: Sequence length per sample
            device: Target device (used for pin_memory optimization)
            
        Returns:
            Configured DataLoader ready for calibration
            
        Raises:
            ValueError: If dataset cannot be loaded or has no text field
        """
        from datasets import load_dataset
        
        logger.info(f"Loading {dataset_name} dataset...")
        
        # Load dataset
        if dataset_name.lower() in cls.WIKITEXT_ALIASES:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text_field = "text"
        else:
            dataset = load_dataset(dataset_name, split="train")
            text_field = cls._detect_text_field(dataset)
        
        # Concatenate all text with proper spacing
        texts = [item[text_field] for item in dataset if item[text_field].strip()]
        text = "\n\n".join(texts)
        
        logger.info(f"Loaded {len(texts)} text samples")
        
        # Tokenize entire text
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,  # Avoid adding extra tokens between chunks
        )["input_ids"][0]
        
        logger.info(f"Total tokens: {len(tokens):,}")
        
        # Create calibration dataset
        calib_dataset = CalibrationDataset(tokens, seq_length)
        
        # Limit samples if specified
        if num_samples is not None and num_samples < len(calib_dataset):
            calib_dataset = torch.utils.data.Subset(
                calib_dataset, range(num_samples)
            )
            effective_samples = num_samples
        else:
            effective_samples = len(calib_dataset)
        
        # Create DataLoader with optimizations
        dataloader = DataLoader(
            calib_dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep order for reproducibility
            num_workers=0,  # Avoid multiprocessing issues with CUDA
            pin_memory=(device == "cuda"),  # Faster GPU transfer
            collate_fn=collate_fn,
            drop_last=False,  # Use all available data
        )
        
        logger.info(
            f"Created DataLoader: {len(dataloader)} batches, "
            f"batch_size={batch_size}, total_samples={effective_samples}"
        )
        
        return dataloader
    
    @classmethod
    def create_from_texts(
        cls,
        texts: list[str],
        tokenizer,
        batch_size: int = 8,
        num_samples: Optional[int] = None,
        seq_length: int = 2048,
        device: str = "cuda",
    ) -> DataLoader:
        """Create DataLoader from custom text list.
        
        Use this method when you have your own domain-specific
        calibration data.
        
        Args:
            texts: List of text strings for calibration
            tokenizer: HuggingFace tokenizer instance
            batch_size: Batch size for DataLoader
            num_samples: Maximum number of samples
            seq_length: Sequence length per sample
            device: Target device
            
        Returns:
            Configured DataLoader
        """
        logger.info(f"Creating DataLoader from {len(texts)} custom texts...")
        
        # Concatenate texts
        full_text = "\n\n".join(texts)
        
        # Tokenize
        tokens = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )["input_ids"][0]
        
        logger.info(f"Total tokens: {len(tokens):,}")
        
        # Create dataset
        calib_dataset = CalibrationDataset(tokens, seq_length)
        
        # Limit samples
        if num_samples is not None and num_samples < len(calib_dataset):
            calib_dataset = torch.utils.data.Subset(
                calib_dataset, range(num_samples)
            )
        
        # Create DataLoader
        dataloader = DataLoader(
            calib_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
            collate_fn=collate_fn,
        )
        
        return dataloader
    
    @classmethod
    def _detect_text_field(cls, dataset) -> str:
        """Detect the text field name in a HuggingFace dataset.
        
        Args:
            dataset: HuggingFace dataset object
            
        Returns:
            Name of the text field
            
        Raises:
            ValueError: If no text field can be detected
        """
        available_fields = list(dataset.features.keys())
        
        for field in cls.TEXT_FIELD_NAMES:
            if field in available_fields:
                logger.info(f"Detected text field: '{field}'")
                return field
        
        # If no common field found, raise helpful error
        raise ValueError(
            f"Could not detect text field in dataset. "
            f"Available fields: {available_fields}. "
            f"Expected one of: {cls.TEXT_FIELD_NAMES}"
        )
