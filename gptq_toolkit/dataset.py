import logging
from typing import List, Dict

import torch
from datasets import load_dataset


logger = logging.getLogger(__name__)


class GPTQDatasetFactory:
    """Factory for creating GPTQ calibration datasets.
    
    This factory handles loading datasets from HuggingFace,
    filtering for sufficient length, and tokenizing for GPTQ.
    
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
        num_samples: int = 256,
        seq_length: int = 512,
    ) -> List[Dict[str, torch.Tensor]]:
        """Create calibration dataset for GPTQ.
        
        Args:
            dataset_name: Name of HuggingFace dataset
            tokenizer: HuggingFace tokenizer instance
            num_samples: Number of calibration samples
            seq_length: Sequence length per sample
            
        Returns:
            List of dicts with 'input_ids' and 'attention_mask' tensors
            
        Raises:
            ValueError: If dataset cannot be loaded or has no text field
        """
        logger.info(f"Loading {dataset_name} dataset...")
        
        # Load dataset
        if dataset_name.lower() in cls.WIKITEXT_ALIASES:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text_field = "text"
        else:
            dataset = load_dataset(dataset_name, split="train")
            text_field = cls._detect_text_field(dataset)
        
        # Filter samples with sufficient length
        logger.info(f"Filtering samples with length >= {seq_length}...")
        dataset = dataset.filter(lambda x: len(x[text_field]) >= seq_length)
        
        logger.info(f"Preparing {num_samples} calibration samples...")
        
        # Tokenize samples
        examples = []
        actual_samples = min(num_samples, len(dataset))
        
        for i, example in enumerate(dataset.select(range(actual_samples))):
            tokenized = tokenizer(
                example[text_field],
                truncation=True,
                max_length=seq_length,
                return_tensors="pt"
            )
            
            examples.append({
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            })
            
            if (i + 1) % 50 == 0:
                logger.debug(f"Processed {i + 1}/{actual_samples} samples")
        
        logger.info(f"✓ Created {len(examples)} calibration samples")
        return examples
    
    @classmethod
    def create_from_texts(
        cls,
        texts: List[str],
        tokenizer,
        seq_length: int = 512,
    ) -> List[Dict[str, torch.Tensor]]:
        """Create calibration dataset from custom text list.
        
        Use this method when you have your own domain-specific
        calibration data.
        
        Args:
            texts: List of text strings for calibration
            tokenizer: HuggingFace tokenizer instance
            seq_length: Sequence length per sample
            
        Returns:
            List of tokenized calibration samples
        """
        logger.info(f"Creating dataset from {len(texts)} custom texts...")
        
        examples = []
        for text in texts:
            if len(text) < seq_length:
                logger.warning(f"Skipping short text (len={len(text)})")
                continue
            
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=seq_length,
                return_tensors="pt"
            )
            
            examples.append({
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            })
        
        logger.info(f"✓ Created {len(examples)} calibration samples")
        return examples
    
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
        
        raise ValueError(
            f"Could not detect text field in dataset. "
            f"Available fields: {available_fields}. "
            f"Expected one of: {cls.TEXT_FIELD_NAMES}"
        )

