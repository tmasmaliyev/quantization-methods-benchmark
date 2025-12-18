import torch
from torch.utils.data import Dataset


class CalibrationDataset(Dataset):
    """PyTorch Dataset for PTQ calibration.
    
    Handles tokenized sequences for calibration, ensuring proper
    tensor formatting and memory layout for efficient GPU transfer.
    
    The dataset splits a long sequence of tokens into fixed-length
    chunks that can be used for calibration forward passes.
    
    Attributes:
        tokens: Contiguous tensor of all token IDs
        seq_length: Length of each sample sequence
        num_samples: Total number of samples available
    """
    
    def __init__(self, tokens: torch.Tensor, seq_length: int):
        """Initialize calibration dataset.
        
        Args:
            tokens: Flattened tensor of token IDs from tokenizer
            seq_length: Length of each sample sequence
            
        Raises:
            ValueError: If tokens is empty or seq_length is invalid
        """
        if len(tokens) == 0:
            raise ValueError("Tokens tensor cannot be empty")
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if seq_length > len(tokens):
            raise ValueError(
                f"seq_length ({seq_length}) cannot exceed "
                f"total tokens ({len(tokens)})"
            )
        
        # Store as contiguous tensor for efficient slicing
        self.tokens = tokens.contiguous()
        self.seq_length = seq_length
        self.num_samples = (len(tokens) - seq_length) // seq_length
        
        if self.num_samples == 0:
            raise ValueError(
                f"Not enough tokens ({len(tokens)}) for even one sample "
                f"of length {seq_length}"
            )
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tensor of token IDs with shape (seq_length,)
            
        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset with "
                f"{self.num_samples} samples"
            )
        
        start = idx * self.seq_length
        # Clone to ensure independent memory
        return self.tokens[start:start + self.seq_length].clone()
    
    def get_total_tokens(self) -> int:
        """Get total number of tokens used in calibration.
        
        Returns:
            Total tokens across all samples
        """
        return self.num_samples * self.seq_length


def collate_fn(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Custom collate function for calibration DataLoader.
    
    Stacks individual token tensors into a batch and ensures
    the result is contiguous for efficient GPU operations.
    
    Args:
        batch: List of token tensors from CalibrationDataset
        
    Returns:
        Dictionary with 'input_ids' key containing stacked,
        contiguous tensor of shape (batch_size, seq_length)
    """
    input_ids = torch.stack(batch, dim=0).contiguous()
    return {"input_ids": input_ids}
