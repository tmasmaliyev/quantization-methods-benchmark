from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QuantFormat(str, Enum):
    """Supported quantization formats.
    
    Attributes:
        FP8: 8-bit floating point quantization
        INT8: 8-bit integer quantization
    """
    FP8 = "fp8"
    INT8 = "int8"


@dataclass
class PTQConfig:
    """Configuration for Post-Training Quantization pipeline.
    
    This dataclass holds all configuration parameters needed to run
    the PTQ pipeline, from model loading through export.
    
    Attributes:
        model_name: HuggingFace model identifier or local path
        quant_format: Quantization format (fp8, fp4, nvfp4, int8)
        dataset_name: Calibration dataset name
        num_calib_samples: Number of samples for calibration
        batch_size: Batch size for calibration
        seq_length: Sequence length for tokenization
        export_path: Local path to save quantized model
        hf_repo_id: HuggingFace repo ID for upload (optional)
        device: Target device for computation
        hf_token: HuggingFace token for gated models
    """
    model_name: str = "meta-llama/Llama-2-7b-hf"
    quant_format: QuantFormat = QuantFormat.FP8
    dataset_name: str = "wikitext"
    num_calib_samples: int = 512
    batch_size: int = 4
    seq_length: int = 1024
    export_path: str = "./quantized_model"
    hf_repo_id: Optional[str] = None
    device: str = "cuda"
    hf_token: Optional[str] = None
    
    def __post_init__(self):
        """Convert string quant_format to QuantFormat enum if needed."""
        if isinstance(self.quant_format, str):
            self.quant_format = QuantFormat(self.quant_format.lower())
    
    def to_dict(self) -> dict:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            "model_name": self.model_name,
            "quant_format": self.quant_format.value,
            "dataset_name": self.dataset_name,
            "num_calib_samples": self.num_calib_samples,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "export_path": self.export_path,
            "hf_repo_id": self.hf_repo_id,
            "device": self.device,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "PTQConfig":
        """Create config from dictionary.
        
        Args:
            config_dict: Dictionary with config parameters
            
        Returns:
            PTQConfig instance
        """
        return cls(**config_dict)
