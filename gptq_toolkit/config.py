from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTQConfig:
    """Configuration for GPTQ quantization pipeline.
    
    Attributes:
        model_name: HuggingFace model identifier or local path
        bits: Number of bits for quantization (4 or 8)
        group_size: Group size for quantization (128 recommended)
        dataset_name: Calibration dataset name
        num_calib_samples: Number of samples for calibration
        seq_length: Sequence length for tokenization
        export_path: Local path to save quantized model
        hf_repo_id: HuggingFace repo ID for upload (optional)
        hf_token: HuggingFace token for gated models
        desc_act: Whether to use desc_act (default: False)
        sym: Whether to use symmetric quantization (default: True)
        damp_percent: Damping percentage for GPTQ (default: 0.01)
    """
    model_name: str = "meta-llama/Llama-2-7b-hf"
    bits: int = 8
    group_size: int = 128
    dataset_name: str = "wikitext"
    num_calib_samples: int = 256
    seq_length: int = 512
    export_path: str = "./quantized_model_gptq"
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    desc_act: bool = False
    sym: bool = True
    damp_percent: float = 0.01
    
    def to_dict(self) -> dict:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            "model_name": self.model_name,
            "bits": self.bits,
            "group_size": self.group_size,
            "dataset_name": self.dataset_name,
            "num_calib_samples": self.num_calib_samples,
            "seq_length": self.seq_length,
            "export_path": self.export_path,
            "hf_repo_id": self.hf_repo_id,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "damp_percent": self.damp_percent,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "GPTQConfig":
        """Create config from dictionary.
        
        Args:
            config_dict: Dictionary with config parameters
            
        Returns:
            GPTQConfig instance
        """
        return cls(**config_dict)

