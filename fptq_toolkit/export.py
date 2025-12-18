import logging
from pathlib import Path
from typing import Optional

import torch


logger = logging.getLogger(__name__)


class ModelExporter:
    """Handles quantized model export and HuggingFace Hub upload.
    
    This class provides utilities for:
    - Exporting quantized models in HuggingFace-compatible format
    - Saving tokenizers alongside models
    - Uploading models to HuggingFace Hub
    
    The exported models can be loaded with:
    - HuggingFace transformers (from_pretrained)
    - vLLM inference engine
    - TensorRT-LLM
    - SGLang
    """
    
    @staticmethod
    def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        """Unwrap model from torch.compile or other wrappers.
        
        Args:
            model: Potentially wrapped model
            
        Returns:
            Unwrapped base model
        """
        # Unwrap torch.compile
        if hasattr(model, '_orig_mod'):
            logger.info("Unwrapping torch.compile model")
            model = model._orig_mod
        
        # Unwrap DataParallel/DistributedDataParallel
        if hasattr(model, 'module'):
            logger.info("Unwrapping DataParallel model")
            model = model.module
        
        return model
    
    @staticmethod
    def export_checkpoint(
        model: torch.nn.Module,
        tokenizer,
        export_path: str,
        save_config: bool = True,
    ) -> Path:
        """Export quantized model to HuggingFace checkpoint format.
        
        Creates a directory with:
        - model weights (safetensors format)
        - config.json
        - tokenizer files
        - quantization config
        
        Args:
            model: Quantized model to export
            tokenizer: Associated tokenizer
            export_path: Output directory path
            save_config: Whether to save model config
            
        Returns:
            Path to exported checkpoint directory
        """
        try:
            from modelopt.torch.export import export_hf_checkpoint
        except ImportError as e:
            raise ImportError(
                "nvidia-modelopt is required for export. "
                "Install with: pip install nvidia-modelopt[hf]"
            ) from e
        
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Unwrap model if needed
        model = ModelExporter.unwrap_model(model)
        
        logger.info(f"Exporting quantized checkpoint to {export_path}...")
        
        # Export model with ModelOpt
        export_hf_checkpoint(
            model,
            export_dir=str(export_path),
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(str(export_path))
        
        logger.info(f"Export complete! Files saved to: {export_path}")
        
        # List exported files
        files = list(export_path.iterdir())
        logger.info(f"Exported files: {[f.name for f in files]}")
        
        return export_path
    
    @staticmethod
    def upload_to_hub(
        export_path: str,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: str = "Upload quantized model",
    ) -> str:
        """Upload exported model to HuggingFace Hub.
        
        Args:
            export_path: Path to exported checkpoint
            repo_id: HuggingFace repo ID (format: username/repo_name)
            token: HuggingFace authentication token
            private: Whether to create private repository
            commit_message: Git commit message for upload
            
        Returns:
            URL of uploaded model on HuggingFace Hub
        """
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for upload. "
                "Install with: pip install huggingface_hub"
            ) from e
        
        export_path = Path(export_path)
        if not export_path.exists():
            raise ValueError(f"Export path does not exist: {export_path}")
        
        logger.info(f"Uploading to HuggingFace Hub: {repo_id}")
        
        # Create repository (won't fail if exists)
        create_repo(
            repo_id,
            exist_ok=True,
            private=private,
            token=token,
        )
        
        # Upload all files
        api = HfApi()
        api.upload_folder(
            folder_path=str(export_path),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message,
        )
        
        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Upload complete! Model available at: {url}")
        
        return url
    
    @staticmethod
    def create_model_card(
        export_path: str,
        model_name: str,
        quant_format: str,
        base_model: str,
        calibration_dataset: str = "wikitext",
        num_calib_samples: int = 512,
        additional_info: Optional[dict] = None,
    ) -> Path:
        """Create a model card README for the exported model.
        
        Args:
            export_path: Path to exported checkpoint
            model_name: Name for the quantized model
            quant_format: Quantization format used
            base_model: Original base model name
            calibration_dataset: Dataset used for calibration
            num_calib_samples: Number of calibration samples
            additional_info: Additional info to include
            
        Returns:
            Path to created README.md
        """
        export_path = Path(export_path)
        readme_path = export_path / "README.md"
        
        content = f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - quantized
  - {quant_format}
  - nvidia-modelopt
  - ptq
---

# {model_name}

This model is a {quant_format.upper()}-quantized version of [{base_model}](https://huggingface.co/{base_model}).

## Quantization Details

- **Format**: {quant_format.upper()}
- **Method**: Post-Training Quantization (PTQ) with Min-Max Calibration
- **Toolkit**: NVIDIA ModelOpt
- **Calibration Dataset**: {calibration_dataset}
- **Calibration Samples**: {num_calib_samples}

## Usage

### With vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{model_name}")
output = llm.generate("Hello, world!")
```

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```

## Performance

[Add benchmark results here]

## Citation

If you use this model, please cite:

```bibtex
@misc{{{model_name.replace('/', '_').replace('-', '_')}}},
  title={{{model_name}}},
  year={{2024}},
  note={{Quantized with NVIDIA ModelOpt}}
}}
```
"""
        
        if additional_info:
            content += "\n## Additional Information\n\n"
            for key, value in additional_info.items():
                content += f"- **{key}**: {value}\n"
        
        readme_path.write_text(content)
        logger.info(f"Created model card: {readme_path}")
        
        return readme_path
