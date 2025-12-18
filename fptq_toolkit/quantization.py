import logging
from typing import Callable, Optional

import torch

from .config import QuantFormat


logger = logging.getLogger(__name__)


class QuantizationEngine:
    """Engine for applying Post-Training Quantization to models.
    
    This class wraps NVIDIA's ModelOpt quantization API and provides
    a clean interface for applying different quantization formats.
    
    Supported formats:
        - FP8: 8-bit floating point (best speed/accuracy balance)
        - INT8: 8-bit integer quantization
    
    Attributes:
        quant_format: Target quantization format
    """
    
    # Mapping from QuantFormat to ModelOpt config names
    QUANT_CONFIG_MAP = {
        QuantFormat.FP8: "FP8_DEFAULT_CFG",
        QuantFormat.INT8: "INT8_DEFAULT_CFG",
    }
    
    def __init__(self, quant_format: QuantFormat = QuantFormat.FP8):
        """Initialize quantization engine.
        
        Args:
            quant_format: Target quantization format
            
        Raises:
            ValueError: If quant_format is not supported
        """
        if quant_format not in self.QUANT_CONFIG_MAP:
            raise ValueError(
                f"Unsupported quantization format: {quant_format}. "
                f"Supported formats: {list(self.QUANT_CONFIG_MAP.keys())}"
            )
        
        self.quant_format = quant_format
        self._mtq = None  # Lazy load
    
    @property
    def mtq(self):
        """Lazy load modelopt.torch.quantization module.
        
        Returns:
            modelopt.torch.quantization module
        """
        if self._mtq is None:
            try:
                import modelopt.torch.quantization as mtq
                self._mtq = mtq
            except ImportError as e:
                raise ImportError(
                    "nvidia-modelopt is required for quantization. "
                    "Install with: pip install nvidia-modelopt[hf]"
                ) from e
        return self._mtq
    
    def get_quant_config(self):
        """Get ModelOpt quantization configuration.
        
        Returns:
            ModelOpt quantization configuration object
        """
        config_name = self.QUANT_CONFIG_MAP[self.quant_format]
        config = getattr(self.mtq, config_name)
        logger.info(f"Using quantization config: {config_name}")
        return config
    
    def quantize(
        self,
        model: torch.nn.Module,
        forward_loop: Callable,
        custom_config: Optional[dict] = None,
    ) -> torch.nn.Module:
        """Apply quantization to model with calibration.
        
        This method:
        1. Inserts quantization hooks into the model
        2. Runs the forward_loop for calibration
        3. Computes scaling factors from collected statistics
        4. Returns the quantized model
        
        Args:
            model: Model to quantize (will be modified)
            forward_loop: Callable that runs calibration passes
            custom_config: Optional custom quantization config to merge
            
        Returns:
            Quantized model with FP8/FP4 operations
        """
        quant_cfg = self.get_quant_config()
        
        # Merge custom config if provided
        if custom_config is not None:
            quant_cfg = {**quant_cfg, **custom_config}
            logger.info("Merged custom config with default")
        
        logger.info(
            f"Applying {self.quant_format.value.upper()} quantization..."
        )
        
        # Apply quantization with calibration
        quantized_model = self.mtq.quantize(
            model,
            quant_cfg,
            forward_loop=forward_loop,
        )
        
        logger.info("Quantization complete!")
        return quantized_model
    
    def get_num_quantizers(self, model: torch.nn.Module) -> int:
        """Count number of quantizer modules in model.
        
        Useful for verifying quantization was applied correctly.
        
        Args:
            model: Model to inspect
            
        Returns:
            Number of quantizer modules found
        """
        count = 0
        for name, module in model.named_modules():
            if "quantizer" in name.lower() or "quant" in type(module).__name__.lower():
                count += 1
        return count


class QuantizationEngineWithValidation(QuantizationEngine):
    """Quantization engine with additional validation capabilities.
    
    Extends the base engine with:
    - Pre-quantization validation
    - Post-quantization quality checks
    - Numerical comparison between original and quantized outputs
    """
    
    def validate_model(self, model: torch.nn.Module) -> bool:
        """Validate model is compatible with quantization.
        
        Checks:
        - Model has Linear layers (primary quantization targets)
        - Model is on CUDA (required for FP8)
        - Model is in eval mode
        
        Args:
            model: Model to validate
            
        Returns:
            True if model is compatible
            
        Raises:
            ValueError: If model is not compatible
        """
        # Check for Linear layers
        has_linear = any(
            isinstance(m, torch.nn.Linear)
            for m in model.modules()
        )
        if not has_linear:
            raise ValueError("Model has no Linear layers to quantize")
        
        # Check device
        try:
            param = next(model.parameters())
            if not param.is_cuda:
                raise ValueError(
                    "Model must be on CUDA for FP8 quantization. "
                    "Call model.cuda() first."
                )
        except StopIteration:
            raise ValueError("Model has no parameters")
        
        # Check training mode
        if model.training:
            logger.warning(
                "Model is in training mode. "
                "Switching to eval mode for quantization."
            )
            model.eval()
        
        logger.info("Model validation passed")
        return True
    
    def compare_outputs(
        self,
        original_model: torch.nn.Module,
        quantized_model: torch.nn.Module,
        sample_input: dict,
        rtol: float = 0.1,
        atol: float = 0.01,
    ) -> dict:
        """Compare outputs between original and quantized models.
        
        Args:
            original_model: Original FP16/FP32 model
            quantized_model: Quantized model
            sample_input: Sample input dict with 'input_ids'
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            
        Returns:
            Dictionary with comparison metrics
        """
        original_model.eval()
        quantized_model.eval()
        
        with torch.no_grad():
            orig_output = original_model(**sample_input)
            quant_output = quantized_model(**sample_input)
        
        # Get logits
        orig_logits = orig_output.logits if hasattr(orig_output, 'logits') else orig_output
        quant_logits = quant_output.logits if hasattr(quant_output, 'logits') else quant_output
        
        # Compute metrics
        abs_diff = torch.abs(orig_logits - quant_logits)
        rel_diff = abs_diff / (torch.abs(orig_logits) + 1e-8)
        
        metrics = {
            "max_abs_diff": abs_diff.max().item(),
            "mean_abs_diff": abs_diff.mean().item(),
            "max_rel_diff": rel_diff.max().item(),
            "mean_rel_diff": rel_diff.mean().item(),
            "within_tolerance": torch.allclose(
                orig_logits, quant_logits, rtol=rtol, atol=atol
            ),
        }
        
        logger.info(f"Output comparison: {metrics}")
        return metrics
