import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .config import PTQConfig
from .dataloader import DataLoaderFactory
from .calibration import CalibrationRunner
from .quantization import QuantizationEngine
from .export import ModelExporter


logger = logging.getLogger(__name__)


class PTQPipeline:
    """Complete Post-Training Quantization pipeline.
    
    Orchestrates the full PTQ workflow:
    1. Load model and tokenizer from HuggingFace
    2. Prepare calibration data
    3. Apply quantization with min-max calibration
    4. Test quantized model generation
    5. Export checkpoint in HuggingFace format
    6. (Optional) Upload to HuggingFace Hub
    
    Attributes:
        config: Pipeline configuration
        model: Loaded model (None until load_model() called)
        tokenizer: Loaded tokenizer
        quantized_model: Quantized model (None until quantize() called)
    """
    
    def __init__(self, config: PTQConfig):
        """Initialize PTQ pipeline.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.quantized_model = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    
    def load_model(self) -> None:
        """Load model and tokenizer from HuggingFace.
        
        Loads the model specified in config.model_name and moves
        it to the specified device. Also loads the tokenizer and
        sets up pad_token if needed.
        
        Raises:
            ImportError: If transformers or huggingface_hub not installed
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import login
        
        # Login if token provided (required for gated models like Llama)
        if self.config.hf_token:
            logger.info("Logging into HuggingFace Hub...")
            login(token=self.config.hf_token)
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,  # Use FP16 as base
        ).to(self.config.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name
        )
        
        # Set pad token for decoder-only models (required for batching)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Log model info
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {num_params / 1e9:.2f}B parameters")
    
    def prepare_dataloader(self) -> DataLoader:
        """Prepare calibration dataloader.
        
        Creates a DataLoader using the configured dataset,
        batch size, and sample count.
        
        Returns:
            DataLoader ready for calibration
            
        Raises:
            RuntimeError: If model not loaded yet
        """
        if self.tokenizer is None:
            raise RuntimeError("Must call load_model() first")
        
        return DataLoaderFactory.create(
            dataset_name=self.config.dataset_name,
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            num_samples=self.config.num_calib_samples,
            seq_length=self.config.seq_length,
            device=self.config.device,
        )
    
    def quantize(self, dataloader: DataLoader) -> None:
        """Apply quantization to loaded model.
        
        Runs calibration using the provided dataloader and
        applies the quantization format specified in config.
        
        Args:
            dataloader: Calibration DataLoader
            
        Raises:
            RuntimeError: If model not loaded yet
        """
        if self.model is None:
            raise RuntimeError("Must call load_model() first")
        
        # Create calibration runner
        calibrator = CalibrationRunner(
            model=self.model,
            dataloader=dataloader,
            device=self.config.device,
        )
        
        # Create quantization engine
        engine = QuantizationEngine(self.config.quant_format)
        
        # Apply quantization with calibration
        self.quantized_model = engine.quantize(
            self.model,
            forward_loop=calibrator.get_forward_loop(),
        )
    
    def test_generation(
        self,
        prompt: str = "Hello world",
        max_new_tokens: int = 20,
        compile_model: bool = True,
    ) -> str:
        """Test quantized model with simple generation.
        
        Runs a quick generation test to verify the quantized
        model produces reasonable outputs.
        
        Args:
            prompt: Test prompt to generate from
            max_new_tokens: Number of tokens to generate
            compile_model: Whether to use torch.compile
            
        Returns:
            Generated text string
            
        Raises:
            RuntimeError: If quantized model not available
        """
        if self.quantized_model is None:
            raise RuntimeError("Must call quantize() first")
        
        logger.info("Testing quantized model generation...")
        
        model = self.quantized_model
        
        # Optionally compile for faster inference
        if compile_model:
            model = torch.compile(model)
        
        # Tokenize and generate
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        logger.info(f"Generated: {generated}")
        return generated
    
    def export(self) -> Path:
        """Export quantized model checkpoint.
        
        Exports the quantized model in HuggingFace-compatible
        format, including tokenizer files.
        
        Returns:
            Path to exported checkpoint directory
            
        Raises:
            RuntimeError: If quantized model not available
        """
        if self.quantized_model is None:
            raise RuntimeError("Must call quantize() first")
        
        # Get base model (unwrap if compiled)
        model = self.quantized_model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        
        return ModelExporter.export_checkpoint(
            model=model,
            tokenizer=self.tokenizer,
            export_path=self.config.export_path,
        )
    
    def upload(self) -> Optional[str]:
        """Upload to HuggingFace Hub if configured.
        
        Uploads the exported checkpoint to HuggingFace Hub
        if hf_repo_id is set in config.
        
        Returns:
            URL of uploaded model, or None if not configured
        """
        if self.config.hf_repo_id is None:
            logger.info("No HuggingFace repo configured, skipping upload")
            return None
        
        return ModelExporter.upload_to_hub(
            export_path=self.config.export_path,
            repo_id=self.config.hf_repo_id,
            token=self.config.hf_token,
        )
    
    def run(
        self,
        test_prompt: str = "Hello world",
        skip_test: bool = False,
        skip_export: bool = False,
        skip_upload: bool = False,
    ) -> dict:
        """Execute full PTQ pipeline.
        
        Runs all steps in sequence:
        1. Load model
        2. Prepare calibration data
        3. Quantize model
        4. Test generation (optional)
        5. Export checkpoint (optional)
        6. Upload to Hub (optional)
        
        Args:
            test_prompt: Prompt for generation test
            skip_test: Skip generation test
            skip_export: Skip checkpoint export
            skip_upload: Skip Hub upload
            
        Returns:
            Dictionary with pipeline results:
            - export_path: Path to exported checkpoint
            - hub_url: HuggingFace Hub URL (if uploaded)
            - test_generation: Generated text from test
        """
        logger.info("=" * 60)
        logger.info("Starting PTQ Pipeline")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Format: {self.config.quant_format.value.upper()}")
        logger.info("=" * 60)
        
        # Step 1: Load model
        logger.info("\n[Step 1/6] Loading model...")
        self.load_model()
        
        # Step 2: Prepare data
        logger.info("\n[Step 2/6] Preparing calibration data...")
        dataloader = self.prepare_dataloader()
        
        # Step 3: Quantize
        logger.info("\n[Step 3/6] Applying quantization...")
        self.quantize(dataloader)
        
        # Step 4: Test
        generated = None
        if not skip_test:
            logger.info("\n[Step 4/6] Testing quantized model...")
            generated = self.test_generation(test_prompt)
        else:
            logger.info("\n[Step 4/6] Skipping test...")
        
        # Step 5: Export
        export_path = None
        if not skip_export:
            logger.info("\n[Step 5/6] Exporting checkpoint...")
            export_path = self.export()
        else:
            logger.info("\n[Step 5/6] Skipping export...")
        
        # Step 6: Upload
        hub_url = None
        if not skip_upload and not skip_export:
            logger.info("\n[Step 6/6] Uploading to Hub...")
            hub_url = self.upload()
        else:
            logger.info("\n[Step 6/6] Skipping upload...")
        
        logger.info("\n" + "=" * 60)
        logger.info("PTQ Pipeline Complete!")
        logger.info("=" * 60)
        
        return {
            "export_path": str(export_path) if export_path else None,
            "hub_url": hub_url,
            "test_generation": generated,
            "config": self.config.to_dict(),
        }


class PTQPipelineBuilder:
    """Builder pattern for PTQPipeline configuration.
    
    Provides a fluent interface for building pipeline configurations.
    """
    
    def __init__(self):
        self._config_dict = {}
    
    def model(self, model_name: str) -> "PTQPipelineBuilder":
        """Set model name."""
        self._config_dict["model_name"] = model_name
        return self
    
    def format(self, quant_format: str) -> "PTQPipelineBuilder":
        """Set quantization format."""
        self._config_dict["quant_format"] = quant_format
        return self
    
    def dataset(self, dataset_name: str) -> "PTQPipelineBuilder":
        """Set calibration dataset."""
        self._config_dict["dataset_name"] = dataset_name
        return self
    
    def samples(self, num_samples: int) -> "PTQPipelineBuilder":
        """Set number of calibration samples."""
        self._config_dict["num_calib_samples"] = num_samples
        return self
    
    def batch_size(self, batch_size: int) -> "PTQPipelineBuilder":
        """Set calibration batch size."""
        self._config_dict["batch_size"] = batch_size
        return self
    
    def seq_length(self, seq_length: int) -> "PTQPipelineBuilder":
        """Set sequence length."""
        self._config_dict["seq_length"] = seq_length
        return self
    
    def export_to(self, export_path: str) -> "PTQPipelineBuilder":
        """Set export path."""
        self._config_dict["export_path"] = export_path
        return self
    
    def hub_repo(self, repo_id: str) -> "PTQPipelineBuilder":
        """Set HuggingFace Hub repo ID."""
        self._config_dict["hf_repo_id"] = repo_id
        return self
    
    def token(self, hf_token: str) -> "PTQPipelineBuilder":
        """Set HuggingFace token."""
        self._config_dict["hf_token"] = hf_token
        return self
    
    def device(self, device: str) -> "PTQPipelineBuilder":
        """Set computation device."""
        self._config_dict["device"] = device
        return self
    
    def build(self) -> PTQPipeline:
        """Build and return PTQPipeline."""
        config = PTQConfig(**self._config_dict)
        return PTQPipeline(config)
