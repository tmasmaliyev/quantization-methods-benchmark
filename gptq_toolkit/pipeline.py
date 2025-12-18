import logging
import time
from pathlib import Path
from typing import Optional, List, Dict

import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

from .config import GPTQConfig
from .dataset import GPTQDatasetFactory


logger = logging.getLogger(__name__)


class GPTQPipeline:
    """Complete GPTQ quantization pipeline.
    
    Orchestrates the full GPTQ workflow:
    1. Load tokenizer from HuggingFace
    2. Prepare calibration data
    3. Load and quantize model with GPTQ
    4. Test quantized model generation
    5. Save model locally
    6. (Optional) Upload to HuggingFace Hub
    
    Attributes:
        config: Pipeline configuration
        tokenizer: Loaded tokenizer
        quantized_model: Quantized model (None until quantize() called)
        setup_time: Time taken for quantization
    """
    
    def __init__(self, config: GPTQConfig):
        """Initialize GPTQ pipeline.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.tokenizer = None
        self.quantized_model = None
        self.setup_time = 0.0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    
    def load_tokenizer(self) -> None:
        """Load tokenizer from HuggingFace.
        
        Loads the tokenizer specified in config.model_name and sets up
        pad_token if needed for decoder-only models.
        
        Raises:
            ImportError: If transformers or huggingface_hub not installed
        """
        from huggingface_hub import login
        
        # Login if token provided
        if self.config.hf_token:
            logger.info("Logging into HuggingFace Hub...")
            login(token=self.config.hf_token)
        
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True
        )
        
        # Set pad token for decoder-only models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        logger.info("✓ Tokenizer loaded")
    
    def prepare_calibration_data(self) -> List[Dict[str, torch.Tensor]]:
        """Prepare calibration dataset.
        
        Creates a calibration dataset using the configured dataset,
        sample count, and sequence length.
        
        Returns:
            List of calibration samples
            
        Raises:
            RuntimeError: If tokenizer not loaded yet
        """
        if self.tokenizer is None:
            raise RuntimeError("Must call load_tokenizer() first")
        
        return GPTQDatasetFactory.create(
            dataset_name=self.config.dataset_name,
            tokenizer=self.tokenizer,
            num_samples=self.config.num_calib_samples,
            seq_length=self.config.seq_length,
        )
    
    def quantize(self, calibration_data: List[Dict[str, torch.Tensor]]) -> None:
        """Apply GPTQ quantization to model.
        
        Loads the unquantized model and applies GPTQ quantization
        using the provided calibration data.
        
        Args:
            calibration_data: List of calibration samples
            
        Raises:
            ImportError: If gptqmodel is not installed
        """
        logger.info("\nApplying GPTQ quantization...")
        logger.info(f"Bits: {self.config.bits}")
        logger.info(f"Group Size: {self.config.group_size}")
        logger.info(f"Calibration Samples: {len(calibration_data)}")
        
        start_time = time.time()
        
        # Create quantization config
        quantize_config = QuantizeConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            desc_act=self.config.desc_act,
            sym=self.config.sym,
            damp_percent=self.config.damp_percent,
        )
        
        # Load unquantized model (forced to CPU by GPTQ)
        logger.info("Loading model for quantization...")
        model = GPTQModel.load(
            self.config.model_name,
            quantize_config=quantize_config
        )
        
        # Quantize model
        logger.info("Running GPTQ quantization algorithm...")
        model.quantize(calibration_data)
        
        self.quantized_model = model
        self.setup_time = time.time() - start_time
        
        logger.info(f"✓ Quantization complete! ({self.setup_time:.2f}s)")
    
    def save_locally(self) -> Path:
        """Save quantized model locally.
        
        Exports the quantized model in HuggingFace-compatible format,
        including tokenizer files.
        
        Returns:
            Path to saved model directory
            
        Raises:
            RuntimeError: If quantized model not available
        """
        if self.quantized_model is None:
            raise RuntimeError("Must call quantize() first")
        
        export_path = Path(self.config.export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving quantized model to {export_path}...")
        
        # Save model
        self.quantized_model.save(str(export_path))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(export_path))
        
        logger.info(f"✓ Model saved to {export_path}")
        
        # List saved files
        files = list(export_path.iterdir())
        logger.info(f"Saved files: {[f.name for f in files]}")
        
        return export_path
    
    def upload_to_hub(self) -> Optional[str]:
        """Upload to HuggingFace Hub if configured.
        
        Uploads the quantized model to HuggingFace Hub
        if hf_repo_id is set in config.
        
        Returns:
            URL of uploaded model, or None if not configured
            
        Raises:
            RuntimeError: If quantized model not available
        """
        if self.config.hf_repo_id is None:
            logger.info("No HuggingFace repo configured, skipping upload")
            return None
        
        if self.quantized_model is None:
            raise RuntimeError("Must call quantize() first")
        
        logger.info(f"Uploading to HuggingFace Hub: {self.config.hf_repo_id}")
        
        # Push model
        GPTQModel.push_to_hub(
            repo_id=self.config.hf_repo_id,
            quantized_path=self.config.export_path,
            token=self.config.hf_token,
        )
        
        # Push tokenizer
        self.tokenizer.push_to_hub(
            self.config.hf_repo_id,
            token=self.config.hf_token,
        )
        
        url = f"https://huggingface.co/{self.config.hf_repo_id}"
        logger.info(f"✓ Upload complete! Model available at: {url}")
        
        return url
    
    def test_generation(
        self,
        prompt: str = "Hello world",
        max_new_tokens: int = 20,
    ) -> str:
        """Test quantized model with simple generation.
        
        Runs a quick generation test to verify the quantized
        model produces reasonable outputs.
        
        Args:
            prompt: Test prompt to generate from
            max_new_tokens: Number of tokens to generate
            
        Returns:
            Generated text string
            
        Raises:
            RuntimeError: If quantized model not available
        """
        if self.quantized_model is None:
            raise RuntimeError("Must call quantize() first")
        
        logger.info("Testing quantized model generation...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.quantized_model.device
        )
        
        with torch.no_grad():
            outputs = self.quantized_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated: {generated}")
        
        return generated
    
    def run(
        self,
        test_prompt: str = "Hello world",
        skip_test: bool = False,
        skip_save: bool = False,
        skip_upload: bool = False,
    ) -> dict:
        """Execute full GPTQ pipeline.
        
        Runs all steps in sequence:
        1. Load tokenizer
        2. Prepare calibration data
        3. Quantize model
        4. Test generation (optional)
        5. Save locally (optional)
        6. Upload to Hub (optional)
        
        Args:
            test_prompt: Prompt for generation test
            skip_test: Skip generation test
            skip_save: Skip local save
            skip_upload: Skip Hub upload
            
        Returns:
            Dictionary with pipeline results:
            - export_path: Path to saved model
            - hub_url: HuggingFace Hub URL (if uploaded)
            - test_generation: Generated text from test
            - setup_time: Time taken for quantization
            - config: Configuration used
        """
        logger.info("=" * 60)
        logger.info("Starting GPTQ Pipeline")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Bits: {self.config.bits}")
        logger.info(f"Group Size: {self.config.group_size}")
        logger.info("=" * 60)
        
        # Step 1: Load tokenizer
        logger.info("\n[Step 1/5] Loading tokenizer...")
        self.load_tokenizer()
        
        # Step 2: Prepare data
        logger.info("\n[Step 2/5] Preparing calibration data...")
        calibration_data = self.prepare_calibration_data()
        
        # Step 3: Quantize
        logger.info("\n[Step 3/5] Applying GPTQ quantization...")
        self.quantize(calibration_data)
        
        # Step 4: Test
        generated = None
        if not skip_test:
            logger.info("\n[Step 4/5] Testing quantized model...")
            generated = self.test_generation(test_prompt)
        else:
            logger.info("\n[Step 4/5] Skipping test...")
        
        # Step 5: Save
        export_path = None
        if not skip_save:
            logger.info("\n[Step 5/5] Saving model...")
            export_path = self.save_locally()
        else:
            logger.info("\n[Step 5/5] Skipping save...")
        
        # Step 6: Upload
        hub_url = None
        if not skip_upload and not skip_save:
            logger.info("\n[Step 6/5] Uploading to Hub...")
            hub_url = self.upload_to_hub()
        else:
            logger.info("\n[Step 6/5] Skipping upload...")
        
        logger.info("\n" + "=" * 60)
        logger.info("GPTQ Pipeline Complete!")
        logger.info("=" * 60)
        
        return {
            "export_path": str(export_path) if export_path else None,
            "hub_url": hub_url,
            "test_generation": generated,
            "setup_time": self.setup_time,
            "config": self.config.to_dict(),
        }


class GPTQPipelineBuilder:
    """Builder pattern for GPTQPipeline configuration.
    
    Provides a fluent interface for building pipeline configurations.
    """
    
    def __init__(self):
        self._config_dict = {}
    
    def model(self, model_name: str) -> "GPTQPipelineBuilder":
        """Set model name."""
        self._config_dict["model_name"] = model_name
        return self
    
    def bits(self, bits: int) -> "GPTQPipelineBuilder":
        """Set quantization bits."""
        self._config_dict["bits"] = bits
        return self
    
    def group_size(self, group_size: int) -> "GPTQPipelineBuilder":
        """Set group size."""
        self._config_dict["group_size"] = group_size
        return self
    
    def dataset(self, dataset_name: str) -> "GPTQPipelineBuilder":
        """Set calibration dataset."""
        self._config_dict["dataset_name"] = dataset_name
        return self
    
    def samples(self, num_samples: int) -> "GPTQPipelineBuilder":
        """Set number of calibration samples."""
        self._config_dict["num_calib_samples"] = num_samples
        return self
    
    def seq_length(self, seq_length: int) -> "GPTQPipelineBuilder":
        """Set sequence length."""
        self._config_dict["seq_length"] = seq_length
        return self
    
    def export_to(self, export_path: str) -> "GPTQPipelineBuilder":
        """Set export path."""
        self._config_dict["export_path"] = export_path
        return self
    
    def hub_repo(self, repo_id: str) -> "GPTQPipelineBuilder":
        """Set HuggingFace Hub repo ID."""
        self._config_dict["hf_repo_id"] = repo_id
        return self
    
    def token(self, hf_token: str) -> "GPTQPipelineBuilder":
        """Set HuggingFace token."""
        self._config_dict["hf_token"] = hf_token
        return self
    
    def desc_act(self, desc_act: bool) -> "GPTQPipelineBuilder":
        """Set desc_act."""
        self._config_dict["desc_act"] = desc_act
        return self
    
    def sym(self, sym: bool) -> "GPTQPipelineBuilder":
        """Set symmetric quantization."""
        self._config_dict["sym"] = sym
        return self
    
    def damp_percent(self, damp_percent: float) -> "GPTQPipelineBuilder":
        """Set damping percentage."""
        self._config_dict["damp_percent"] = damp_percent
        return self
    
    def build(self) -> GPTQPipeline:
        """Build and return GPTQPipeline."""
        config = GPTQConfig(**self._config_dict)
        return GPTQPipeline(config)
