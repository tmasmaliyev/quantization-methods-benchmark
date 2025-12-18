import logging
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


logger = logging.getLogger(__name__)


class CalibrationRunner:
    """Runs forward pass calibration for Post-Training Quantization.
    
    This class handles the calibration forward loop that ModelOpt uses
    to collect activation statistics (min/max values) for computing
    quantization scaling factors.
    
    The calibration process:
    1. Runs batches through the model in inference mode
    2. ModelOpt hooks capture activation statistics
    3. Statistics are used to compute optimal scaling factors
    
    Attributes:
        model: The model being calibrated
        dataloader: DataLoader providing calibration batches
        device: Target device for computation
        log_interval: Number of batches between progress logs
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        log_interval: int = 10,
    ):
        """Initialize calibration runner.
        
        Args:
            model: Model to calibrate (will be modified in place)
            dataloader: DataLoader providing calibration data
            device: Target device for computation
            log_interval: Number of batches between progress logs
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.log_interval = log_interval
        self._calibration_complete = False
    
    def run(self) -> None:
        """Execute calibration forward loop.
        
        Runs forward passes through the model to collect activation
        statistics for FP8/FP4 scaling factor computation.
        
        This method should be called after quantization hooks have
        been inserted by ModelOpt.
        """
        logger.info("Running calibration forward loop...")
        logger.info(f"Total batches: {len(self.dataloader)}")
        
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(
                self.dataloader,
                desc="Calibrating",
                unit="batch"
            )):
                # Move to device with async transfer
                input_ids = batch["input_ids"].to(
                    self.device,
                    non_blocking=True
                ).contiguous()
                
                # Forward pass - hooks collect statistics
                try:
                    self.model(input_ids=input_ids)
                except Exception as e:
                    logger.error(f"Error in calibration batch {i}: {e}")
                    raise
                
                # Progress logging
                if (i + 1) % self.log_interval == 0:
                    logger.debug(
                        f"Calibrated {i + 1}/{len(self.dataloader)} batches"
                    )
        
        self._calibration_complete = True
        logger.info("Calibration complete!")
    
    def get_forward_loop(self) -> Callable:
        """Get forward loop callable for mtq.quantize().
        
        Returns a callable that can be passed to mtq.quantize()
        as the forward_loop parameter. The callable takes a model
        and runs calibration on it.
        
        Returns:
            Callable[[torch.nn.Module], None] for mtq.quantize()
        """
        def forward_loop(model: torch.nn.Module) -> None:
            # Update model reference (mtq may wrap the model)
            self.model = model
            self.run()
        
        return forward_loop
    
    @property
    def is_complete(self) -> bool:
        """Check if calibration has been completed."""
        return self._calibration_complete
    
    def reset(self) -> None:
        """Reset calibration state for re-running."""
        self._calibration_complete = False


class MultiDatasetCalibrationRunner(CalibrationRunner):
    """Calibration runner that supports multiple dataloaders.
    
    Useful when you want to calibrate on multiple domains or
    dataset types for better generalization.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        dataloaders: list[DataLoader],
        device: str = "cuda",
        log_interval: int = 10,
    ):
        """Initialize multi-dataset calibration runner.
        
        Args:
            model: Model to calibrate
            dataloaders: List of DataLoaders for different datasets
            device: Target device
            log_interval: Batches between progress logs
        """
        # Use first dataloader as primary (for parent class compatibility)
        super().__init__(model, dataloaders[0], device, log_interval)
        self.dataloaders = dataloaders
    
    def run(self) -> None:
        """Execute calibration across all dataloaders."""
        logger.info(f"Running calibration on {len(self.dataloaders)} datasets...")
        
        total_batches = sum(len(dl) for dl in self.dataloaders)
        logger.info(f"Total batches across all datasets: {total_batches}")
        
        self.model.eval()
        batch_count = 0
        
        with torch.no_grad():
            for dataset_idx, dataloader in enumerate(self.dataloaders):
                logger.info(
                    f"Calibrating on dataset {dataset_idx + 1}/{len(self.dataloaders)} "
                    f"({len(dataloader)} batches)"
                )
                
                for batch in tqdm(
                    dataloader,
                    desc=f"Dataset {dataset_idx + 1}",
                    unit="batch"
                ):
                    input_ids = batch["input_ids"].to(
                        self.device,
                        non_blocking=True
                    ).contiguous()
                    
                    self.model(input_ids=input_ids)
                    batch_count += 1
                    
                    if batch_count % self.log_interval == 0:
                        logger.debug(
                            f"Calibrated {batch_count}/{total_batches} batches"
                        )
        
        self._calibration_complete = True
        logger.info("Multi-dataset calibration complete!")
