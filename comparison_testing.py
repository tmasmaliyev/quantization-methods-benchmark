import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
from tqdm import tqdm

import torch


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MethodResults:
    """Results container for quantization method benchmarks."""
    method_name: str
    model_path: str = ""
    bits_per_weight: float = 0.0
    model_size_mb: float = 0.0
    compression_ratio: float = 1.0
    inference_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    peak_memory_gb: float = 0.0
    perplexity: float = 0.0
    setup_time_seconds: float = 0.0
    generation_sample: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return f"""
Method: {self.method_name}
Model: {self.model_path}
----------------------------------------
Bits per Weight: {self.bits_per_weight}
Model Size: {self.model_size_mb:.2f} MB
Compression Ratio: {self.compression_ratio:.2f}x
----------------------------------------
Inference Latency: {self.inference_latency_ms:.2f} ms
Throughput: {self.throughput_tokens_per_sec:.2f} tokens/sec
Peak Memory: {self.peak_memory_gb:.2f} GB
----------------------------------------
Perplexity: {self.perplexity:.2f}
Setup Time: {self.setup_time_seconds:.2f} s
"""


# =============================================================================
# vLLM-based Quantization Tester
# =============================================================================

class VLLMQuantizationTester:
    """
    vLLM-based tester for quantized models.
    
    Uses vLLM for high-performance inference benchmarking.
    
    Attributes:
        model_path: HuggingFace model path or local path
        method_name: Name of the quantization method
        bits_per_weight: Bits per weight for this quantization
    """
    
    def __init__(
        self,
        model_path: str,
        method_name: str,
        bits_per_weight: float,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        """
        Initialize vLLM tester.
        
        Args:
            model_path: Path to model (HF repo or local)
            method_name: Name for this quantization method
            bits_per_weight: Bits per weight
            max_model_len: Maximum context length
            gpu_memory_utilization: Fraction of GPU memory to use
            dtype: Data type ("auto", "float16", "bfloat16")
            trust_remote_code: Trust remote code in model
        """
        self.model_path = model_path
        self.method_name = method_name
        self.bits_per_weight = bits_per_weight
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        
        self.llm = None
        self.sampling_params = None
        self.results = MethodResults(
            method_name=method_name,
            model_path=model_path,
            bits_per_weight=bits_per_weight,
        )
        self.setup_start_time = None
        
        logger.info(f"Initializing {method_name}")
    
    def load_model(self) -> None:
        """Load model with vLLM."""
        from vllm import LLM, SamplingParams
        
        logger.info(f"Loading model: {self.model_path}")
        self.setup_start_time = time.time()
        
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        
        self.sampling_params = SamplingParams(
            max_tokens=100,
            temperature=0.0,  # Greedy for reproducibility
        )
        
        setup_time = time.time() - self.setup_start_time
        self.results.setup_time_seconds = setup_time
        
        logger.info(f"✓ Model loaded in {setup_time:.2f}s")
    
    def get_method_name(self) -> str:
        return self.method_name
    
    def get_bits_per_weight(self) -> float:
        return self.bits_per_weight
    
    def estimate_model_size_mb(self) -> float:
        """
        Estimate model size based on bits per weight.
        
        Note: vLLM doesn't expose direct model size, so we estimate
        based on parameter count and quantization bits.
        """
        # Llama-2-7B has ~7B parameters
        # Actual size = params * bits / 8 / 1024^2
        estimated_params = 7e9  # 7B parameters
        size_mb = (estimated_params * self.bits_per_weight) / 8 / (1024**2)
        return size_mb
    
    def calculate_compression_ratio(self, baseline_bits: float = 16.0) -> float:
        """Calculate compression ratio vs baseline."""
        return baseline_bits / self.bits_per_weight
    
    def measure_inference_latency(
        self,
        num_samples: int = 100,
        max_tokens: int = 20,
    ) -> float:
        """
        Measure average inference latency in milliseconds.
        
        Args:
            num_samples: Number of inference runs
            max_tokens: Tokens to generate per run
            
        Returns:
            Average latency in milliseconds
        """
        from vllm import SamplingParams
        
        logger.info(f"Measuring inference latency ({num_samples} samples)...")
        
        if self.llm is None:
            logger.error("Model not loaded!")
            return 0.0
        
        prompt = "The quick brown fox jumps over the lazy dog"
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )
        
        # Warmup
        logger.info("  Warming up...")
        for _ in tqdm(range(10), desc="Warmup"):
            _ = self.llm.generate([prompt], sampling_params)
        
        # Measure
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        logger.info(f"  Running {num_samples} inference samples...")
        latencies = []
        
        for _ in tqdm(range(num_samples), desc="Measuring latency"):
            start = time.perf_counter()
            _ = self.llm.generate([prompt], sampling_params)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency_ms = sum(latencies) / len(latencies)
        
        logger.info(f"  ✓ Average latency: {avg_latency_ms:.2f} ms")
        return avg_latency_ms
    
    def measure_throughput(
        self,
        duration_seconds: int = 30,
        max_tokens: int = 50,
        batch_size: int = 8,
    ) -> float:
        """
        Measure throughput in tokens per second.
        
        Args:
            duration_seconds: How long to run the test
            max_tokens: Tokens per generation
            batch_size: Number of prompts per batch
            
        Returns:
            Tokens per second
        """
        from vllm import SamplingParams
        
        logger.info(f"Measuring throughput ({duration_seconds}s test, batch_size={batch_size})...")
        
        if self.llm is None:
            logger.error("Model not loaded!")
            return 0.0
        
        prompts = ["The future of artificial intelligence is"] * batch_size
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )
        
        start_time = time.time()
        total_tokens = 0
        iterations = 0
        
        while (time.time() - start_time) < duration_seconds:
            outputs = self.llm.generate(prompts, sampling_params)
            for output in outputs:
                total_tokens += len(output.outputs[0].token_ids)
            iterations += 1
        
        elapsed = time.time() - start_time
        throughput = total_tokens / elapsed
        
        logger.info(f"  ✓ Throughput: {throughput:.2f} tokens/sec ({iterations} iterations)")
        return throughput
    
    def measure_memory_usage(self) -> float:
        """
        Measure peak GPU memory usage in GB.
        
        Returns:
            Peak memory in GB
        """
        from vllm import SamplingParams
        
        logger.info("Measuring memory usage...")
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping memory measurement")
            return 0.0
        
        if self.llm is None:
            logger.error("Model not loaded!")
            return 0.0
        
        torch.cuda.reset_peak_memory_stats()
        
        # Run inference to capture peak memory
        prompt = "Test prompt for memory measurement " * 10
        sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
        _ = self.llm.generate([prompt], sampling_params)
        
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        logger.info(f"  ✓ Peak memory: {peak_memory_gb:.2f} GB")
        return peak_memory_gb
    
    def calculate_perplexity(self, max_samples: Optional[int] = 100) -> float:
        """
        Calculate perplexity on WikiText-2 test set using vLLM.
        
        Uses vLLM's prompt_logprobs to get token log probabilities
        for perplexity calculation.
        
        Args:
            max_samples: Maximum samples to evaluate
            
        Returns:
            Perplexity score
        """
        from vllm import SamplingParams
        
        logger.info(f"Calculating perplexity (max {max_samples} samples)...")
        
        if self.llm is None:
            logger.error("Model not loaded!")
            return 0.0
        
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            logger.info(f"  Dataset loaded: {len(dataset)} total samples")
        except Exception as e:
            logger.error(f"Failed to load WikiText dataset: {e}")
            return 0.0
        
        # Use prompt_logprobs to get log probabilities
        sampling_params = SamplingParams(
            max_tokens=1,  # We only need logprobs, not generation
            temperature=0.0,
            prompt_logprobs=1,  # Get logprobs for prompt tokens
        )
        
        total_log_prob = 0.0
        total_tokens = 0
        evaluated = 0
        
        # Collect valid texts
        texts = []
        for example in dataset:
            text = example.get("text", "")
            if len(text.strip()) >= 10:
                texts.append(text[:512])  # Truncate to 512 chars
                if max_samples and len(texts) >= max_samples:
                    break
        
        logger.info(f"  Processing {len(texts)} samples...")
        
        # Process in batches for efficiency
        batch_size = 8
        for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                outputs = self.llm.generate(batch_texts, sampling_params)
                
                for output in outputs:
                    if output.prompt_logprobs is not None:
                        # Sum log probabilities (skip first token - no prior context)
                        for token_logprob in output.prompt_logprobs[1:]:
                            if token_logprob is not None:
                                # Get the logprob of the actual token
                                for token_id, logprob_obj in token_logprob.items():
                                    total_log_prob += logprob_obj.logprob
                                    total_tokens += 1
                                    break  # Only count the actual token
                    
                    evaluated += 1
                    
            except Exception as e:
                logger.warning(f"  Batch error: {e}")
                continue
        
        if total_tokens == 0:
            logger.error("  No tokens evaluated!")
            return 0.0
        
        # Perplexity = exp(-avg_log_prob)
        avg_log_prob = total_log_prob / total_tokens
        perplexity = torch.exp(torch.tensor(-avg_log_prob)).item()
        
        logger.info(f"  ✓ Perplexity: {perplexity:.2f} ({evaluated} samples, {total_tokens:,} tokens)")
        
        return perplexity
    
    def measure_all_metrics(
        self,
        baseline_bits: float = 16.0,
        include_perplexity: bool = True,
        perplexity_samples: int = 100,
    ) -> MethodResults:
        """
        Measure all metrics for this model.
        
        Args:
            baseline_bits: Baseline bits per weight for compression ratio
            include_perplexity: Whether to calculate perplexity
            perplexity_samples: Number of samples for perplexity
            
        Returns:
            MethodResults with all metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"MEASURING ALL METRICS: {self.method_name}")
        logger.info(f"{'='*60}\n")
        
        if self.llm is None:
            logger.error("Model not loaded! Call load_model() first.")
            return self.results
        
        # Basic info
        self.results.model_size_mb = self.estimate_model_size_mb()
        self.results.bits_per_weight = self.bits_per_weight
        self.results.compression_ratio = self.calculate_compression_ratio(baseline_bits)
        
        # Speed metrics
        self.results.inference_latency_ms = self.measure_inference_latency()
        self.results.throughput_tokens_per_sec = self.measure_throughput()
        
        # Memory
        self.results.peak_memory_gb = self.measure_memory_usage()
        
        # Perplexity (optional)
        if include_perplexity:
            self.results.perplexity = self.calculate_perplexity(max_samples=perplexity_samples)
        
        logger.info(f"\n✓ All metrics measured for {self.method_name}")
        
        return self.results
    
    def test_generation(
        self,
        prompt: str = "The future of AI is",
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """
        Test text generation with this model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        from vllm import SamplingParams
        
        if self.llm is None:
            logger.error("Model not loaded!")
            return ""
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        full_output = prompt + generated_text
        self.results.generation_sample = full_output
        
        print(f"\n{'='*60}")
        print(f"GENERATION TEST: {self.method_name}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Output: {full_output}")
        print(f"{'='*60}\n")
        
        return full_output
    
    def print_results(self) -> None:
        """Print formatted results."""
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.method_name}")
        print(f"{'='*60}")
        print(self.results)
        print(f"{'='*60}\n")
    
    def save_results(self, filepath: Optional[str] = None) -> None:
        """Save results to JSON file."""
        if filepath is None:
            safe_name = self.method_name.lower().replace(" ", "_")
            filepath = f"{safe_name}_results.json"
        
        with open(filepath, "w") as f:
            json.dump(self.results.to_dict(), f, indent=2)
        
        logger.info(f"✓ Results saved to {filepath}")

def print_comparison_summary(results: Dict[str, MethodResults]) -> None:
    """Print a comparison summary table."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Method':<20} {'Bits':>6} {'Latency(ms)':>12} {'Throughput':>12} {'Memory(GB)':>11} {'Perplexity':>11}")
    print("-" * 80)
    
    # Rows
    for name, r in results.items():
        print(
            f"{name:<20} "
            f"{r.bits_per_weight:>6.1f} "
            f"{r.inference_latency_ms:>12.2f} "
            f"{r.throughput_tokens_per_sec:>12.2f} "
            f"{r.peak_memory_gb:>11.2f} "
            f"{r.perplexity:>11.2f}"
        )
    
    print(f"{'='*80}\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for comparison testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Quantization Comparison Testing")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--name", type=str, required=True, help="Method name")
    parser.add_argument("--bits", type=float, required=True, help="Bits per weight")
    parser.add_argument("--no-perplexity", action="store_true", help="Skip perplexity")
    parser.add_argument("--perplexity-samples", type=int, default=100)
    
    args = parser.parse_args()
    
    tester = VLLMQuantizationTester(
        model_path=args.model,
        method_name=args.name,
        bits_per_weight=args.bits,
    )
    
    tester.load_model()
    tester.test_generation()
    tester.measure_all_metrics(
        include_perplexity=not args.no_perplexity,
        perplexity_samples=args.perplexity_samples,
    )
    tester.print_results()
    tester.save_results()


if __name__ == "__main__":
    main()