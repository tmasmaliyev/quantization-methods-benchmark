import logging
from typing import Optional, Union

import torch


logger = logging.getLogger(__name__)


class VLLMInference:
    """vLLM-based inference engine for quantized models.
    
    This class provides a convenient wrapper around vLLM for
    running inference on quantized models with high throughput.
    
    vLLM automatically detects ModelOpt FP8 checkpoints and
    applies appropriate optimizations.
    
    Attributes:
        llm: vLLM LLM instance
        SamplingParams: vLLM SamplingParams class
    """
    
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        **kwargs,
    ):
        """Initialize vLLM inference engine.
        
        Args:
            model_path: Path to quantized model or HuggingFace repo ID
            max_model_len: Maximum context length
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type ("auto", "float16", "bfloat16")
            trust_remote_code: Trust remote code in model config
            **kwargs: Additional arguments passed to vLLM LLM
            
        Raises:
            ImportError: If vLLM is not installed
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is required for inference. "
                "Install with: pip install vllm"
            ) from e
        
        logger.info(f"Loading model from {model_path}")
        
        self.llm = LLM(
            model=model_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )
        self.SamplingParams = SamplingParams
        
        logger.info("Model loaded successfully!")
    
    def generate(
        self,
        prompts: Union[str, list[str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> list[str]:
        """Generate text completions for prompts.
        
        Args:
            prompts: Single prompt string or list of prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter (-1 = disabled)
            stop: List of stop strings
            **kwargs: Additional SamplingParams arguments
            
        Returns:
            List of generated text strings
        """
        # Handle single prompt
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Configure sampling
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            **kwargs,
        )
        
        logger.info(f"Generating for {len(prompts)} prompts...")
        
        # Generate
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract text from outputs
        results = [output.outputs[0].text for output in outputs]
        
        return results
    
    def generate_with_metadata(
        self,
        prompts: Union[str, list[str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> list[dict]:
        """Generate with full output metadata.
        
        Returns detailed information including token counts,
        finish reasons, and timing.
        
        Args:
            prompts: Single prompt or list of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional sampling parameters
            
        Returns:
            List of dicts with text, tokens, finish_reason, etc.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            result = {
                "prompt": output.prompt,
                "text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
                "num_tokens": len(output.outputs[0].token_ids),
            }
            results.append(result)
        
        return results
    
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Chat-style generation with message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional sampling parameters
            
        Returns:
            Assistant's response text
        """
        # Build conversation string
        # This is a simple implementation - production should use
        # the model's actual chat template
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        
        parts.append("Assistant:")
        prompt = "\n".join(parts)
        
        outputs = self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        return outputs[0]


class TransformersInference:
    """HuggingFace Transformers-based inference for quantized models.
    
    Provides inference using standard HuggingFace transformers,
    useful for debugging or when vLLM is not available.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        use_compile: bool = True,
    ):
        """Initialize Transformers inference.
        
        Args:
            model_path: Path to model or HuggingFace repo ID
            device: Target device
            torch_dtype: Model dtype (auto-detected if None)
            use_compile: Whether to use torch.compile
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        
        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Optionally compile for faster inference
        if use_compile and device == "cuda":
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        
        self.device = device
        logger.info("Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """Generate text completion.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling (False = greedy)
            **kwargs: Additional generate() arguments
            
        Returns:
            Generated text (prompt + completion)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        **kwargs,
    ) -> list[str]:
        """Generate completions for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum new tokens per prompt
            **kwargs: Additional generate() arguments
            
        Returns:
            List of generated texts
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
