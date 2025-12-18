import argparse
import sys
import os
import logging


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="GPTQ Quantization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic INT8 quantization
    gptq-quantize --model meta-llama/Llama-2-7b-hf --bits 8

"""
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name or local path (default: meta-llama/Llama-2-7b-hf)"
    )
    parser.add_argument(
        "--bits", "-b",
        type=int,
        default=8,
        choices=[4, 8],
        help="Number of bits for quantization (default: 8)"
    )
    parser.add_argument(
        "--group-size", "-g",
        type=int,
        default=128,
        help="Group size for quantization (default: 128)"
    )
    
    # Calibration settings
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="wikitext",
        help="Calibration dataset name (default: wikitext)"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)"
    )
    parser.add_argument(
        "--seq-length", "-s",
        type=int,
        default=512,
        help="Sequence length for calibration (default: 512)"
    )
    
    # Output settings
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./quantized_model_gptq",
        help="Output directory for quantized model (default: ./quantized_model_gptq)"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace Hub repo ID for upload (format: username/repo_name)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace authentication token (defaults to HF_TOKEN env var)"
    )
    
    # Advanced settings
    parser.add_argument(
        "--desc-act",
        action="store_true",
        help="Use desc_act (activation order heuristic)"
    )
    parser.add_argument(
        "--no-sym",
        action="store_true",
        help="Disable symmetric quantization"
    )
    parser.add_argument(
        "--damp-percent",
        type=float,
        default=0.01,
        help="Damping percentage for GPTQ (default: 0.01)"
    )
    
    # Control flags
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip generation test after quantization"
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Skip local save"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip HuggingFace Hub upload"
    )
    parser.add_argument(
        "--test-prompt",
        type=str,
        default="Hello world",
        help="Prompt to use for generation test (default: 'Hello world')"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    return parser


def main(args=None):
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args(args)
    
    # Configure logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    elif args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    
    # Import here to avoid slow startup for --help
    from .config import GPTQConfig
    from .pipeline import GPTQPipeline
    
    # Get HF token from argument or environment variable
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Create configuration
    config = GPTQConfig(
        model_name=args.model,
        bits=args.bits,
        group_size=args.group_size,
        dataset_name=args.dataset,
        num_calib_samples=args.samples,
        seq_length=args.seq_length,
        export_path=args.output,
        hf_repo_id=args.hf_repo,
        hf_token=hf_token,
        desc_act=args.desc_act,
        sym=not args.no_sym,
        damp_percent=args.damp_percent,
    )
    
    # Run pipeline
    pipeline = GPTQPipeline(config)
    
    try:
        results = pipeline.run(
            test_prompt=args.test_prompt,
            skip_test=args.skip_test,
            skip_save=args.skip_save,
            skip_upload=args.skip_upload,
        )
        
        # Print summary
        if not args.quiet:
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            if results["export_path"]:
                print(f"Export Path: {results['export_path']}")
            if results["hub_url"]:
                print(f"Hub URL: {results['hub_url']}")
            if results["test_generation"]:
                print(f"Test Output: {results['test_generation'][:100]}...")
            if results["setup_time"]:
                print(f"Setup Time: {results['setup_time']:.2f}s")
            print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())