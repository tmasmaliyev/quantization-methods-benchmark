import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="FP4/FP8 Post-Training Quantization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic FP8 quantization
    ptq-quantize --model meta-llama/Llama-2-7b-hf --format fp8

    # FP4 with custom settings
    ptq-quantize --model meta-llama/Llama-2-7b-hf --format nvfp4 \\
        --samples 1024 --batch-size 2 --output ./llama2-nvfp4

    # With HuggingFace Hub upload
    ptq-quantize --model meta-llama/Llama-2-7b-hf --format fp8 \\
        --hf-repo username/llama2-fp8 --hf-token $HF_TOKEN
        """,
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name or local path (default: meta-llama/Llama-2-7b-hf)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="fp8",
        choices=["fp8", "int8"],
        help="Quantization format (default: fp8)"
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
        default=512,
        help="Number of calibration samples (default: 512)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Calibration batch size (default: 4)"
    )
    parser.add_argument(
        "--seq-length", "-s",
        type=int,
        default=1024,
        help="Sequence length for calibration (default: 1024)"
    )
    
    # Output settings
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./quantized_model",
        help="Output directory for quantized model (default: ./quantized_model)"
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
    
    # Control flags
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip generation test after quantization"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip checkpoint export"
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
    import logging
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
    import os
    from .config import PTQConfig
    from .pipeline import PTQPipeline
    
    # Get HF token from argument or environment variable
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Create configuration
    config = PTQConfig(
        model_name=args.model,
        quant_format=args.format,
        dataset_name=args.dataset,
        num_calib_samples=args.samples,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        export_path=args.output,
        hf_repo_id=args.hf_repo,
        hf_token=hf_token,
    )
    
    # Run pipeline
    pipeline = PTQPipeline(config)
    
    try:
        results = pipeline.run(
            test_prompt=args.test_prompt,
            skip_test=args.skip_test,
            skip_export=args.skip_export,
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