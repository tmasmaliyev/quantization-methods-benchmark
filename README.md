# LLM Quantization Toolkit

A toolkit for quantizing Large Language Models using GPTQ and PTQ methods.

## Installation

```bash
pip install torch transformers datasets huggingface_hub tqdm gptqmodel vllm
pip install nvidia-modelopt[hf]  # For PTQ only
```

## CLI Usage

### GPTQ Quantization

```bash
# Full example with all options
python -m gptq_toolkit \
    --model meta-llama/Llama-2-7b-hf \
    --bits 8 \
    --group-size 128 \
    --dataset wikitext \
    --samples 512 \
    --seq-length 512 \
    --output ./llama2-7b-gptq-int8 \
    --hf-repo username/Llama-2-7b-hf-gptq-int8

# Skip generation test
python -m gptq_toolkit --model meta-llama/Llama-2-7b-hf --bits 8 --skip-test
```

#### GPTQ Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | `meta-llama/Llama-2-7b-hf` | Model name or path |
| `--bits` | `-b` | `8` | Quantization bits (4 or 8) |
| `--group-size` | `-g` | `128` | Group size |
| `--dataset` | `-d` | `wikitext` | Calibration dataset |
| `--samples` | `-n` | `256` | Number of calibration samples |
| `--seq-length` | `-s` | `512` | Sequence length |
| `--output` | `-o` | `./quantized_model_gptq` | Output directory |
| `--hf-repo` | | | HuggingFace repo ID for upload |
| `--hf-token` | | | HuggingFace token |
| `--desc-act` | | | Use desc_act heuristic |
| `--no-sym` | | | Disable symmetric quantization |
| `--damp-percent` | | `0.01` | Damping percentage |
| `--skip-test` | | | Skip generation test |
| `--skip-save` | | | Skip local save |
| `--skip-upload` | | | Skip HuggingFace upload |
| `--test-prompt` | | `"Hello world"` | Test prompt |
| `--verbose` | `-v` | | Verbose logging |
| `--quiet` | `-q` | | Suppress output |

---

### FP8/INT8 PTQ Quantization

```bash
python -m fptq_toolkit \
    --model google/gemma-2-9b-it \
    --format fp8 \
    --dataset wikitext \
    --samples 512 \
    --batch-size 4 \
    --seq-length 512 \
    --output ./gemma-2-9b-it \
    --hf-repo username/gemma-2-9b-it-fp8
```

#### FP8/INT8 PTQ Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | `meta-llama/Llama-2-7b-hf` | Model name or path |
| `--format` | `-f` | `fp8` | Quantization format (fp8, int8) |
| `--dataset` | `-d` | `wikitext` | Calibration dataset |
| `--samples` | `-n` | `512` | Number of calibration samples |
| `--batch-size` | `-b` | `4` | Batch size |
| `--seq-length` | `-s` | `1024` | Sequence length |
| `--output` | `-o` | `./quantized_model` | Output directory |
| `--hf-repo` | | | HuggingFace repo ID for upload |
| `--hf-token` | | | HuggingFace token |
| `--skip-test` | | | Skip generation test |
| `--skip-export` | | | Skip checkpoint export |
| `--skip-upload` | | | Skip HuggingFace upload |
| `--test-prompt` | | `"Hello world"` | Test prompt |
| `--verbose` | `-v` | | Verbose logging |
| `--quiet` | `-q` | | Suppress output |

---

### Benchmarking

To test a quantized model, edit `main.py` with your HuggingFace repository path and run:

```bash
python main.py
```

Example `main.py`:

```python
from comparison_testing import VLLMQuantizationTester

if __name__ == '__main__':
    tester = VLLMQuantizationTester(
        model_path="username/Llama-2-7b-hf-gptq-int8",  # Your HF repo
        method_name="INT8 GPTQ",
        bits_per_weight=8.0,
    )
    tester.load_model()
    tester.measure_all_metrics()
    tester.print_results()
```
