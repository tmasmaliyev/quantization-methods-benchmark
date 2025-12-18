from comparison_testing import VLLMQuantizationTester

if __name__ == '__main__':
    # Test single model
    tester = VLLMQuantizationTester(
        model_path="taharmasmaliyev07/Llama-2-7b-hf-gptq-int8",
        method_name="INT8 GPTQ",
        bits_per_weight=8.0,
    )
    tester.load_model()
    tester.measure_all_metrics()
    tester.print_results()