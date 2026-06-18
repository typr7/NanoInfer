# NanoInfer

NanoInfer is a lightweight large language model inference framework written in C++17 and CUDA. It implements most of the operators required by `Llama-3.2-1B-Instruct` directly in CUDA, including token embedding, RMSNorm, RoPE, KV cache updates, grouped-query attention (GQA), causal softmax, SwiGLU, residual addition.

![demo](pic/demo.gif)

## Current Scope

- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Weight format: Hugging Face SafeTensors
- Inference mode: single-batch interactive chat with greedy decoding
- Maximum context length: 4096 tokens
- Data type: BF16 weights and intermediate tensors; FP32 is used where required by kernels or cuBLAS paths

## Roadmap

- [x] Single-batch inference for Llama-3.2-1B-Instruct: SafeTensors weight loading, BF16 CUDA kernels, KV cache, and greedy decoding
- [x] Correctness validation: verify layer/block tensors and end-to-end greedy token sequences against PyTorch/Hugging Face
- [x] Benchmarking: report TTFT, TPOT
- [ ] FlashAttention
- [ ] Continuous Batching + PagedAttention
- [ ] Weight-only INT8/INT4 quantized inference
- [ ] ...

## Usage

### Build

1. Install the Python dependencies:

```sh
python3 -m pip install -r python/requirements.txt
```

2. Configure and build the project:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

### Prepare Model Weights

NanoInfer currently supports only [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). Support for additional models will be added later.

To obtain the model weights, visit the model's [Hugging Face page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), request access, and download the weights after access is granted:

```sh
hf download meta-llama/Llama-3.2-1B-Instruct model.safetensors --local-dir model_weights
```

### Run

```sh
./build-release/nano_infer --weights model_weights/model.safetensors
```

## Benchmark

See [benchmark_ttft_tpot.py](https://github.com/typr7/NanoInfer/blob/benchmark-ttft-tpot/benchmarks/benchmark_ttft_tpot.py) for benchmark.

Test Configuration:

- Hardware: RTX 4060 Laptop GPU (8GB)
- Model: Llama-3.2-1B-Instruct
- Batch Size: 1
- Input Lengths: 128, 512, 1024 tokens
- Output Length: 256 tokens
- Warmup Runs: 3
- Measured Runs: 10
- Decoding: greedy
- Timing: model loading and tokenization excluded

Performance Results:

| Input Tokens | Inference Engine | TTFT (ms) | TPOT (ms/token) |
| ---: | --- | ---: | ---: |
| 128 | transformers | **18.098** | 14.140 |
| 128 | NanoInfer | 21.381 | **11.054** |
| 512 | transformers | **53.660** | 15.134 |
| 512 | NanoInfer | 54.957 | **11.582** |
| 1024 | transformers | 108.003 | 14.980 |
| 1024 | NanoInfer | **102.190** | **12.519** |

## Correctness Validation

| Test Name | Test Standard | Test Result |
| --- | --- | --- |
| Layer/block tensor alignment validation | Compare NanoInfer per-layer hidden states and final RMSNorm states with Hugging Face. Values must stay within explicit BF16 tolerances for max error, RMS error, and cosine similarity. | Passed: 6 / 6 prompt-step pairs. All checked tensors stayed within tolerance. |
| E2E greedy token sequence validation | Compare generated token IDs with Hugging Face greedy decoding. Exact matches pass directly; long-sequence divergences pass only when the NanoInfer token is in Hugging Face top-k with a small logit margin. | Passed: 36 / 36 cases. 32 matched exactly; 4 were accepted as near-cache divergences with rank 2 and margin <= 0.125. |

## References

This project was inspired by [jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm).
