# NanoInfer

NanoInfer 是一个使用 C++17 和 CUDA 编写的轻量级大语言模型推理框架。它直接用 CUDA 实现了 `Llama-3.2-1B-Instruct` 所需的大部分算子，包括 token embedding、RMSNorm、RoPE、KV cache 更新、分组查询注意力（GQA）、causal softmax、SwiGLU、残差加法。

![demo](pic/demo.gif)

## 当前支持范围

- 模型：`meta-llama/Llama-3.2-1B-Instruct`
- 权重格式：Hugging Face SafeTensors
- 推理方式：单 batch 交互式对话，使用 greedy decoding
- 最大上下文长度：4096 tokens
- 数据类型：权重和中间张量使用 BF16；kernel 或 cuBLAS 路径需要时使用 FP32 计算

## Roadmap

### P0

- [x] Llama-3.2-1B-Instruct 单 batch 推理：SafeTensors 权重加载、BF16 CUDA kernel、KV cache 和 greedy decoding
- [ ] 正确性验证：kernel 单元测试，以及与 PyTorch/Hugging Face 输出对齐
- [ ] Benchmark 与 profiling：统计 TTFT、decode tokens/s、显存占用，并记录 Nsight Compute 热点

### P1

- [ ] Batched prefill 和 batched decode
- [ ] Continuous batching：动态加入和移除请求，支持不同 prompt/generation 长度
- [ ] KV cache 管理：从静态连续缓存演进到 paged/block-based KV cache
- [ ] HTTP/gRPC demo server：暴露简单的 OpenAI-compatible chat/completions API

### P2

- [ ] Decode attention kernel 优化：减少同步和全局内存访问，并给出 profiling 前后对比
- [ ] Prefill attention 优化：替换当前按 head 执行 dense attention 的实现
- [ ] Kernel fusion：融合 RMSNorm、residual、SwiGLU 等热点路径
- [ ] CUDA Graph 和 stream 优化：降低 decode step 的 launch overhead

### P3

- [ ] 配置驱动地加载 Llama-family 模型参数，减少硬编码假设
- [ ] 支持更多 Llama 变体，以及 Qwen、Mistral 等结构相近的模型
- [ ] Weight-only INT8/INT4 量化推理

## 使用方法

### 构建

1. 安装 Python 依赖：

```sh
python3 -m pip install -r python/requirements.txt
```

2. 配置并构建项目：

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

### 准备模型权重

NanoInfer 目前仅支持 [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)，后续会扩展对更多模型的支持。

为了获取模型权重，请访问该模型在 [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 上的主页，申请访问权限，并在获得授权后下载权重：

```sh
hf download meta-llama/Llama-3.2-1B-Instruct model.safetensors --local-dir model_weights
```

### 运行

```sh
./build-release/nano_infer --weights model_weights/model.safetensors
```

## 参考

本项目受到 [jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm) 的启发。
