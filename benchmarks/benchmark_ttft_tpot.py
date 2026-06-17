#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "Explain what a GPU kernel is in one paragraph."
DEFAULT_INPUT_LENGTHS = [128, 512, 1024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TTFT and TPOT for NanoInfer and transformers."
    )
    parser.add_argument(
        "--backend",
        choices=("nanoinfer", "transformers", "both"),
        default="both",
        help="Backend to benchmark.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User prompt text used for the benchmark.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Read the benchmark prompt from a text file.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "--weights",
        default="model_weights/model.safetensors",
        help="NanoInfer safetensors weight path.",
    )
    parser.add_argument(
        "--nano-binary",
        default="build-release/nano_infer",
        help="Path to the NanoInfer executable.",
    )
    parser.add_argument(
        "--tokenizer-script",
        default="python/tokenizer.py",
        help="Path to NanoInfer tokenizer.py.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=64,
        help="Maximum generated tokens per run.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=non_negative_int,
        default=1,
        help="Untimed warmup runs.",
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=3,
        help="Measured runs.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for transformers benchmark.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Torch dtype for transformers.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result JSON.",
    )
    parser.add_argument(
        "--output-json",
        help="Write the full result JSON to this path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward NanoInfer diagnostic stderr.",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run a multi-input-length benchmark suite.",
    )
    parser.add_argument(
        "--input-lengths",
        nargs="+",
        type=positive_int,
        default=DEFAULT_INPUT_LENGTHS,
        help="Target prompt token lengths for --suite.",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text()
    return args.prompt


def run_nanoinfer(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    command = [
        args.nano_binary,
        "--benchmark-json",
        "--weights",
        args.weights,
        "--tokenizer-script",
        args.tokenizer_script,
        "--tokenizer-model",
        args.model,
        "--python",
        sys.executable,
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-runs",
        str(args.warmup_runs),
        "--runs",
        str(args.runs),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if args.verbose and completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            "NanoInfer benchmark failed with exit code "
            f"{completed.returncode}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def resolve_torch_dtype(torch_module: Any, dtype_name: str, device: Any) -> Any:
    if dtype_name == "auto":
        return torch_module.bfloat16 if device.type == "cuda" else torch_module.float32
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[dtype_name]


def sync_device(torch_module: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def eos_id_set(model: Any, tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    for candidate in (
        getattr(model.generation_config, "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple, set)):
            ids.update(int(item) for item in candidate if item is not None)
        else:
            ids.add(int(candidate))
    return ids


def encode_chat_prompt(tokenizer: Any, prompt: str, device: Any) -> tuple[Any, Any]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    attention_mask = input_ids.new_ones(input_ids.shape)
    return input_ids, attention_mask


def count_chat_tokens(tokenizer: Any, prompt: str) -> int:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(ids)
    return len(tokenizer(prompt).input_ids)


def make_prompt_for_token_target(tokenizer: Any, target_tokens: int) -> str:
    seed = (
        "The GPU executes tensor kernels while the model reads cached attention "
        "states and produces one deterministic token at a time."
    )
    low = 1
    high = 1
    while count_chat_tokens(tokenizer, " ".join([seed] * high)) < target_tokens:
        high *= 2

    while low < high:
        mid = (low + high + 1) // 2
        prompt = " ".join([seed] * mid)
        if count_chat_tokens(tokenizer, prompt) <= target_tokens:
            low = mid
        else:
            high = mid - 1

    prompt = " ".join([seed] * low)
    filler_words = [
        "latency",
        "throughput",
        "prefill",
        "decode",
        "attention",
        "cache",
        "kernel",
        "stream",
    ]
    best_prompt = prompt
    best_delta = abs(count_chat_tokens(tokenizer, best_prompt) - target_tokens)

    for _ in range(target_tokens):
        improved = False
        for word in filler_words:
            candidate = (best_prompt + " " + word).strip()
            token_count = count_chat_tokens(tokenizer, candidate)
            delta = abs(token_count - target_tokens)
            if token_count <= target_tokens and delta < best_delta:
                best_prompt = candidate
                best_delta = delta
                improved = True
                break
        if not improved:
            break

    return best_prompt


def transformers_once(
    torch_module: Any,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    max_new_tokens: int,
    eos_ids: set[int],
    device: Any,
) -> dict[str, Any]:
    generated_tokens = 0
    decode_tokens = 0
    decode_ms_total = 0.0
    stopped_eos = False

    with torch_module.inference_mode():
        sync_device(torch_module, device)
        ttft_start = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sync_device(torch_module, device)
        ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

        past_key_values = outputs.past_key_values
        next_token_id = int(next_token.item())
        if next_token_id in eos_ids:
            stopped_eos = True
        else:
            generated_tokens = 1

        current_attention_mask = torch_module.cat(
            [
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1)),
            ],
            dim=1,
        )

        while generated_tokens < max_new_tokens and not stopped_eos:
            sync_device(torch_module, device)
            decode_start = time.perf_counter()
            outputs = model(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sync_device(torch_module, device)
            decode_ms = (time.perf_counter() - decode_start) * 1000.0

            past_key_values = outputs.past_key_values
            next_token_id = int(next_token.item())
            if next_token_id in eos_ids:
                stopped_eos = True
                break

            decode_ms_total += decode_ms
            decode_tokens += 1
            generated_tokens += 1
            current_attention_mask = torch_module.cat(
                [
                    current_attention_mask,
                    current_attention_mask.new_ones((current_attention_mask.shape[0], 1)),
                ],
                dim=1,
            )

    return {
        "prompt_tokens": int(input_ids.shape[-1]),
        "generated_tokens": generated_tokens,
        "decode_tokens": decode_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": decode_ms_total / decode_tokens if decode_tokens else 0.0,
        "decode_ms_total": decode_ms_total,
        "stopped_eos": stopped_eos,
    }


def summarize(
    backend: str,
    runs: list[dict[str, Any]],
    prompt_tokens: int,
    max_new_tokens: int,
    warmup_runs: int,
) -> dict[str, Any]:
    ttft_ms_total = sum(float(run["ttft_ms"]) for run in runs)
    decode_ms_total = sum(float(run["decode_ms_total"]) for run in runs)
    decode_tokens_total = sum(int(run["decode_tokens"]) for run in runs)
    generated_tokens_total = sum(int(run["generated_tokens"]) for run in runs)
    return {
        "backend": backend,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "runs": runs,
        "summary": {
            "runs": len(runs),
            "ttft_ms_avg": ttft_ms_total / len(runs),
            "tpot_ms_avg": (
                decode_ms_total / decode_tokens_total
                if decode_tokens_total
                else 0.0
            ),
            "decode_ms_total": decode_ms_total,
            "decode_tokens_total": decode_tokens_total,
            "generated_tokens_avg": generated_tokens_total / len(runs),
        },
    }


def run_transformers(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers benchmark requires torch and transformers in this Python env"
        ) from exc

    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message=".*past_key_values.*deprecated.*")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for transformers but is not available")

    torch_dtype = resolve_torch_dtype(torch, args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()

    input_ids, attention_mask = encode_chat_prompt(tokenizer, prompt, device)
    eos_ids = eos_id_set(model, tokenizer)

    for _ in range(args.warmup_runs):
        transformers_once(
            torch,
            model,
            input_ids,
            attention_mask,
            args.max_new_tokens,
            eos_ids,
            device,
        )

    runs = [
        transformers_once(
            torch,
            model,
            input_ids,
            attention_mask,
            args.max_new_tokens,
            eos_ids,
            device,
        )
        for _ in range(args.runs)
    ]

    return summarize(
        "transformers",
        runs,
        int(input_ids.shape[-1]),
        args.max_new_tokens,
        args.warmup_runs,
    )


def load_prompt_tokenizer(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "benchmark suite prompt generation requires transformers in this Python env"
        ) from exc

    return AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = load_prompt_tokenizer(args)
    cases = []

    for target_tokens in args.input_lengths:
        prompt = make_prompt_for_token_target(tokenizer, target_tokens)
        actual_tokens = count_chat_tokens(tokenizer, prompt)
        case_results: list[dict[str, Any]] = []

        if args.backend in ("nanoinfer", "both"):
            case_results.append(run_nanoinfer(args, prompt))
        if args.backend in ("transformers", "both"):
            case_results.append(run_transformers(args, prompt))

        cases.append(
            {
                "target_input_tokens": target_tokens,
                "actual_input_tokens": actual_tokens,
                "prompt": prompt,
                "results": case_results,
                "comparison": build_comparison(case_results),
            }
        )

    return {
        "model": args.model,
        "python": sys.executable,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "cases": cases,
    }


def build_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_backend = {item["backend"]: item for item in results}
    if "nanoinfer" not in by_backend or "transformers" not in by_backend:
        return {}

    nano_summary = by_backend["nanoinfer"]["summary"]
    transformers_summary = by_backend["transformers"]["summary"]
    return {
        "ttft_speedup_vs_transformers": (
            transformers_summary["ttft_ms_avg"] / nano_summary["ttft_ms_avg"]
            if nano_summary["ttft_ms_avg"]
            else None
        ),
        "tpot_speedup_vs_transformers": (
            transformers_summary["tpot_ms_avg"] / nano_summary["tpot_ms_avg"]
            if nano_summary["tpot_ms_avg"]
            else None
        ),
    }


def print_table(results: list[dict[str, Any]]) -> None:
    print(
        f"{'backend':<14} {'prompt':>8} {'gen avg':>8} "
        f"{'TTFT ms':>12} {'TPOT ms/token':>15}"
    )
    for item in results:
        summary = item["summary"]
        print(
            f"{item['backend']:<14} "
            f"{item['prompt_tokens']:>8} "
            f"{summary['generated_tokens_avg']:>8.1f} "
            f"{summary['ttft_ms_avg']:>12.3f} "
            f"{summary['tpot_ms_avg']:>15.3f}"
        )


def print_suite_table(output: dict[str, Any]) -> None:
    print(
        f"{'input':>8} {'backend':<14} {'gen avg':>8} "
        f"{'TTFT ms':>12} {'TPOT ms/token':>15}"
    )
    for case in output["cases"]:
        for item in case["results"]:
            summary = item["summary"]
            print(
                f"{case['actual_input_tokens']:>8} "
                f"{item['backend']:<14} "
                f"{summary['generated_tokens_avg']:>8.1f} "
                f"{summary['ttft_ms_avg']:>12.3f} "
                f"{summary['tpot_ms_avg']:>15.3f}"
            )


def main() -> int:
    args = parse_args()

    if args.suite:
        output = run_suite(args)
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(output, indent=2) + "\n")
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print_suite_table(output)
        return 0

    prompt = load_prompt(args)

    results: list[dict[str, Any]] = []
    if args.backend in ("nanoinfer", "both"):
        results.append(run_nanoinfer(args, prompt))
    if args.backend in ("transformers", "both"):
        results.append(run_transformers(args, prompt))

    output = {
        "prompt": prompt,
        "model": args.model,
        "python": sys.executable,
        "results": results,
        "comparison": build_comparison(results),
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, indent=2) + "\n")

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print_table(results)
        if output["comparison"]:
            comparison = output["comparison"]
            print(
                "speedup vs transformers: "
                f"TTFT {comparison['ttft_speedup_vs_transformers']:.3f}x, "
                f"TPOT {comparison['tpot_speedup_vs_transformers']:.3f}x"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
