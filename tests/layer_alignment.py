#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys


DEFAULT_CASES = [
    {
        "name": "chinese",
        "prompt": "用中文简单介绍一下注意力机制。",
    },
    {
        "name": "translation",
        "prompt": "Translate to English: 今天天气很好。",
    },
    {
        "name": "json",
        "prompt": "Return a compact JSON object with fields name and age.",
    },
]


def run_nanoinfer(args, prompt, step):
    command = [
        args.runner,
        "--weights",
        args.weights,
        "--prompt",
        prompt,
        "--tokenizer-model",
        args.model,
        "--tokenizer-script",
        args.tokenizer_script,
        "--python",
        args.python,
        "--max-new-tokens",
        str(step + 1),
        "--dump-layer-hidden-step",
        str(step),
        "--dump-final-norm-step",
        str(step),
    ]
    output = subprocess.check_output(command, text=True)
    return json.loads(output)


def normalize_case(raw_case, index):
    if isinstance(raw_case, str):
        return {
            "name": f"case_{index}",
            "prompt": raw_case,
        }
    if not isinstance(raw_case, dict):
        raise ValueError(f"case {index} must be a string or object")

    name = raw_case.get("name", f"case_{index}")
    prompt = raw_case.get("prompt")
    if not isinstance(name, str) or not name:
        raise ValueError(f"case {index} has invalid name")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"case {index} must contain a non-empty prompt")
    return {
        "name": name,
        "prompt": prompt,
    }


def load_cases(path):
    if path is None:
        return DEFAULT_CASES

    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    if path.endswith(".jsonl"):
        raw_cases = [
            json.loads(line)
            for line in content.splitlines()
            if line.strip()
        ]
    else:
        raw_cases = json.loads(content)

    if not isinstance(raw_cases, list):
        raise ValueError("case file must contain a JSON array or JSONL records")
    return [
        normalize_case(raw_case, index)
        for index, raw_case in enumerate(raw_cases)
    ]


class HuggingFaceInspector:
    def __init__(self, model_name):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing Python dependency for layer alignment: install torch and transformers"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def prompt_ids(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

    def inspect(self, prompt, generated_prefix):
        layer_outputs = []
        handles = []

        def capture_layer(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_outputs.append(hidden[0, -1, :].detach().float().cpu())

        for layer in self.model.model.layers:
            handles.append(layer.register_forward_hook(capture_layer))

        input_ids = self.prompt_ids(prompt)
        if generated_prefix:
            prefix = self.torch.tensor(
                [generated_prefix],
                dtype=input_ids.dtype,
                device=self.device,
            )
            input_ids = self.torch.cat([input_ids, prefix], dim=1)

        try:
            with self.torch.no_grad():
                output = self.model.model(
                    input_ids,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        final_norm = output.last_hidden_state[0, -1, :].detach().float().cpu()
        return layer_outputs, final_norm


def metrics(left, right, torch):
    left_tensor = torch.tensor(left, dtype=torch.float32)
    right_tensor = right.to(dtype=torch.float32)
    diff = left_tensor - right_tensor
    denom = torch.linalg.norm(left_tensor) * torch.linalg.norm(right_tensor)
    cosine = 1.0
    if denom.item() != 0.0:
        cosine = (torch.dot(left_tensor, right_tensor) / denom).item()
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rms": torch.sqrt(torch.mean(diff * diff)).item(),
        "cosine": cosine,
    }


def passes_layer(metric, args):
    return (
        metric["max_abs"] <= args.layer_max_abs
        and metric["rms"] <= args.layer_rms
        and metric["cosine"] >= args.layer_min_cosine
    )


def passes_final(metric, args):
    return (
        metric["max_abs"] <= args.final_max_abs
        and metric["rms"] <= args.final_rms
        and metric["cosine"] >= args.final_min_cosine
    )


def find_step(trace, step):
    for step_trace in trace:
        if step_trace.get("step") == step:
            return step_trace
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare NanoInfer per-layer hidden states and final RMSNorm state "
            "against Hugging Face within explicit BF16 implementation tolerances."
        )
    )
    parser.add_argument("--runner", default="build-release/nano_infer_prompt")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokenizer-script", default="python/tokenizer.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--case-file")
    parser.add_argument("--layer-max-abs", type=float, default=0.15)
    parser.add_argument("--layer-rms", type=float, default=0.020)
    parser.add_argument("--layer-min-cosine", type=float, default=0.9992)
    parser.add_argument("--final-max-abs", type=float, default=0.40)
    parser.add_argument("--final-rms", type=float, default=0.090)
    parser.add_argument("--final-min-cosine", type=float, default=0.9993)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cases = load_cases(args.case_file)
    inspector = HuggingFaceInspector(args.model)
    failures = []
    total = 0

    for case in cases:
        for step in args.steps:
            total += 1
            result = run_nanoinfer(args, case["prompt"], step)
            generated = result["generated_token_ids"]
            step_trace = find_step(result["trace"], step)
            if step_trace is None:
                failures.append((case["name"], step, "missing NanoInfer trace step"))
                print(f"[FAIL] {case['name']} step={step} missing NanoInfer trace step")
                continue

            nano_layers = step_trace.get("layer_hidden_states", [])
            nano_final = step_trace.get("final_norm", [])
            hf_layers, hf_final = inspector.inspect(case["prompt"], generated[:step])

            if len(nano_layers) != len(hf_layers):
                reason = f"layer count mismatch nano={len(nano_layers)} hf={len(hf_layers)}"
                failures.append((case["name"], step, reason))
                print(f"[FAIL] {case['name']} step={step} {reason}")
                continue

            worst_layer = None
            for index, (nano_layer, hf_layer) in enumerate(zip(nano_layers, hf_layers)):
                metric = metrics(nano_layer, hf_layer, inspector.torch)
                if worst_layer is None or metric["rms"] > worst_layer[1]["rms"]:
                    worst_layer = (index, metric)
                if args.verbose:
                    print(
                        f"  layer={index:02d} max_abs={metric['max_abs']:.6g} "
                        f"rms={metric['rms']:.6g} cosine={metric['cosine']:.9f}"
                    )
                if not passes_layer(metric, args):
                    reason = (
                        f"layer {index} outside tolerance: "
                        f"max_abs={metric['max_abs']:.6g} "
                        f"rms={metric['rms']:.6g} "
                        f"cosine={metric['cosine']:.9f}"
                    )
                    failures.append((case["name"], step, reason))

            final_metric = metrics(nano_final, hf_final, inspector.torch)
            if not passes_final(final_metric, args):
                reason = (
                    f"final_norm outside tolerance: "
                    f"max_abs={final_metric['max_abs']:.6g} "
                    f"rms={final_metric['rms']:.6g} "
                    f"cosine={final_metric['cosine']:.9f}"
                )
                failures.append((case["name"], step, reason))

            if failures and failures[-1][0] == case["name"] and failures[-1][1] == step:
                print(f"[FAIL] {case['name']} step={step} {failures[-1][2]}")
                continue

            layer_index, layer_metric = worst_layer
            print(
                f"[PASS] {case['name']} step={step} "
                f"worst_layer={layer_index} layer_rms={layer_metric['rms']:.6g} "
                f"final_rms={final_metric['rms']:.6g} "
                f"final_cosine={final_metric['cosine']:.9f}"
            )

    if failures:
        print(f"[SUMMARY] failed {len(failures)} checks across {total} case/step pairs")
        return 1

    print(f"[SUMMARY] passed {total} / {total} case/step pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
