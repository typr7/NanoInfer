#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys


EOT_TOKEN_IDS = {128001, 128008, 128009}


DEFAULT_CASES = [
    {
        "name": "hello",
        "prompt": "Hello",
    },
    {
        "name": "english_explanation",
        "prompt": "Explain what a transformer is in one sentence.",
    },
    {
        "name": "python_code",
        "prompt": "Write a Python function to reverse a string.",
    },
    {
        "name": "math",
        "prompt": "What is 17 * 23? Answer with only the number.",
    },
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
    {
        "name": "punctuation",
        "prompt": "Continue this text exactly once: A, B, C,",
    },
    {
        "name": "longer_prompt",
        "prompt": (
            "You are checking whether two inference implementations produce "
            "the same greedy token sequence. Briefly describe why comparing "
            "token IDs is stricter than comparing decoded text."
        ),
    },
]


def strip_eot(ids):
    for index, token_id in enumerate(ids):
        if token_id in EOT_TOKEN_IDS:
            return ids[:index]
    return ids


def first_difference(left, right):
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def run_nanoinfer(args, prompt, max_new_tokens):
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
        str(max_new_tokens),
        "--dump-token-ids",
    ]
    output = subprocess.check_output(command, text=True)
    return json.loads(output)


class HuggingFaceRunner:
    def __init__(self, model_name):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing Python dependency for e2e alignment: install torch and transformers"
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

        # Some instruct models ship sampling defaults such as temperature/top_p.
        # This test wants deterministic greedy decoding.
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0

    def generate(self, prompt, max_new_tokens):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        input_ids = input_ids.to(self.device)
        attention_mask = self.torch.ones_like(input_ids, device=self.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = input_ids.shape[-1]
        return output_ids[0, prompt_len:].detach().cpu().tolist()


def normalize_case(raw_case, index):
    if isinstance(raw_case, str):
        return {
            "name": f"case_{index}",
            "prompt": raw_case,
        }

    if not isinstance(raw_case, dict):
        raise ValueError(f"case {index} must be a string or object")

    prompt = raw_case.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"case {index} must contain a non-empty string prompt")

    name = raw_case.get("name", f"case_{index}")
    if not isinstance(name, str) or not name:
        raise ValueError(f"case {index} has invalid name")

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a matrix of NanoInfer vs Hugging Face greedy token alignment cases."
    )
    parser.add_argument("--runner", default="build-release/nano_infer_prompt")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokenizer-script", default="python/tokenizer.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="max_new_tokens values to test",
    )
    parser.add_argument(
        "--case-file",
        help=(
            "Optional JSON or JSONL file. Each case may be either a prompt string "
            "or an object with fields: name, prompt."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first mismatch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases = load_cases(args.case_file)
    hf = HuggingFaceRunner(args.model)

    failures = []
    total = 0

    for case in cases:
        for max_new_tokens in args.lengths:
            total += 1
            name = case["name"]
            prompt = case["prompt"]

            nano_ids = strip_eot(run_nanoinfer(args, prompt, max_new_tokens))
            hf_ids = strip_eot(hf.generate(prompt, max_new_tokens))

            diff_index = first_difference(nano_ids, hf_ids)
            if diff_index is None:
                print(
                    f"[PASS] {name} max_new_tokens={max_new_tokens} "
                    f"matched={len(nano_ids)}"
                )
                continue

            failure = {
                "case": name,
                "max_new_tokens": max_new_tokens,
                "first_difference": diff_index,
                "nanoinfer": nano_ids,
                "huggingface": hf_ids,
            }
            failures.append(failure)

            print(f"[FAIL] {name} max_new_tokens={max_new_tokens}")
            print(f"  first_difference={diff_index}")
            print(f"  nanoinfer={nano_ids}")
            print(f"  huggingface={hf_ids}")

            if args.fail_fast:
                print(f"[SUMMARY] failed 1 / {total} cases")
                return 1

    if failures:
        print(f"[SUMMARY] failed {len(failures)} / {total} cases")
        return 1

    print(f"[SUMMARY] passed {total} / {total} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
