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
        "--dump-top-k-fp32",
        str(args.top_k),
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
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0

    def prompt_ids(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

    def generate(self, prompt, max_new_tokens):
        input_ids = self.prompt_ids(prompt)
        attention_mask = self.torch.ones_like(input_ids, device=self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        prompt_len = input_ids.shape[-1]
        generated = output.sequences[0, prompt_len:].detach().cpu().tolist()
        scores = [
            score[0].detach().float().cpu()
            for score in output.scores
        ]
        return generated, scores

    def full_recompute_logits(self, prompt, generated_prefix):
        input_ids = self.prompt_ids(prompt)
        if generated_prefix:
            prefix = self.torch.tensor(
                [generated_prefix],
                dtype=input_ids.dtype,
                device=self.device,
            )
            input_ids = self.torch.cat([input_ids, prefix], dim=1)

        with self.torch.no_grad():
            output = self.model(input_ids, use_cache=False, return_dict=True)
        return output.logits[0, -1, :].detach().float().cpu()

    def token_text(self, token_id):
        return repr(self.tokenizer.decode([token_id]))


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


def token_rank(logits, token_id, torch):
    score = logits[token_id]
    ids = torch.arange(logits.numel(), dtype=torch.long)
    better = (logits > score) | ((logits == score) & (ids < token_id))
    return int(better.sum().item()) + 1


def top_token(logits):
    return int(logits.argmax().item())


def near_reference(logits, reference_id, candidate_id, args, torch):
    reference_score = float(logits[reference_id].item())
    candidate_score = float(logits[candidate_id].item())
    rank = token_rank(logits, candidate_id, torch)
    margin = max(0.0, reference_score - candidate_score)
    return {
        "accepted": rank <= args.top_k and margin <= args.margin_tolerance,
        "rank": rank,
        "margin": margin,
        "reference_score": reference_score,
        "candidate_score": candidate_score,
    }


def trace_step(trace, step):
    for item in trace:
        if item.get("step") == step:
            return item
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run NanoInfer vs Hugging Face e2e alignment. Exact token matches "
            "pass immediately; first divergences may pass only when the NanoInfer "
            "token is within an explicit top-k and logit-margin neighborhood."
        )
    )
    parser.add_argument("--runner", default="build-release/nano_infer_prompt")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokenizer-script", default="python/tokenizer.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--case-file")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--margin-tolerance", type=float, default=0.25)
    parser.add_argument("--fail-fast", action="store_true")
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

            nano_result = run_nanoinfer(args, prompt, max_new_tokens)
            nano_ids = strip_eot(nano_result["generated_token_ids"])
            hf_generated, hf_scores = hf.generate(prompt, max_new_tokens)
            hf_ids = strip_eot(hf_generated)

            diff_index = first_difference(nano_ids, hf_ids)
            if diff_index is None:
                print(
                    f"[PASS exact] {name} max_new_tokens={max_new_tokens} "
                    f"matched={len(nano_ids)}"
                )
                continue

            if diff_index >= len(nano_ids) or diff_index >= len(hf_ids):
                reason = (
                    f"length divergence first_difference={diff_index} "
                    f"nano_len={len(nano_ids)} hf_len={len(hf_ids)}"
                )
                failures.append((name, max_new_tokens, reason))
                print(f"[FAIL] {name} max_new_tokens={max_new_tokens} {reason}")
                if args.fail_fast:
                    print(f"[SUMMARY] failed 1 / {total} cases")
                    return 1
                continue

            if diff_index >= len(hf_scores):
                reason = f"missing Hugging Face score for step {diff_index}"
                failures.append((name, max_new_tokens, reason))
                print(f"[FAIL] {name} max_new_tokens={max_new_tokens} {reason}")
                if args.fail_fast:
                    print(f"[SUMMARY] failed 1 / {total} cases")
                    return 1
                continue

            nano_token = nano_ids[diff_index]
            hf_token = hf_ids[diff_index]
            common_prefix = hf_ids[:diff_index]
            cache_logits = hf_scores[diff_index]
            full_logits = hf.full_recompute_logits(prompt, common_prefix)
            cache_check = near_reference(
                cache_logits,
                hf_token,
                nano_token,
                args,
                hf.torch,
            )
            full_top = top_token(full_logits)
            full_check = near_reference(
                full_logits,
                full_top,
                nano_token,
                args,
                hf.torch,
            )
            nano_trace = trace_step(nano_result["trace"], diff_index)
            nano_fp32_top = None
            if nano_trace and nano_trace.get("top_logits"):
                nano_fp32_top = nano_trace["top_logits"][0]["token_id"]

            if cache_check["accepted"]:
                print(
                    f"[PASS near-cache] {name} max_new_tokens={max_new_tokens} "
                    f"first_difference={diff_index} "
                    f"nano={nano_token}{hf.token_text(nano_token)} "
                    f"hf={hf_token}{hf.token_text(hf_token)} "
                    f"rank={cache_check['rank']} margin={cache_check['margin']:.6g} "
                    f"nano_fp32_top={nano_fp32_top}"
                )
                continue

            if full_check["accepted"]:
                print(
                    f"[PASS near-full] {name} max_new_tokens={max_new_tokens} "
                    f"first_difference={diff_index} "
                    f"nano={nano_token}{hf.token_text(nano_token)} "
                    f"hf={hf_token}{hf.token_text(hf_token)} "
                    f"full_top={full_top}{hf.token_text(full_top)} "
                    f"rank={full_check['rank']} margin={full_check['margin']:.6g} "
                    f"nano_fp32_top={nano_fp32_top}"
                )
                continue

            reason = (
                f"first_difference={diff_index} "
                f"nano={nano_token}{hf.token_text(nano_token)} "
                f"hf={hf_token}{hf.token_text(hf_token)} "
                f"cache_rank={cache_check['rank']} "
                f"cache_margin={cache_check['margin']:.6g} "
                f"full_top={full_top}{hf.token_text(full_top)} "
                f"full_rank={full_check['rank']} "
                f"full_margin={full_check['margin']:.6g} "
                f"nano_fp32_top={nano_fp32_top}"
            )
            failures.append((name, max_new_tokens, reason))
            print(f"[FAIL] {name} max_new_tokens={max_new_tokens} {reason}")

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
