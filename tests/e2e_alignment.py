#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys


EOT_TOKEN_IDS = {128001, 128008, 128009}


def strip_eot(ids):
    for index, token_id in enumerate(ids):
        if token_id in EOT_TOKEN_IDS:
            return ids[:index]
    return ids


def run_nanoinfer(args):
    command = [
        args.runner,
        "--weights",
        args.weights,
        "--prompt",
        args.prompt,
        "--tokenizer-model",
        args.model,
        "--tokenizer-script",
        args.tokenizer_script,
        "--python",
        args.python,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dump-token-ids",
    ]
    output = subprocess.check_output(command, text=True)
    return json.loads(output)


def run_huggingface(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency for e2e alignment: install torch and transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(device),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[-1]
    return output_ids[0, prompt_len:].detach().cpu().tolist()


def first_difference(left, right):
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare NanoInfer greedy token IDs against Hugging Face."
    )
    parser.add_argument("--runner", default="build-release/nano_infer_prompt")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokenizer-script", default="python/tokenizer.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    nano_ids = strip_eot(run_nanoinfer(args))
    hf_ids = strip_eot(run_huggingface(args))

    diff_index = first_difference(nano_ids, hf_ids)
    if diff_index is not None:
        print("Token mismatch")
        print(f"first_difference={diff_index}")
        print(f"nanoinfer={nano_ids}")
        print(f"huggingface={hf_ids}")
        return 1

    print(f"[PASS] matched {len(nano_ids)} generated token IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
