import argparse
import json


def load_tokenizer(model):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing Python dependency: install transformers to use tokenizer.py") from exc

    return AutoTokenizer.from_pretrained(model)


def normalize_token_ids(ids):
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise SystemExit("Expected a single prompt from chat template")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="Text to tokenize")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--decode", action="store_true", help="Decode mode: IDs → text")
    parser.add_argument("--ids", nargs="+", type=int, help="Token IDs to decode")
    parser.add_argument("--ids-file", help="JSON file containing token IDs to decode")
    parser.add_argument("--skip-special-tokens", action="store_true")
    parser.add_argument("--chat-template", action="store_true", help="Apply a chat template to messages")
    parser.add_argument("--messages-file", help="JSON file containing [{role, content}, ...]")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)

    if args.decode:
        ids = args.ids
        if args.ids_file:
            with open(args.ids_file) as f:
                ids = json.load(f)
        if not ids:
            parser.error("--decode requires --ids or --ids-file")
        text = tokenizer.decode(
            ids,
            skip_special_tokens=args.skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        print(text, end="")
    elif args.chat_template:
        if not args.messages_file:
            parser.error("--chat-template requires --messages-file")
        with open(args.messages_file) as f:
            messages = json.load(f)
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        print(json.dumps(normalize_token_ids(ids)))
    else:
        if not args.text:
            parser.error("Provide text to tokenize")
        ids = tokenizer.encode(args.text)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(ids, f)
            print(f"Wrote {len(ids)} tokens to {args.output}")
        else:
            space_delimited_tokens = ""
            for index, id in enumerate(ids):
                space_delimited_tokens += f"{id}"
                if index != len(ids) - 1:
                    space_delimited_tokens += " "
            print(space_delimited_tokens)


if __name__ == "__main__":
    main()
