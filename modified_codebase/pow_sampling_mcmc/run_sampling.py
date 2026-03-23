import argparse
import os
import random

import pandas as pd
import torch
import transformers

from framework import (
    HFDatasetAdapter,
    JSONListAdapter,
    run_framework,
)


def build_adapter(args):
    if args.dataset_source == "json":
        return JSONListAdapter(
            path=args.dataset_path,
            question_key=args.question_key,
            answer_key=args.answer_key,
        )

    return HFDatasetAdapter(
        dataset_name=args.dataset_name,
        subset=args.dataset_subset,
        split=args.dataset_split,
        question_key=args.question_key,
        answer_key=args.answer_key,
    )


def main():
    parser = argparse.ArgumentParser(description="Generic power-sampling MCMC runner")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_source", choices=["json", "hf"], default="json")

    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--dataset_subset", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")

    parser.add_argument("--question_key", type=str, default="prompt")
    parser.add_argument("--answer_key", type=str, default="answer")

    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=str, default="modified_codebase/pow_sampling_mcmc/results.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).to(device)

    adapter = build_adapter(args)
    data = adapter.load()
    if args.limit > 0:
        data = data[: args.limit]

    results = run_framework(
        dataset=data,
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=args.temperature,
        mcmc_steps=args.mcmc_steps,
        max_new_tokens=args.max_new_tokens,
        cot=args.cot,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Saved {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
