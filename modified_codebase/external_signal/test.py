from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import TRAVEL_MODES, build_model
from train import InstructionAnswerDataset, preprocess


def load_instruction_answer_pairs(path: str) -> pd.DataFrame:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return pd.DataFrame(records)
    return pd.read_csv(path)


def evaluate(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model_type = checkpoint["model_type"]
    model_name = checkpoint["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_df = load_instruction_answer_pairs(args.data_path)
    texts, labels = preprocess(data_df)
    dataset = InstructionAnswerDataset(texts, labels, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(model_type, model_name, num_labels=len(TRAVEL_MODES))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels_batch.cpu().tolist())

    correct = sum(int(p == y) for p, y in zip(all_preds, all_labels))
    acc = correct / max(len(all_labels), 1)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=TRAVEL_MODES, digits=4, zero_division=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test external travel-mode classifier.")
    parser.add_argument("--data_path", type=str, required=True, help="Test dataset path (json/csv).")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
