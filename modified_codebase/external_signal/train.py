from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from model import ID_TO_MODE, MODE_TO_ID, TRAVEL_MODES, LabelEncoder, build_model


@dataclass
class TrainConfig:
    model_type: str
    model_name: str
    max_length: int
    batch_size: int
    epochs: int
    lr: float
    seed: int


class InstructionAnswerDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_instruction_answer_pairs(path: str) -> pd.DataFrame:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return pd.DataFrame(records)
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    if "instruction" not in df.columns or "answer" not in df.columns:
        raise ValueError("Dataset must contain instruction and answer columns.")

    encoder = LabelEncoder(mode_to_id=MODE_TO_ID)
    texts = df["instruction"].astype(str).tolist()
    labels = [encoder.encode(x) for x in df["answer"].astype(str).tolist()]
    return texts, labels


def evaluate(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    data_df = load_instruction_answer_pairs(args.data_path)
    texts, labels = preprocess(data_df)

    x_train, x_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_set = InstructionAnswerDataset(x_train, y_train, tokenizer, args.max_length)
    val_set = InstructionAnswerDataset(x_val, y_val, tokenizer, args.max_length)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model_type, args.model_name, num_labels=len(TRAVEL_MODES))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": args.model_type,
                    "model_name": args.model_name,
                    "label_to_id": MODE_TO_ID,
                    "id_to_label": ID_TO_MODE,
                    "config": asdict(
                        TrainConfig(
                            model_type=args.model_type,
                            model_name=args.model_name,
                            max_length=args.max_length,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            lr=args.lr,
                            seed=args.seed,
                        )
                    ),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train external travel-mode classifier.")
    parser.add_argument("--data_path", type=str, required=True, help="Processed instruction-answer dataset (json/csv).")
    parser.add_argument("--output_dir", type=str, default="modified_codebase/external_signal/checkpoints")
    parser.add_argument("--model_type", choices=["bert", "frozen_llm"], default="bert")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
