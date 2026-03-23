"""Models for external reward signal learning from instruction-answer pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModel


TRAVEL_MODES: List[str] = [
    "Auto",
    "Riding",
    "Subway",
    "Bus",
    "Subway&Bus",
    "Taxi",
    "Cycling",
    "Walk",
]

MODE_ALIASES: Dict[str, str] = {
    "auto": "Auto",
    "riding": "Riding",
    "subway": "Subway",
    "bus": "Bus",
    "subway&bus": "Subway&Bus",
    "subway and bus": "Subway&Bus",
    "taxi": "Taxi",
    "cycling": "Cycling",
    "walk": "Walk",
}

MODE_TO_ID = {name: idx for idx, name in enumerate(TRAVEL_MODES)}
ID_TO_MODE = {idx: name for idx, name in enumerate(TRAVEL_MODES)}


@dataclass
class LabelEncoder:
    mode_to_id: Dict[str, int]

    def encode(self, answer: str) -> int:
        normalized = MODE_ALIASES.get(answer.strip().lower(), answer.strip())
        if normalized not in self.mode_to_id:
            raise ValueError(f"Unknown travel mode label: {answer}")
        return self.mode_to_id[normalized]


class BertTravelModeClassifier(nn.Module):
    def __init__(self, bert_name: str = "bert-base-uncased", num_labels: int = 8, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(bert_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))
        return logits


class FrozenLLMTravelModeClassifier(nn.Module):
    """Frozen small LLM encoder + trainable classification head."""

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        num_labels: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(llm_name, trust_remote_code=True)
        # 冻结参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        hidden = hidden * mask
        summed = hidden.sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-6)
        return summed / count

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        return logits


def build_model(model_type: str, model_name: str, num_labels: int) -> nn.Module:
    if model_type == "bert":
        return BertTravelModeClassifier(bert_name=model_name, num_labels=num_labels)
    if model_type == "frozen_llm":
        return FrozenLLMTravelModeClassifier(llm_name=model_name, num_labels=num_labels)
    raise ValueError(f"Unsupported model_type: {model_type}")
