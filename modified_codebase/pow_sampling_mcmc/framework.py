"""通用的 power-sampling MCMC 框架（仅聚焦 mcmc_power_samp）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import random

import numpy as np
import torch
from torch.nn import functional as F
from datasets import load_dataset


@dataclass
class SampleItem:
    """统一后的样本结构。"""

    question: str
    answer: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class DatasetAdapter:
    """数据集适配器基类：把不同来源的数据转换为 SampleItem。"""

    def load(self) -> List[SampleItem]:
        raise NotImplementedError


@dataclass
class JSONListAdapter(DatasetAdapter):
    """读取本地 JSON 列表数据集。"""

    path: str
    question_key: str = "prompt"
    answer_key: Optional[str] = "answer"

    def load(self) -> List[SampleItem]:
        with open(self.path, "r", encoding="utf-8") as f:
            records = json.load(f)

        items: List[SampleItem] = []
        for row in records:
            items.append(
                SampleItem(
                    question=row[self.question_key],
                    answer=row[self.answer_key] if self.answer_key else None,
                    raw=row,
                )
            )
        return items


@dataclass
class HFDatasetAdapter(DatasetAdapter):
    """读取 HuggingFace datasets。"""

    dataset_name: str
    split: str = "test"
    question_key: str = "question"
    answer_key: Optional[str] = None
    subset: Optional[str] = None

    def load(self) -> List[SampleItem]:
        ds = load_dataset(self.dataset_name, self.subset, split=self.split)
        items: List[SampleItem] = []
        for row in ds:
            items.append(
                SampleItem(
                    question=row[self.question_key],
                    answer=row[self.answer_key] if self.answer_key else None,
                    raw=dict(row),
                )
            )
        return items


class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = model.config.max_position_embeddings


# 低温 proposal q
@torch.no_grad()
def naive_temp(
    sampler: AutoregressiveSampler,
    context: Sequence[int],
    temp: float,
    seq_len: int,
) -> Tuple[List[int], List[float], List[float]]:
    c = len(context)
    input_ids = torch.tensor([list(context)], dtype=torch.long, device=sampler.device)
    output = sampler.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=sampler.tokenizer.eos_token_id,
        pad_token_id=sampler.tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )

    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    proposal = output.sequences[0].tolist()

    idx = tokens.view(unscaled_logits.shape[0], 1, 1)
    # target p^(1/temp) 的未归一化 logprob
    target_log_probs = (
        (1 / temp) * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)
    ).view(-1).tolist()
    # proposal q 的归一化 logprob
    proposal_log_probs = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    return proposal, proposal_log_probs, target_log_probs


# 仅保留 power-sampling MCMC
@torch.no_grad()
def mcmc_power_samp(
    sampler: AutoregressiveSampler,
    context: Sequence[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 16,
) -> Tuple[List[int], float]:
    c = len(context)
    generated = list(context)
    log_q: List[float] = []
    log_target: List[float] = []

    if max_new_tokens % block_num != 0:
        raise ValueError("max_new_tokens 必须可以被 block_num 整除")

    jump_size = max_new_tokens // block_num
    attempts = 0
    accepts = 0

    for _ in range(block_num):
        generated, lp_q, lp_target = naive_temp(
            sampler,
            generated,
            temp=temp,
            seq_len=jump_size + len(generated),
        )
        log_q.extend(lp_q)
        log_target.extend(lp_target)

        for _ in range(mcmc_steps):
            attempts += 1
            t = len(generated)
            idx = random.randint(c, t - 1)

            prop, prop_log_q, prop_log_target = naive_temp(
                sampler,
                generated[:idx],
                temp=temp,
                seq_len=t,
            )
            s = len(prop)
            cur_log_q = log_q[idx - c : s - c]
            cur_log_target = log_target[idx - c : s - c]

            log_r = sum(prop_log_target) + sum(cur_log_q) - sum(cur_log_target) - sum(prop_log_q)
            if np.random.rand() < np.exp(log_r):
                accepts += 1
                generated = prop
                log_q[idx - c :] = prop_log_q
                log_target[idx - c :] = prop_log_target

        if sampler.tokenizer.eos_token_id in generated:
            eos_idx = generated.index(sampler.tokenizer.eos_token_id)
            generated = generated[: eos_idx + 1]
            break

    acceptance_ratio = accepts / attempts if attempts else 0.0
    return generated, acceptance_ratio


def default_prompt_builder(question: str, cot: bool = True) -> str:
    prompt = "Please reason step by step and give the final answer.\n\nQuestion: "
    suffix = "\n\nLet's think step by step." if cot else "\n\nAnswer directly."
    return prompt + question + suffix


def run_framework(
    *,
    dataset: Iterable[SampleItem],
    model,
    tokenizer,
    device: str,
    temperature: float,
    mcmc_steps: int,
    max_new_tokens: int,
    cot: bool = True,
    prompt_builder: Callable[[str, bool], str] = default_prompt_builder,
    postprocess: Optional[Callable[[str], Any]] = None,
) -> List[Dict[str, Any]]:
    sampler = AutoregressiveSampler(model, tokenizer, device)
    results: List[Dict[str, Any]] = []

    for item in dataset:
        prompt_text = prompt_builder(item.question, cot)
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        prefix_ids = [x.item() for x in input_ids[0]]

        output_ids, acc_ratio = mcmc_power_samp(
            sampler,
            context=prefix_ids,
            temp=temperature,
            mcmc_steps=mcmc_steps,
            max_new_tokens=max_new_tokens,
        )

        completion = tokenizer.decode(torch.tensor(output_ids).to("cpu"), skip_special_tokens=True)
        parsed = postprocess(completion) if postprocess else completion

        results.append(
            {
                "question": item.question,
                "correct_answer": item.answer,
                "mcmc_completion": completion,
                "mcmc_parsed": parsed,
                "acceptance_ratio": acc_ratio,
                "raw": item.raw,
            }
        )

    return results
