"""通用的 power-sampling MCMC 框架（仅聚焦 mcmc_power_samp）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import random
from pathlib import Path
import importlib.util

import numpy as np
import torch
from torch.nn import functional as F
from datasets import load_dataset
import transformers


EXTERNAL_SIGNAL_MODEL_PATH = Path(__file__).resolve().parents[1] / "external_signal" / "model.py"
spec = importlib.util.spec_from_file_location("external_signal_model_module", EXTERNAL_SIGNAL_MODEL_PATH)
external_signal_model_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(external_signal_model_module)

build_external_model = external_signal_model_module.build_model
MODE_TO_ID = external_signal_model_module.MODE_TO_ID


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


@dataclass
class ExternalSignalBundle:
    model: torch.nn.Module
    tokenizer: Any
    mode_to_id: Dict[str, int]
    max_length: int = 512


def load_external_signal_bundle(
    checkpoint_path: str,
    device: str,
    max_length: int = 512,
) -> ExternalSignalBundle:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint["model_type"]
    model_name = checkpoint["model_name"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_external_model(model_type, model_name, num_labels=len(MODE_TO_ID))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    mode_to_id = checkpoint.get("label_to_id", MODE_TO_ID)
    return ExternalSignalBundle(model=model, tokenizer=tokenizer, mode_to_id=mode_to_id, max_length=max_length)


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


def _extract_mode_name_from_text(text: str, mode_names: Sequence[str]) -> Optional[str]:
    lower_text = text.lower()
    for mode in mode_names:
        if mode.lower() in lower_text:
            return mode
    return None


def _find_last_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    if len(subsequence) == 0 or len(subsequence) > len(sequence):
        return -1
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start
    return -1


def _lm_log_prob_for_mode_from_sequence(
    sampler: AutoregressiveSampler,
    sequence_ids: Sequence[int],
    mode_name: str,
) -> float:
    mode_ids = sampler.tokenizer.encode(mode_name, add_special_tokens=False)
    start_idx = _find_last_subsequence(sequence_ids, mode_ids)
    if start_idx <= 0:
        return -20.0

    input_ids = torch.tensor([list(sequence_ids)], dtype=torch.long, device=sampler.device)
    with torch.no_grad():
        outputs = sampler.model(input_ids=input_ids)
        token_log_probs = F.log_softmax(outputs.logits[0], dim=-1)

    score = 0.0
    for i, token_id in enumerate(mode_ids):
        token_position = start_idx + i
        if token_position == 0:
            continue
        score += token_log_probs[token_position - 1, token_id].item()
    return score / max(len(mode_ids), 1)


def _external_log_probs_for_modes(
    bundle: ExternalSignalBundle,
    instruction_text: str,
    device: str,
) -> Dict[str, float]:
    enc = bundle.tokenizer(
        instruction_text,
        truncation=True,
        padding="max_length",
        max_length=bundle.max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(logits, dim=-1)[0]

    result: Dict[str, float] = {}
    for mode_name, mode_idx in bundle.mode_to_id.items():
        result[mode_name] = log_probs[mode_idx].item()
    return result


def _joint_reward(
    sampler: AutoregressiveSampler,
    sequence_ids: Sequence[int],
    completion_text: str,
    external_mode_log_probs: Dict[str, float],
) -> float:
    mode_name = _extract_mode_name_from_text(completion_text, list(external_mode_log_probs.keys()))
    if mode_name is None:
        return -20.0

    ext_log_prob = external_mode_log_probs[mode_name]
    lm_log_prob = _lm_log_prob_for_mode_from_sequence(sampler, sequence_ids, mode_name)
    return -abs(ext_log_prob - lm_log_prob)


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


@torch.no_grad()
def mcmc_power_samp_with_external_signal(
    sampler: AutoregressiveSampler,
    context: Sequence[int],
    instruction_text: str,
    external_bundle: ExternalSignalBundle,
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 16,
    reward_weight: float = 1.0,
) -> Tuple[List[int], float]:
    c = len(context)
    generated = list(context)
    log_q: List[float] = []
    log_target: List[float] = []

    if max_new_tokens % block_num != 0:
        raise ValueError("max_new_tokens 必须可以被 block_num 整除")

    ext_mode_log_probs = _external_log_probs_for_modes(external_bundle, instruction_text, sampler.device)

    jump_size = max_new_tokens // block_num
    attempts = 0
    accepts = 0
    current_reward = -20.0

    for _ in range(block_num):
        generated, lp_q, lp_target = naive_temp(
            sampler,
            generated,
            temp=temp,
            seq_len=jump_size + len(generated),
        )
        log_q.extend(lp_q)
        log_target.extend(lp_target)

        current_completion = sampler.tokenizer.decode(generated, skip_special_tokens=True)
        current_reward = _joint_reward(
            sampler=sampler,
            sequence_ids=generated,
            completion_text=current_completion,
            external_mode_log_probs=ext_mode_log_probs,
        )

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

            prop_completion = sampler.tokenizer.decode(prop, skip_special_tokens=True)
            prop_reward = _joint_reward(
                sampler=sampler,
                sequence_ids=prop,
                completion_text=prop_completion,
                external_mode_log_probs=ext_mode_log_probs,
            )

            log_r_base = sum(prop_log_target) + sum(cur_log_q) - sum(cur_log_target) - sum(prop_log_q)
            log_r = log_r_base + reward_weight * (prop_reward - current_reward)

            if np.random.rand() < np.exp(log_r):
                accepts += 1
                generated = prop
                log_q[idx - c :] = prop_log_q
                log_target[idx - c :] = prop_log_target
                current_reward = prop_reward

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
