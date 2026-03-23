"""Build instruction-answer dataset for travel mode choice modeling.

This script converts tabular trip-choice records into English instruction-answer pairs.
It can optionally call a strong LLM (via OpenAI-compatible API) to polish instructions.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # optional dependency
    OpenAI = None

# mode名字
MODE_CODE_TO_NAME = {
    "A": "Auto",
    "R": "Riding",
    "S": "Subway",
    "B": "Bus",
    "SB": "Subway&Bus",
    "T": "Taxi",
    "C": "Cycling",
    "W": "Walk",
}
# 出发目的
TPN_MAP = {
    1: "Business",
    2: "Leisure",
    3: "Commuting",
    4: "Schooling",
    5: "Shopping",
    6: "Other",
    7: "Returning home",
}
# departure time mapping dict
DT_MAP = {
    1: "00:00-06:00",
    2: "06:00-12:00",
    3: "12:00-18:00",
    4: "18:00-24:00",
}


@dataclass
class PromptFields:
    male: int
    age: str
    income: str
    lic: int
    purpose: str
    depart_time: str
    trip_order: int
    first_trip: int
    stt: float
    std: float
    stc: float
    nsub: float
    nbus: float
    npark: float
    nbike: float
    origin_name: str
    destination_name: str

# age mapping
def infer_age_group(row: pd.Series) -> str:
    if int(row.get("AGE01", 0)) == 1:
        return "0-19"
    if int(row.get("AGE23", 0)) == 1:
        return "20-39"
    if int(row.get("AGE45", 0)) == 1:
        return "40-59"
    if int(row.get("AGE6", 0)) == 1:
        return "60+"
    return "Unknown"

# income group
def infer_income_group(row: pd.Series) -> str:
    if int(row.get("INC1", 0)) == 1:
        return "<1400"
    if int(row.get("INC2", 0)) == 1:
        return "1400-3500"
    if int(row.get("INC3", 0)) == 1:
        return ">3500"
    return "Unknown"


def build_admin_code_to_name(region_json_path: str) -> Dict[str, str]:
    with open(region_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # 可能存在过大的数据集//
    mapping: Dict[str, str] = {}
    for region in payload.get("regions", []):
        mapping[str(region.get("adm_cd", ""))] = str(region.get("adm_nm", ""))
        for district in region.get("districts", []):
            mapping[str(district.get("adm_cd", ""))] = str(district.get("adm_nm", ""))
            for town in district.get("towns", []):
                mapping[str(town.get("adm_cd", ""))] = str(town.get("adm_nm", ""))
    return mapping


def resolve_place_name(code: Any, code_to_name: Dict[str, str]) -> str:
    if pd.isna(code):
        return "Unknown"
    s = str(code)
    return code_to_name.get(s, f"Unknown(code={s})")


def collect_prompt_fields(row: pd.Series, code_to_name: Dict[str, str]) -> PromptFields:
    tpn_value = int(row.get("TPN", 6)) if pd.notna(row.get("TPN")) else 6
    dt_value = int(row.get("DT", 2)) if pd.notna(row.get("DT")) else 2

    return PromptFields(
        male=int(row.get("MALE", 0)),
        age=infer_age_group(row),
        income=infer_income_group(row),
        lic=int(row.get("LIC", 0)),
        purpose=TPN_MAP.get(tpn_value, "Other"),
        depart_time=DT_MAP.get(dt_value, "06:00-12:00"),
        trip_order=int(row.get("TP", 1)) if pd.notna(row.get("TP")) else 1,
        first_trip=int(row.get("PREMOI", 0)) if pd.notna(row.get("PREMOI")) else 0,
        stt=float(row.get("STT", 0.0)),
        std=float(row.get("STD", 0.0)),
        stc=float(row.get("STC", 0.0)),
        nsub=float(row.get("NSUB", 0.0)),
        nbus=float(row.get("NBUS", 0.0)),
        npark=float(row.get("NPARK", 0.0)),
        nbike=float(row.get("NBIKE", 0.0)),
        # code to name---json document!
        origin_name=resolve_place_name(row.get("ORIGIN"), code_to_name),
        destination_name=resolve_place_name(row.get("DESTIN"), code_to_name),
    )


def build_instruction(fields: PromptFields) -> str:
    gender = "Male" if fields.male == 1 else "Female"
    has_license = "Yes" if fields.lic == 1 else "No"
    is_first_trip = "Yes" if fields.first_trip == 1 else "No"
    
    # change to role play style
    # Availabel travel mode应该也被当作一个可选项 根据数据确定'available_mode'
    # prompt style: https://github.com/tsinghua-fib-lab/CoPB ---计划行为理论 
    #3 questions
    '''
    Q: What are the preferred activities of a person with <Profile>:[manager, 
    high income, female...] ? 
    A: <Preference>: [sport, shopping...]

    Q: What routines do a person with <Profile>:[manager, high income, 
    female...] typically have? 
    A: <Routine>: [Going to work at 9AM, Exercise at 5PM...]

    Q: Please evaluate the likelihood that a person with <Profile> , <Routine>, 
    <Behavior History> will have the <Intention> next?
    A: <Perceived Likelihood>: [work: unlikely, shopping: very likely,...]

    Reasoning:
    Q: Please select the next <Intention> 
    for a person with <Preference>, 
    <Routine>, <Perceived Likelihood>.
    A: Eat
    '''
    return (
        "You are a transportation planning expert. "
        "Given traveler attributes and trip context, predict the most likely travel mode.\n\n"
        "## Available travel modes\n"
        "- Auto\n- Riding\n- Subway\n- Bus\n- Subway&Bus\n- Taxi\n- Cycling\n- Walk\n\n"
        "## Traveler profile\n"
        f"- Gender: {gender}\n"
        f"- Age group: {fields.age}\n"
        f"- Income level: {fields.income}\n"
        f"- Driver's license: {has_license}\n\n"
        "## Trip context\n"
        f"- Origin: {fields.origin_name}\n"
        f"- Destination: {fields.destination_name}\n"
        f"- Trip purpose: {fields.purpose}\n"
        f"- Departure time window: {fields.depart_time}\n"
        f"- Trip sequence index: {fields.trip_order}\n"
        f"- Is this the first trip of the day: {is_first_trip}\n\n"
        "## Travel cost\n"
        f"- Time: {fields.stt:.2f} minutes\n"
        f"- Distance: {fields.std:.2f} km\n"
        f"- Cost: {fields.stc:.2f}\n\n"
        "## Destination infrastructure\n"
        f"- Number of subway stations: {fields.nsub:.0f}\n"
        f"- Number of bus stops: {fields.nbus:.0f}\n"
        f"- Number of parking lots: {fields.npark:.0f}\n"
        f"- Number of shared-bike stations: {fields.nbike:.0f}\n\n"
        "## Task\n"
        "Analyze the information, provide your reasoning, and choose the most likely travel mode.\n"
        "Output format:\n"
        "Reasoning: <your analysis>\n"
        "Choice: <one mode name from the list>"
    )
    # one mode name from the available mode name list


def normalize_answer(raw: Any) -> str:
    if pd.isna(raw):
        return "Unknown"
    value = str(raw).strip()
    if value in MODE_CODE_TO_NAME:
        return MODE_CODE_TO_NAME[value]
    if value in MODE_CODE_TO_NAME.values():
        return value
    return value


def polish_instruction_with_llm(
    instruction: str,
    answer: str,
    model_name: str,
    api_base: Optional[str],
    api_key_env: str,
) -> Tuple[str, str]:
    if OpenAI is None:
        raise ImportError("openai package is required when --use_llm is enabled.")

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key in env var: {api_key_env}")

    client = OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)

    system_prompt = (
        "You rewrite transportation-choice training examples into concise, natural English. "
        "Keep every factual detail unchanged. Return valid JSON with keys: instruction, answer."
    )
    user_prompt = json.dumps({"instruction": instruction, "answer": answer}, ensure_ascii=False)

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    parsed = json.loads(content)
    return parsed["instruction"], parsed["answer"]


def convert_rows(
    df: pd.DataFrame,
    code_to_name: Dict[str, str],
    use_llm: bool,
    llm_model: str,
    api_base: Optional[str],
    api_key_env: str,
    max_rows: int,
) -> List[Dict[str, Any]]:
    rows = df if max_rows <= 0 else df.head(max_rows)
    outputs: List[Dict[str, Any]] = []

    for _, row in rows.iterrows():
        fields = collect_prompt_fields(row, code_to_name)
        instruction = build_instruction(fields)
        answer = normalize_answer(row.get("MChoice"))

        if use_llm:
            instruction, answer = polish_instruction_with_llm(
                instruction=instruction,
                answer=answer,
                model_name=llm_model,
                api_base=api_base,
                api_key_env=api_key_env,
            )

        outputs.append(
            {
                "instruction": instruction,
                "answer": answer,
                "origin_code": str(row.get("ORIGIN", "")),
                "destination_code": str(row.get("DESTIN", "")),
            }
        )
    return outputs


def save_outputs(records: List[Dict[str, Any]], output_path: str, output_format: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return

    pd.DataFrame(records).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instruction-answer dataset for mode-choice modeling.")
    parser.add_argument("--input_csv", type=str, required=True, help="Raw tabular data path (CSV).")
    parser.add_argument(
        "--region_json",
        type=str,
        required=True,
        help="Region code-name JSON path. Includes destination/origin admin info.",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_format", choices=["csv", "json"], default="json")
    parser.add_argument("--max_rows", type=int, default=0, help="0 means all rows.")

    parser.add_argument("--use_llm", action="store_true", help="Use strong LLM to polish instruction-answer pairs.")
    parser.add_argument("--llm_model", type=str, default="deepseek-chat")
    parser.add_argument("--api_base", type=str, default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    code_to_name = build_admin_code_to_name(args.region_json)

    records = convert_rows(
        df=df,
        code_to_name=code_to_name,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        max_rows=args.max_rows,
    )

    save_outputs(records, args.output_path, args.output_format)
    print(f"Saved {len(records)} instruction-answer pairs to {args.output_path}")


if __name__ == "__main__":
    main()
