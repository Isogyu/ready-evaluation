"""全モデル共通のプロンプトと、出力の解析・ラベルID変換。"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass

PROMPT_VERSION = "v1"


def system_prompt(labels: list[str]) -> str:
    label_list = "\n".join(labels)
    return (
        "You classify customer messages sent to a bank's support channel into exactly one of 77 intent "
        "categories.\n\n"
        "Categories (use the name exactly as written):\n"
        f"{label_list}\n\n"
        "Return a JSON object with two fields:\n"
        '- "predicted_label": the single best category name from the list above.\n'
        '- "confidence": a number from 0 to 1 giving the probability that your predicted_label is correct.\n'
        "Return only the JSON object."
    )


def user_prompt(text: str) -> str:
    return f"Customer message:\n{text}"


def output_schema(labels: list[str]) -> dict:
    # 数値の範囲制約（minimum/maximum）は構造化出力で使えないため、範囲外は解析時に記録する。
    return {
        "type": "object",
        "properties": {
            "predicted_label": {"type": "string", "enum": list(labels)},
            "confidence": {"type": "number"},
        },
        "required": ["predicted_label", "confidence"],
        "additionalProperties": False,
    }


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


@dataclass
class Parsed:
    status: str               # ok / invalid_output / invalid_label
    predicted_id: int         # 失敗時は -1
    predicted_label: str | None
    confidence: float | None  # 0〜1 に収めた値。失敗時は None
    confidence_raw: float | None
    label_match: str          # exact / normalized / fuzzy / none
    note: str = ""


def to_label_id(raw_label: str, labels: list[str], fuzzy_cutoff: float = 0.92) -> tuple[int, str]:
    """ラベル名を ID に変換する。完全一致 → 正規化一致 → 高い類似度の近似一致 の順。"""
    if raw_label in labels:
        return labels.index(raw_label), "exact"
    normed = {_norm(l): i for i, l in enumerate(labels)}
    key = _norm(raw_label)
    if key in normed:
        return normed[key], "normalized"
    close = difflib.get_close_matches(key, list(normed), n=1, cutoff=fuzzy_cutoff)
    if close:
        return normed[close[0]], "fuzzy"
    return -1, "none"


def parse_output(text: str | None, labels: list[str]) -> Parsed:
    if not text:
        return Parsed("invalid_output", -1, None, None, None, "none", "empty")
    try:
        m = re.search(r"\{.*\}", text, flags=re.S)
        obj = json.loads(m.group(0) if m else text)
        raw_label = obj["predicted_label"]
        conf_raw = obj["confidence"]
    except Exception as e:  # noqa: BLE001
        return Parsed("invalid_output", -1, None, None, None, "none", f"parse:{type(e).__name__}")
    try:
        conf_raw = float(conf_raw)
    except (TypeError, ValueError):
        return Parsed("invalid_output", -1, str(raw_label), None, None, "none", "confidence_not_number")
    note = ""
    conf = conf_raw
    if conf_raw > 1.0 and conf_raw <= 100.0:
        # 百分率で答えた場合（例：95）。変換したことを記録する。
        conf, note = conf_raw / 100.0, "confidence_percent_rescaled"
    if not (0.0 <= conf <= 1.0):
        conf, note = min(max(conf, 0.0), 1.0), "confidence_clipped"
    pid, how = to_label_id(str(raw_label), labels)
    if pid < 0:
        return Parsed("invalid_label", -1, str(raw_label), conf, conf_raw, how, note)
    return Parsed("ok", pid, labels[pid], conf, conf_raw, how, note)


# ---------------------------------------------------------------- 自己検証（v3）
def verify_system_prompt(labels: list[str]) -> str:
    label_list = "\n".join(labels)
    return (
        "You review intent labels assigned to customer messages sent to a bank's support channel. "
        "There are 77 intent categories:\n"
        f"{label_list}\n\n"
        "You will see a customer message and a proposed category. Estimate the probability that the proposed "
        "category is the single correct category for the message.\n"
        'Return a JSON object with one field, "probability_correct": a number from 0 to 1. Return only the JSON object.'
    )


def verify_user_prompt(text: str, proposed_label: str) -> str:
    return f"Customer message:\n{text}\n\nProposed category: {proposed_label}"


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"probability_correct": {"type": "number"}},
    "required": ["probability_correct"],
    "additionalProperties": False,
}


def parse_verify(text: str | None) -> tuple[str, float | None, str]:
    """(status, probability, note)。範囲外は記録して丸める。"""
    if not text:
        return "invalid_output", None, "empty"
    try:
        m = re.search(r"\{.*\}", text, flags=re.S)
        v = float(json.loads(m.group(0) if m else text)["probability_correct"])
    except Exception as e:  # noqa: BLE001
        return "invalid_output", None, f"parse:{type(e).__name__}"
    note = ""
    if 1.0 < v <= 100.0:
        v, note = v / 100.0, "percent_rescaled"
    if not (0.0 <= v <= 1.0):
        v, note = min(max(v, 0.0), 1.0), "clipped"
    return "ok", v, note
