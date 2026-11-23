"""実行管理：逐次保存・キャッシュ・費用上限・シミュレーション。"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .prompt import PROMPT_VERSION

FAIL_STATUSES = ("invalid_output", "invalid_label", "api_error", "timeout", "refusal", "max_tokens")
EST_INPUT_TOKENS = 4000  # 費用上限の予約計算に使う入力トークンの上限見込み（保守的）


def cache_key(m: dict, case: dict) -> str:
    payload = {
        "model_id": m["model_id"], "provider": m["provider"], "case_id": case["case_id"], "text": case["text"],
        "prompt_version": PROMPT_VERSION, "effort": m.get("effort"), "max_tokens": m["max_tokens"],
    }
    # v3 で追加した呼び出しの種類。既定値（分類・1回目）のときは含めず、v1・v2 のキャッシュキーを変えない。
    if m.get("task", "classify") != "classify":
        payload["task"] = m["task"]
        payload["proposed_label"] = case.get("proposed_label")
    if m.get("sample", 0):
        payload["sample"] = int(m["sample"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def raw_path(raw_root: Path, m: dict, case: dict) -> Path:
    return raw_root / m["key"] / f"{cache_key(m, case)}.json"


def spent_usd(raw_root: Path) -> float:
    """保存済みの全応答の費用合計（実験をまたいだ累積）。"""
    total = 0.0
    for p in raw_root.glob("*/*.json"):
        try:
            total += float(json.loads(p.read_text()).get("cost_usd") or 0.0)
        except (json.JSONDecodeError, OSError):
            pass
    return total


def _write_atomic(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


class BudgetExceeded(RuntimeError):
    pass


def run_models(cfg: dict, cases: pd.DataFrame, labels: list[str], models: list[dict], raw_root: Path,
               caller_factory, retry_errors: bool = False, log=print) -> dict:
    """未実行の (モデル, ケース) だけを実行し、1件ごとに保存する。"""
    budget = float(cfg["run"]["budget_usd"])
    lock = threading.Lock()
    state = {"spent": spent_usd(raw_root), "reserved": 0.0, "stopped": False, "done": 0, "cached": 0}
    log(f"既存の累積費用: ${state['spent']:.4f} / 上限 ${budget:.2f}")
    callers: dict[str, object] = {}
    jobs = []
    for m in models:
        for case in cases.to_dict("records"):
            p = raw_path(raw_root, m, case)
            if p.exists():
                rec = json.loads(p.read_text())
                if not (retry_errors and rec.get("status") in ("api_error", "timeout")):
                    state["cached"] += 1
                    continue
            jobs.append((m, case, p))
    log(f"キャッシュ利用: {state['cached']} 件 / 新規実行: {len(jobs)} 件")
    if not jobs:
        return state
    for m in {j[0]["key"]: j[0] for j in jobs}.values():
        callers[m["key"]] = caller_factory(m)

    def reserve(m: dict) -> float:
        return EST_INPUT_TOKENS * m["price_in"] / 1e6 * 1.25 + m["max_tokens"] * m["price_out"] / 1e6

    def work(m, case, p):
        r = reserve(m)
        with lock:
            if state["stopped"] or state["spent"] + state["reserved"] + r > budget:
                state["stopped"] = True
                return None
            state["reserved"] += r
        try:
            rec = callers[m["key"]](m, case, labels)
            rec["cache_key"] = cache_key(m, case)
            rec["prompt_version"] = PROMPT_VERSION
            _write_atomic(p, rec)
            return rec
        finally:
            with lock:
                state["reserved"] -= r

    with ThreadPoolExecutor(max_workers=int(cfg["run"]["concurrency"])) as ex:
        futs = [ex.submit(work, *j) for j in jobs]
        for f in as_completed(futs):
            rec = f.result()
            if rec is None:
                continue
            with lock:
                state["spent"] += float(rec.get("cost_usd") or 0.0)
                state["done"] += 1
                if state["done"] % 25 == 0:
                    log(f"  {state['done']}/{len(jobs)} 件完了、累積 ${state['spent']:.4f}")
    if state["stopped"]:
        log(f"費用上限 ${budget:.2f} に達する見込みのため停止しました（累積 ${state['spent']:.4f}）。")
    log(f"新規 {state['done']} 件完了。累積費用 ${state['spent']:.4f}")
    return state


def collect_predictions(raw_root: Path, models: list[dict], cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in models:
        for case in cases.to_dict("records"):
            p = raw_path(raw_root, m, case)
            if p.exists():
                rec = json.loads(p.read_text())
            else:
                rec = {"model_key": m["key"], "model_id": m["model_id"], "case_id": case["case_id"],
                       "status": "not_run", "predicted_id": -1, "confidence": None, "cost_usd": 0.0}
            usage = rec.get("usage") or {}
            rows.append({
                "model_key": m["key"], "model_id": m["model_id"], "response_model": rec.get("response_model"),
                "case_id": case["case_id"], "split": case["split"], "text": case["text"],
                "label_id": int(case["label_id"]), "label": labels_name(case),
                "status": rec["status"], "predicted_id": int(rec.get("predicted_id", -1)),
                "predicted_label": rec.get("predicted_label"), "confidence": rec.get("confidence"),
                "confidence_raw": rec.get("confidence_raw"), "label_match": rec.get("label_match"),
                "parse_note": rec.get("parse_note"), "stop_reason": rec.get("stop_reason"),
                "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                "cost_usd": rec.get("cost_usd", 0.0), "latency_s": rec.get("latency_s"),
                "timestamp": rec.get("timestamp"), "error": rec.get("error"),
            })
    df = pd.DataFrame(rows)
    return finalize_predictions(df)


def labels_name(case: dict) -> str | None:
    return case.get("label")


def finalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """正誤とレビュー振り分け用スコアを付ける。失敗は誤答扱い・スコア0（最優先でレビュー）。"""
    df = df.copy()
    ok = df["status"] == "ok"
    df["correct"] = (ok & (df["predicted_id"] == df["label_id"])).astype(int)
    df["score"] = np.where(ok, pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0), 0.0)
    return df


# ---------------------------------------------------------------- シミュレーション
def simulate_predictions(cfg: dict, cases: pd.DataFrame, n_labels: int = 77) -> pd.DataFrame:
    """APIを使わずに仮想モデルの予測を作る。結果は実測ではない（パイプライン検証用）。

    難易度はケース共通の潜在変数とし、モデル間に相関を持たせる。
    calibration=good は確信度が正答確率に沿う。poor は確信度が高い値に偏り、正誤との関係が弱い。
    """
    rng = np.random.default_rng(cfg["seed"] + 7)
    n = len(cases)
    difficulty = rng.normal(0, 1.5, n)
    rows = []
    for sm in cfg["simulation"]["models"]:
        target = sm["accuracy"]
        lo, hi = -10.0, 10.0  # 平均正答率が target になる切片を二分探索
        for _ in range(60):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(mid - difficulty)))).mean() < target:
                lo = mid
            else:
                hi = mid
        p = 1 / (1 + np.exp(-(mid - difficulty)))
        correct = rng.random(n) < p
        if sm["calibration"] == "good":
            conf = np.clip(p + rng.normal(0, 0.08, n), 0.05, 0.99)
        else:
            conf = np.clip(rng.choice([0.9, 0.95, 0.99], size=n, p=[0.2, 0.5, 0.3]) - 0.05 * (~correct) * rng.random(n), 0, 1)
        status = np.where(rng.random(n) < 0.01, "invalid_output", "ok")
        for i, case in enumerate(cases.to_dict("records")):
            pred = case["label_id"] if correct[i] else int((case["label_id"] + rng.integers(1, n_labels)) % n_labels)
            st = status[i]
            rows.append({
                "model_key": sm["key"], "model_id": f"simulation:{sm['key']}", "response_model": None,
                "case_id": case["case_id"], "split": case["split"], "text": case["text"],
                "label_id": int(case["label_id"]), "label": case.get("label"), "status": st,
                "predicted_id": pred if st == "ok" else -1, "predicted_label": None,
                "confidence": round(float(conf[i]), 2) if st == "ok" else None, "confidence_raw": None,
                "label_match": None, "parse_note": "SIMULATED", "stop_reason": None,
                "input_tokens": None, "output_tokens": None, "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None, "cost_usd": sm.get("cost_usd_per_case", 0.005),
                "latency_s": None, "timestamp": datetime.now(timezone.utc).isoformat(), "error": None,
            })
    return finalize_predictions(pd.DataFrame(rows))
