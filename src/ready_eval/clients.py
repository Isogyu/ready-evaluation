"""モデル呼び出し。1件ごとに記録用の dict を返す。

- 失敗（APIエラー・タイムアウト・拒否・出力上限・不正出力）は例外にせず status に記録する。
- Temperature は現行の Claude API では指定できないため送らない。代わりに effort を固定する。
- 拒否時のフォールバック（別モデルでの再実行）は使わない。比較対象のモデルが入れ替わるため。
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from .prompt import (VERIFY_SCHEMA, output_schema, parse_output, parse_verify, system_prompt, user_prompt,
                     verify_system_prompt, verify_user_prompt)

ANTHROPIC_BASE_URL = "https://api.anthropic.com"  # Claude Code の環境変数 ANTHROPIC_BASE_URL を使わないよう明示


def cost_usd(m: dict, usage: dict) -> float:
    p_in, p_out = m["price_in"] / 1e6, m["price_out"] / 1e6
    return (
        usage.get("input_tokens", 0) * p_in
        + usage.get("cache_creation_input_tokens", 0) * p_in * 1.25
        + usage.get("cache_read_input_tokens", 0) * p_in * 0.10
        + usage.get("output_tokens", 0) * p_out
    )


def _base_record(m: dict, case: dict) -> dict:
    return {
        "model_key": m["key"], "model_id": m["model_id"], "provider": m["provider"],
        "case_id": case["case_id"], "split": case["split"], "label_id": int(case["label_id"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _finish(rec: dict, text: str | None, labels: list[str], status_override: str | None = None) -> dict:
    p = parse_output(text, labels)
    rec.update({
        "raw_text": text, "status": status_override or p.status, "predicted_id": p.predicted_id,
        "predicted_label": p.predicted_label, "confidence": p.confidence,
        "confidence_raw": p.confidence_raw, "label_match": p.label_match, "parse_note": p.note,
    })
    if status_override:  # 出力上限・拒否などは解析できても失敗として扱う
        rec["predicted_id"] = -1
    return rec


class AnthropicCaller:
    def __init__(self, timeout_s: float, max_retries: int):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), base_url=ANTHROPIC_BASE_URL,
            timeout=timeout_s, max_retries=max_retries,
        )

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list()]

    def __call__(self, m: dict, case: dict, labels: list[str]) -> dict:
        a = self.anthropic
        rec = _base_record(m, case)
        task = m.get("task", "classify")
        rec["settings"] = {"effort": m.get("effort"), "max_tokens": m["max_tokens"], "thinking": "model default",
                           "task": task, "sample": m.get("sample", 0)}
        if task == "verify":
            sys_text = verify_system_prompt(labels)
            user = verify_user_prompt(case["text"], case["proposed_label"])
            schema = VERIFY_SCHEMA
        else:
            sys_text, user, schema = system_prompt(labels), user_prompt(case["text"]), output_schema(labels)
        t0 = time.monotonic()
        try:
            resp = self.client.messages.create(
                model=m["model_id"],
                max_tokens=m["max_tokens"],
                system=[{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": m.get("effort", "low"),
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except a.APITimeoutError as e:
            rec.update(latency_s=time.monotonic() - t0, error=str(e)[:500], cost_usd=0.0)
            return _finish(rec, None, labels, "timeout")
        except a.APIStatusError as e:
            rec.update(latency_s=time.monotonic() - t0, error=f"{e.status_code}: {str(e)[:500]}", cost_usd=0.0)
            return _finish(rec, None, labels, "api_error")
        except a.APIConnectionError as e:
            rec.update(latency_s=time.monotonic() - t0, error=str(e)[:500], cost_usd=0.0)
            return _finish(rec, None, labels, "api_error")
        usage = {k: getattr(resp.usage, k, 0) or 0 for k in
                 ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
        text = next((b.text for b in resp.content if b.type == "text"), None)
        rec.update(
            latency_s=time.monotonic() - t0, response_model=resp.model, stop_reason=resp.stop_reason,
            request_id=getattr(resp, "_request_id", None), usage=usage, cost_usd=cost_usd(m, usage),
        )
        override = {"refusal": "refusal", "max_tokens": "max_tokens"}.get(resp.stop_reason)
        if task == "verify":
            st, prob, note = parse_verify(text)
            rec.update({"raw_text": text, "status": override or st, "predicted_id": -1,
                        "predicted_label": case["proposed_label"], "confidence": prob if not override else None,
                        "confidence_raw": prob, "label_match": None, "parse_note": note})
            return rec
        return _finish(rec, text, labels, override)


class OpenAICaller:
    """GPT系モデル用（OPENAI_API_KEY がある場合のみ）。このプロジェクトでは実行検証していない。"""

    def __init__(self, timeout_s: float, max_retries: int):
        import openai
        self.openai = openai
        self.client = openai.OpenAI(timeout=timeout_s, max_retries=max_retries)

    def __call__(self, m: dict, case: dict, labels: list[str]) -> dict:
        o = self.openai
        rec = _base_record(m, case)
        rec["settings"] = {"max_completion_tokens": m["max_tokens"]}
        t0 = time.monotonic()
        try:
            resp = self.client.chat.completions.create(
                model=m["model_id"],
                max_completion_tokens=m["max_tokens"],
                messages=[{"role": "system", "content": system_prompt(labels)},
                          {"role": "user", "content": user_prompt(case["text"])}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "classification", "schema": output_schema(labels), "strict": True}},
            )
        except o.APITimeoutError as e:
            rec.update(latency_s=time.monotonic() - t0, error=str(e)[:500], cost_usd=0.0)
            return _finish(rec, None, labels, "timeout")
        except o.APIError as e:
            rec.update(latency_s=time.monotonic() - t0, error=str(e)[:500], cost_usd=0.0)
            return _finish(rec, None, labels, "api_error")
        choice = resp.choices[0]
        usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
        rec.update(latency_s=time.monotonic() - t0, response_model=resp.model, stop_reason=choice.finish_reason,
                   usage=usage, cost_usd=cost_usd(m, usage))
        override = {"length": "max_tokens", "content_filter": "refusal"}.get(choice.finish_reason)
        if getattr(choice.message, "refusal", None):
            override = "refusal"
        return _finish(rec, choice.message.content, labels, override)
