"""追加実験 v4：別モデルによる検証と、他モデルとの一致を振り分け信号にする。

予測（正誤）は常に対象モデルの元の分類を使い、信号だけを入れ替える。仮説と判定基準は docs/hypotheses_v4.md。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import review_cost_per_case
from .runner import collect_predictions, run_models
from .signals import _auroc_diff, _row, _sel_diff, _verdict_pos

STRONG = {"opus-5-5": "sonnet-5-5", "sonnet-5-5": "opus-5-5", "haiku-5-5": "opus-5-5"}
SHORT = {"opus-5-5": "Opus", "sonnet-5-5": "Sonnet", "haiku-5-5": "Haiku"}


def cross_verifier(verifier: dict, target_key: str) -> dict:
    return dict(verifier, key=f"{verifier['key']}__verify_of__{target_key}", task="verify")


def run_cross(cfg: dict, cases: pd.DataFrame, labels: list[str], raw_root: Path, caller_factory, log=print) -> dict:
    models = cfg["models"]
    log("1/2 元の分類（3モデル）")
    st = run_models(cfg, cases, labels, models, raw_root, caller_factory, log=log)
    if st["stopped"]:
        return st
    log("2/2 別モデルによる検証（6通り）")
    base = collect_predictions(raw_root, models, cases)
    for t in models:
        b = base[(base.model_key == t["key"]) & (base.status == "ok")][["case_id", "predicted_label"]]
        vc = cases.merge(b.rename(columns={"predicted_label": "proposed_label"}), on="case_id")
        for v in models:
            if v["key"] == t["key"]:
                continue
            st = run_models(cfg, vc, labels, [cross_verifier(v, t["key"])], raw_root, caller_factory, log=log)
            if st["stopped"]:
                return st
    return st


def build_frame(cfg: dict, cases: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    models = cfg["models"]
    base = {m["key"]: collect_predictions(raw_root, [m], cases).set_index("case_id") for m in models}
    rows = []
    for t in models:
        tk = t["key"]
        b = base[tk]
        ok = (b.status == "ok").to_numpy()
        others = [m["key"] for m in models if m["key"] != tk]
        agree = np.zeros(len(b))
        for o in others:
            ob = base[o].loc[b.index]
            agree += ((ob.status == "ok") & (ob.predicted_id == b.predicted_id)).to_numpy()
        xagree = np.where(ok, agree / len(others), 0.0)
        vcases = cases.merge(b[ok][["predicted_label"]].rename(columns={"predicted_label": "proposed_label"}),
                             left_on="case_id", right_index=True, how="left").dropna(subset=["proposed_label"])
        xver, xcost = {}, {}
        for v in models:
            if v["key"] == tk:
                continue
            vd = collect_predictions(raw_root, [cross_verifier(v, tk)], vcases).set_index("case_id").reindex(b.index)
            xver[v["key"]] = np.where(ok & (vd.status == "ok").to_numpy(),
                                      pd.to_numeric(vd.confidence, errors="coerce").fillna(0).to_numpy(), 0.0)
            xcost[v["key"]] = vd.cost_usd.fillna(0).to_numpy()
        self_s = b.score.to_numpy()
        strong = STRONG[tk]
        sig = {"self": (self_s, 0.0), "xagree": (xagree, 0.0)}
        for vk in xver:
            sig[f"xver_{vk}"] = (xver[vk], xcost[vk])
        sig["combx"] = (np.where(ok, (self_s + xver[strong] + xagree) / 3, 0.0), xcost[strong])
        for name, (score, add) in sig.items():
            f = b.reset_index().copy()
            f["model_key"], f["base_model"], f["signal"] = f"{tk}:{name}", tk, name
            f["score"] = score
            f["confidence"] = score
            f["cost_usd"] = b.cost_usd.fillna(0).to_numpy() + add
            rows.append(f)
    return pd.concat(rows, ignore_index=True)


def evaluate(cfg: dict, frame: pd.DataFrame, tables: dict) -> pd.DataFrame:
    n_boot = cfg["analysis"]["bootstrap_n"]
    models = list(dict.fromkeys(frame.base_model))
    out = []

    def fmt(res):
        return "、".join(f"{k}: {d:+.3f}（{lo:+.3f}〜{hi:+.3f}）" for k, (d, lo, hi) in res.items())

    r1 = {SHORT[t]: _auroc_diff(cfg, frame, f"{t}:xver_opus-5-5", f"{t}:self", n_boot) for t in ("sonnet-5-5", "haiku-5-5")}
    out.append(_row("H1", "AUROC：Opus による検証 > 自己申告（Sonnet・Haiku）", _verdict_pos(r1.values()), fmt(r1), r1))

    r2 = {SHORT[t]: _auroc_diff(cfg, frame, f"{t}:xagree", f"{t}:self", n_boot) for t in models}
    out.append(_row("H2", "AUROC：他モデル一致 > 自己申告（全モデル）", _verdict_pos(r2.values()), fmt(r2), r2))

    d3 = {SHORT[t]: _sel_diff(tables["selection_bootstrap_diffs"], 1.0, 0.90, f"{t}:combx", f"{t}:self") for t in models}
    if all(hi < 0 for _, _, hi in d3.values()):
        v3 = "支持"
    elif any(lo > 0 for _, lo, _ in d3.values()):
        v3 = "不支持"
    else:
        v3 = "判定保留"
    weak = [m for m, (_, _, hi) in d3.items() if hi < 0]
    s3 = "、".join(f"{m}: {d*100:+.1f}pt（{lo*100:+.1f}〜{hi*100:+.1f}）" for m, (d, lo, hi) in d3.items())
    s3 += f"。95%区間が0未満のモデル：{', '.join(weak) if weak else 'なし'}"
    out.append(_row("H3", "目標90%の必要レビュー率：組み合わせ < 自己申告（全モデル）", v3, s3, {**d3, "weak_form_models": weak}))

    r4 = {"Haiku の予測：Opus 検証−Sonnet 検証": _auroc_diff(cfg, frame, "haiku-5-5:xver_opus-5-5", "haiku-5-5:xver_sonnet-5-5", n_boot),
          "Opus の予測：Sonnet 検証−Haiku 検証": _auroc_diff(cfg, frame, "opus-5-5:xver_sonnet-5-5", "opus-5-5:xver_haiku-5-5", n_boot)}
    out.append(_row("H4", "検証モデルが強いほど AUROC が高い", _verdict_pos(r4.values()), fmt(r4), r4))

    ready, met = tables["ready_qualification"], tables["metrics"]
    unit = review_cost_per_case(cfg)
    c5 = {}
    for t in models:
        def get(sig):
            r = ready[(ready.model_key == f"{t}:{sig}") & (ready.a_h == 1.0) & (ready.margin == 0.05) & (ready.target == 0.90)].iloc[0]
            api = met[(met.model_key == f"{t}:{sig}") & (met.split == "test")].api_cost_jpy_per_case.iloc[0]
            return r.test_review_rate, api
        rr_s, api_s = get("self")
        rr_c, api_c = get("combx")
        _, lo, hi = d3[SHORT[t]]
        c5[SHORT[t]] = {"total_self": api_s + rr_s * unit, "total_comb": api_c + rr_c * unit,
                        "diff": (api_c - api_s) + (rr_c - rr_s) * unit, "extra_api": api_c - api_s,
                        "diff_lo": (api_c - api_s) + lo * unit, "diff_hi": (api_c - api_s) + hi * unit}
    v5 = "支持" if all(x["diff"] < 0 for x in c5.values()) else "不支持"
    s5 = "、".join(f"{m}: {x['total_self']:.1f}円→{x['total_comb']:.1f}円（追加API {x['extra_api']:.2f}円、"
                   f"差の目安 {x['diff_lo']:+.1f}〜{x['diff_hi']:+.1f}円）" for m, x in c5.items())
    out.append(_row("H5", "別モデルの情報を加えても総コストが下がる（全モデル、点推定）", v5, s5, c5))
    return pd.DataFrame(out)


def simulate_frame(cfg: dict, cases: pd.DataFrame) -> pd.DataFrame:
    """API を使わない動作確認用。モデル名は実モデルのキーを借りるが、値はすべて架空（実測ではない）。"""
    rng = np.random.default_rng(cfg["seed"] + 11)
    keys = [m["key"] for m in cfg["models"]]
    n = len(cases)
    diff = rng.normal(0, 1.5, n)
    acc = {"opus-5-5": 0.88, "sonnet-5-5": 0.85, "haiku-5-5": 0.78}
    correct = {k: (rng.random(n) < 1 / (1 + np.exp(-(np.log(acc[k] / (1 - acc[k])) - diff)))).astype(int) for k in keys}
    rows = []
    for t in keys:
        c = correct[t]
        self_s = np.clip(0.75 + 0.15 * c + rng.normal(0, 0.12, n), 0, 1).round(2)
        others = [o for o in keys if o != t]
        xagree = np.mean([(correct[o] == 1) & (c == 1) | (rng.random(n) < 0.1) for o in others], axis=0)
        xv = {o: np.clip(0.55 + 0.35 * c + rng.normal(0, 0.15, n), 0, 1).round(2) for o in others}
        sig = {"self": self_s, "xagree": xagree, **{f"xver_{o}": xv[o] for o in others}}
        sig["combx"] = (self_s + xv[STRONG[t]] + xagree) / 3
        for name, s in sig.items():
            f = cases.copy()
            f["model_key"], f["base_model"], f["signal"] = f"{t}:{name}", t, name
            f["model_id"] = f"simulation:{t}"
            f["status"], f["correct"], f["score"], f["confidence"] = "ok", c, s, s
            f["predicted_id"] = np.where(c == 1, f.label_id, (f.label_id + 1) % 77)
            f["cost_usd"] = 0.001 if name in ("self", "xagree") else 0.002
            for col in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
                f[col] = np.nan
            rows.append(f)
    return pd.concat(rows, ignore_index=True)
