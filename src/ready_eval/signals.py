"""追加実験 v3：振り分け信号の比較（自己申告・一致度・自己検証・組み合わせ）と、下限法によるしきい値選定。

予測（正誤）は常に元の1回目の分類を使い、信号だけを入れ替える。仮説と判定基準は docs/hypotheses_v3.md。
分析では「モデル:信号」を1つの擬似モデルとして扱い、既存の analysis.run_all をそのまま使う。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .analysis import _arrays, review_cost_per_case
from .runner import collect_predictions, run_models

SIGNALS = ["self", "cons", "ver", "comb"]
SIGNAL_NAMES = {"self": "自己申告", "cons": "一致度", "ver": "自己検証", "comb": "組み合わせ"}


def sample_models(m: dict, k: int) -> list[dict]:
    return [dict(m, key=f"{m['key']}__s{i}", sample=i) for i in range(1, k + 1)]


def verify_model(m: dict) -> dict:
    return dict(m, key=f"{m['key']}__verify", task="verify")


def run_signals(cfg: dict, cases: pd.DataFrame, labels: list[str], raw_root: Path, caller_factory, log=print) -> dict:
    """元の分類 → 追加分類（k回）→ 自己検証 の順に実行する。すべてキャッシュと費用上限が効く。"""
    models = cfg["models"]
    k = cfg["signals"]["n_extra_samples"]
    log("1/3 元の分類")
    st = run_models(cfg, cases, labels, models, raw_root, caller_factory, log=log)
    if st["stopped"]:
        return st
    log(f"2/3 追加分類（{k}回）")
    extra = [sm for m in models for sm in sample_models(m, k)]
    st = run_models(cfg, cases, labels, extra, raw_root, caller_factory, log=log)
    if st["stopped"]:
        return st
    log("3/3 自己検証")
    base = collect_predictions(raw_root, models, cases)
    for m in models:
        b = base[(base.model_key == m["key"]) & (base.status == "ok")][["case_id", "predicted_label"]]
        vc = cases.merge(b.rename(columns={"predicted_label": "proposed_label"}), on="case_id")
        st = run_models(cfg, vc, labels, [verify_model(m)], raw_root, caller_factory, log=log)
        if st["stopped"]:
            return st
    return st


def build_signal_frame(cfg: dict, cases: pd.DataFrame, raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """擬似モデル「モデル:信号」ごとの予測表と、追加分類のばらつきの要約を返す。"""
    k = cfg["signals"]["n_extra_samples"]
    rows, var_rows = [], []
    for m in cfg["models"]:
        base = collect_predictions(raw_root, [m], cases).set_index("case_id")
        smp = [collect_predictions(raw_root, [sm], cases).set_index("case_id") for sm in sample_models(m, k)]
        ok = base.status == "ok"
        agree = np.zeros(len(base))
        extra_cost = np.zeros(len(base))
        for s in smp:
            s = s.loc[base.index]
            agree += ((s.status == "ok") & (s.predicted_id == base.predicted_id) & ok).to_numpy()
            extra_cost += s.cost_usd.fillna(0).to_numpy()
            var_rows.append({"model_key": m["key"], "sample": s["model_key"].iloc[0],
                             "share_same_label": float(((s.predicted_id == base.predicted_id) & ok).mean()),
                             "share_ok": float((s.status == "ok").mean())})
        cons = agree / k
        vcases = cases.merge(base[ok][["predicted_label"]].rename(columns={"predicted_label": "proposed_label"}),
                             left_on="case_id", right_index=True, how="left")
        ver_df = collect_predictions(raw_root, [verify_model(m)], vcases.dropna(subset=["proposed_label"]))
        ver = ver_df.set_index("case_id").reindex(base.index)
        ver_score = np.where((ver.status == "ok").to_numpy(),
                             pd.to_numeric(ver.confidence, errors="coerce").fillna(0).to_numpy(), 0.0)
        ver_cost = ver.cost_usd.fillna(0).to_numpy()
        self_score = base.score.to_numpy()
        sig = {
            "self": (self_score, 0.0),
            "cons": (np.where(ok, cons, 0.0), extra_cost),
            "ver": (np.where(ok, ver_score, 0.0), ver_cost),
            "comb": (np.where(ok, (self_score + cons + ver_score) / 3, 0.0), extra_cost + ver_cost),
        }
        missing = int((ver.status == "not_run").sum() + sum((s.status == "not_run").sum() for s in smp))
        for name, (score, add_cost) in sig.items():
            f = base.reset_index().copy()
            f["model_key"] = f"{m['key']}:{name}"
            f["base_model"] = m["key"]
            f["signal"] = name
            f["score"] = score
            f["confidence"] = score
            f["cost_usd"] = base.cost_usd.fillna(0).to_numpy() + add_cost
            f["n_missing_extra"] = missing
            rows.append(f)
    frame = pd.concat(rows, ignore_index=True)
    return frame, pd.DataFrame(var_rows)


# ------------------------------------------------------------------ 選定方法の比較（H5）
def method_bootstrap(cfg: dict, frame: pd.DataFrame, Y: float = 0.90, a_h: float = 1.0, n_boot: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """δ=0.05 法と下限法を、開発・最終評価の両方を再標本化して比べる（同じ再標本を両方法に使う）。"""
    level = cfg["analysis"]["confidence_level"]
    keys = list(frame.model_key.unique())
    dev = {k: _arrays(frame, k, "dev")[:2] for k in keys}
    tst = {k: _arrays(frame, k, "test")[:2] for k in keys}
    n_dev, n_test = len(dev[keys[0]][0]), len(tst[keys[0]][0])
    rng = np.random.default_rng(cfg["seed"] + 505)
    rec = []
    for b in range(n_boot):
        idev, itst = rng.integers(0, n_dev, n_dev), rng.integers(0, n_test, n_test)
        for k in keys:
            cd, sd = dev[k][0][idev], dev[k][1][idev]
            ct, st = tst[k][0][itst], tst[k][1][itst]
            for method in ("delta", "lcb"):
                sel = M.select_threshold(cd, sd, Y, a_h, 0.05) if method == "delta" else M.select_threshold_lcb(cd, sd, Y, a_h, level)
                q = M.qualify(ct, st, sel["tau"], Y, a_h, level)
                rec.append({"b": b, "model_key": k, "method": method, "review_rate": q["test_review_rate"],
                            "qualified": q["qualified"]})
    boot = pd.DataFrame(rec)
    summ = boot.groupby(["model_key", "method"]).agg(
        review_rate_mean=("review_rate", "mean"),
        review_rate_lo=("review_rate", lambda x: np.nanpercentile(x, 2.5)),
        review_rate_hi=("review_rate", lambda x: np.nanpercentile(x, 97.5)),
        share_qualified=("qualified", "mean")).reset_index()
    w = boot.pivot_table(index=["b", "model_key"], columns="method", values="review_rate").reset_index()
    w["diff"] = w["lcb"] - w["delta"]
    diff = w.groupby("model_key")["diff"].agg(
        diff_mean="mean", diff_lo=lambda x: np.nanpercentile(x, 2.5), diff_hi=lambda x: np.nanpercentile(x, 97.5)).reset_index()
    return summ, diff


def point_methods(cfg: dict, frame: pd.DataFrame, Y: float = 0.90, a_h: float = 1.0) -> pd.DataFrame:
    """実際のデータ（再標本化なし）での δ法・下限法の選定結果。"""
    level = cfg["analysis"]["confidence_level"]
    rows = []
    for k in frame.model_key.unique():
        cd, sd, _ = _arrays(frame, k, "dev")
        ct, st, _ = _arrays(frame, k, "test")
        for method in ("delta", "lcb"):
            sel = M.select_threshold(cd, sd, Y, a_h, 0.05) if method == "delta" else M.select_threshold_lcb(cd, sd, Y, a_h, level)
            q = M.qualify(ct, st, sel["tau"], Y, a_h, level)
            rows.append({"model_key": k, "method": method, "target": Y, "a_h": a_h, **sel, **q,
                         "oracle_review_rate": M.select_threshold(ct, st, Y, a_h, 0.0)["dev_review_rate"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 仮説判定
def _row(hid, hyp, verdict, summary, details):
    return {"id": hid, "hypothesis": hyp, "verdict": verdict, "summary": summary,
            "details": json.dumps(details, ensure_ascii=False, default=float)}


def _auroc_diff(cfg, frame, a_key, b_key, n_boot):
    ca, sa, _ = _arrays(frame, a_key, "test")
    cb, sb, _ = _arrays(frame, b_key, "test")
    assert np.array_equal(ca, cb)  # 予測が同じ（正誤が同じ）ことの確認
    d = M.auroc_error_detection(ca, sa) - M.auroc_error_detection(cb, sb)
    rng = np.random.default_rng(cfg["seed"] + 606)
    vals = []
    n = len(ca)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = M.auroc_error_detection(ca[i], sa[i]) - M.auroc_error_detection(cb[i], sb[i])
        if not np.isnan(v):
            vals.append(v)
    return d, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _verdict_pos(results):
    if all(lo > 0 for _, lo, _ in results):
        return "支持"
    if any(hi < 0 for _, _, hi in results):
        return "不支持"
    return "判定保留"


def _sel_diff(diffs, a_h, Y, k1, k2):
    sel = diffs[(diffs.a_h == a_h) & (diffs.target == Y)]
    r = sel[(sel.model_1 == k1) & (sel.model_2 == k2)]
    if len(r):
        r = r.iloc[0]
        return r.review_rate_diff_mean, r.diff_lo, r.diff_hi
    r = sel[(sel.model_1 == k2) & (sel.model_2 == k1)].iloc[0]
    return -r.review_rate_diff_mean, -r.diff_hi, -r.diff_lo


def evaluate(cfg: dict, frame: pd.DataFrame, tables: dict, msumm: pd.DataFrame, mdiff: pd.DataFrame) -> pd.DataFrame:
    n_boot = cfg["analysis"]["bootstrap_n"]
    models = list(dict.fromkeys(frame.base_model))
    out = []

    def auroc_h(hid, hyp, sig):
        res = {m: _auroc_diff(cfg, frame, f"{m}:{sig}", f"{m}:self", n_boot) for m in models}
        s = "、".join(f"{m}: {d:+.3f}（{lo:+.3f}〜{hi:+.3f}）" for m, (d, lo, hi) in res.items())
        return res, s

    res, s = auroc_h("H1", "", "comb")
    out.append(_row("H1", "AUROC：組み合わせ > 自己申告（全モデル）", _verdict_pos(res.values()), s, res))

    d2 = {m: _sel_diff(tables["selection_bootstrap_diffs"], 1.0, 0.90, f"{m}:comb", f"{m}:self") for m in models}
    if all(hi < 0 for _, _, hi in d2.values()):
        v2 = "支持"
    elif any(lo > 0 for _, lo, _ in d2.values()):
        v2 = "不支持"
    else:
        v2 = "判定保留"
    weak = [m for m, (_, _, hi) in d2.items() if hi < 0]
    s2 = "、".join(f"{m}: {d*100:+.1f}pt（{lo*100:+.1f}〜{hi*100:+.1f}）" for m, (d, lo, hi) in d2.items())
    s2 += f"。95%区間が0未満のモデル：{', '.join(weak) if weak else 'なし'}"
    out.append(_row("H2", "目標90%の必要レビュー率：組み合わせ < 自己申告（全モデル）", v2, s2, {**d2, "weak_form_models": weak}))

    res, s = auroc_h("H3", "", "cons")
    out.append(_row("H3", "AUROC：一致度 > 自己申告（全モデル）", _verdict_pos(res.values()), s, res))

    res, s = auroc_h("H4", "", "ver")
    if all(lo >= -0.05 and hi <= 0.05 for _, lo, hi in res.values()):
        v4 = "支持"
    elif any(lo > 0 or hi < 0 for _, lo, hi in res.values()):
        v4 = "不支持"
    else:
        v4 = "判定保留"
    out.append(_row("H4", "AUROC：自己検証 ≒ 自己申告（差の区間が±0.05以内、全モデル）", v4, s, res))

    c1, qual = {}, {}
    for m in models:
        k = f"{m}:self"
        r = mdiff[mdiff.model_key == k].iloc[0]
        c1[m] = (r.diff_mean, r.diff_lo, r.diff_hi)
        qual[m] = float(msumm[(msumm.model_key == k) & (msumm.method == "lcb")].share_qualified.iloc[0])
    cond1 = all(hi < 0 for _, _, hi in c1.values())
    cond2 = all(q >= 0.90 for q in qual.values())
    v5 = "支持" if cond1 and cond2 else ("不支持" if any(lo > 0 for _, lo, _ in c1.values()) else "判定保留")
    s5 = "、".join(f"{m}: 差 {d*100:+.1f}pt（{lo*100:+.1f}〜{hi*100:+.1f}）・下限法の適格率 {qual[m]:.0%}"
                   for m, (d, lo, hi) in c1.items())
    out.append(_row("H5", "下限法は δ=0.05 よりレビューが少なく、適格率90%以上（自己申告、全モデル）", v5, s5,
                    {"diff": c1, "share_qualified": qual}))

    ready = tables["ready_qualification"]
    met = tables["metrics"]
    unit = review_cost_per_case(cfg)
    c6 = {}
    for m in models:
        def get(sig):
            r = ready[(ready.model_key == f"{m}:{sig}") & (ready.a_h == 1.0) & (ready.margin == 0.05) & (ready.target == 0.90)].iloc[0]
            api = met[(met.model_key == f"{m}:{sig}") & (met.split == "test")].api_cost_jpy_per_case.iloc[0]
            return r.test_review_rate, api
        rr_s, api_s = get("self")
        rr_c, api_c = get("comb")
        d, lo, hi = d2[m]
        c6[m] = {"total_self": api_s + rr_s * unit, "total_comb": api_c + rr_c * unit,
                 "diff": (api_c - api_s) + (rr_c - rr_s) * unit, "extra_api": api_c - api_s,
                 "diff_lo": (api_c - api_s) + lo * unit, "diff_hi": (api_c - api_s) + hi * unit}
    v6 = "支持" if all(x["diff"] < 0 for x in c6.values()) else "不支持"
    s6 = "、".join(f"{m}: {x['total_self']:.1f}円→{x['total_comb']:.1f}円（追加API {x['extra_api']:.2f}円、"
                   f"差の目安 {x['diff_lo']:+.1f}〜{x['diff_hi']:+.1f}円）" for m, x in c6.items())
    out.append(_row("H6", "組み合わせ信号は追加API費用込みでも総コストを下げる（全モデル、点推定）", v6, s6, c6))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ シミュレーション
def simulate_signal_frame(cfg: dict, cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """APIを使わない動作確認用の信号表。実測ではない。"""
    from .runner import simulate_predictions
    base = simulate_predictions(cfg, cases)
    rng = np.random.default_rng(cfg["seed"] + 9)
    rows = []
    for mk in base.model_key.unique():
        b = base[base.model_key == mk].reset_index(drop=True)
        c = b.correct.to_numpy()
        cons = np.clip(np.round((0.55 + 0.4 * c + rng.normal(0, 0.2, len(b))) * 4) / 4, 0, 1)
        ver = np.clip(0.6 + 0.25 * c + rng.normal(0, 0.15, len(b)), 0, 1).round(2)
        sig = {"self": b.score.to_numpy(), "cons": cons, "ver": ver}
        sig["comb"] = (sig["self"] + sig["cons"] + sig["ver"]) / 3
        for name, sc in sig.items():
            f = b.copy()
            f["model_key"], f["base_model"], f["signal"] = f"{mk}:{name}", mk, name
            f["score"] = np.where(b.status == "ok", sc, 0.0)
            f["confidence"] = f["score"]
            f["cost_usd"] = b.cost_usd * (1 if name == "self" else 5)
            rows.append(f)
    return pd.concat(rows, ignore_index=True), pd.DataFrame()
