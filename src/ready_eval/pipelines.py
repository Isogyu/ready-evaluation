"""追加実験 v5：運用構成（誰が答え、何で人間に回すか）どうしの総コスト比較。仮説は docs/hypotheses_v5.md。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .analysis import _arrays, review_cost_per_case
from .runner import collect_predictions, run_models
from .signals_v4 import cross_verifier

PIPELINES = ["P_opus", "P_sonnet", "P_haiku", "P_hv", "P_hg", "P_sv"]
PIPE_LABEL = {"P_opus": "Opus 単独", "P_sonnet": "Sonnet 単独", "P_haiku": "Haiku 単独",
              "P_hv": "Haiku＋Opus 検証（全件）", "P_hg": "Haiku＋Opus 検証（一部）", "P_sv": "Sonnet＋Opus 検証（全件）"}
VERIFY_PAIRS = [("opus-5-5", "haiku-5-5"), ("opus-5-5", "sonnet-5-5")]


def _model(cfg, key):
    return next(m for m in cfg["models"] if m["key"] == key)


def run(cfg: dict, cases: pd.DataFrame, labels: list[str], raw_root: Path, caller_factory, log=print) -> dict:
    log("1/2 元の分類（3モデル）")
    st = run_models(cfg, cases, labels, cfg["models"], raw_root, caller_factory, log=log)
    if st["stopped"]:
        return st
    log("2/2 Opus による検証（Haiku・Sonnet の予測）")
    base = collect_predictions(raw_root, cfg["models"], cases)
    for vk, tk in VERIFY_PAIRS:
        b = base[(base.model_key == tk) & (base.status == "ok")][["case_id", "predicted_label"]]
        vc = cases.merge(b.rename(columns={"predicted_label": "proposed_label"}), on="case_id")
        st = run_models(cfg, vc, labels, [cross_verifier(_model(cfg, vk), tk)], raw_root, caller_factory, log=log)
        if st["stopped"]:
            return st
    return st


def _verify_scores(cfg, raw_root, cases, base_t, vk, tk):
    ok = (base_t.status == "ok").to_numpy()
    vcases = cases.merge(base_t[ok][["predicted_label"]].rename(columns={"predicted_label": "proposed_label"}),
                         left_on="case_id", right_index=True, how="left").dropna(subset=["proposed_label"])
    vd = collect_predictions(raw_root, [cross_verifier(_model(cfg, vk), tk)], vcases).set_index("case_id").reindex(base_t.index)
    score = np.where(ok & (vd.status == "ok").to_numpy(), pd.to_numeric(vd.confidence, errors="coerce").fillna(0).to_numpy(), 0.0)
    return score, vd.cost_usd.fillna(0).to_numpy()


def assemble(base: dict, ver: dict, gate: float) -> pd.DataFrame:
    """base: モデル→予測表（case_id 索引）、ver: (検証, 対象)→(確率, 費用)。"""
    rows = []

    def add(name, b, score, cost):
        f = b.reset_index().copy()
        f["model_key"], f["pipeline"] = name, name
        f["base_model"] = name
        f["score"], f["confidence"], f["cost_usd"] = score, np.clip(score, 0, 1), cost
        rows.append(f)

    for name, mk in (("P_opus", "opus-5-5"), ("P_sonnet", "sonnet-5-5"), ("P_haiku", "haiku-5-5")):
        add(name, base[mk], base[mk].score.to_numpy(), base[mk].cost_usd.fillna(0).to_numpy())
    hb = base["haiku-5-5"]
    hs, hc = ver[("opus-5-5", "haiku-5-5")]
    add("P_hv", hb, hs, hb.cost_usd.fillna(0).to_numpy() + hc)
    gated = hb.score.to_numpy() < gate
    add("P_hg", hb, np.where(gated, hs, 1.0 + hb.score.to_numpy()), hb.cost_usd.fillna(0).to_numpy() + np.where(gated, hc, 0.0))
    sb = base["sonnet-5-5"]
    ss, sc = ver[("opus-5-5", "sonnet-5-5")]
    add("P_sv", sb, ss, sb.cost_usd.fillna(0).to_numpy() + sc)
    return pd.concat(rows, ignore_index=True)


def build_frame(cfg: dict, cases: pd.DataFrame, raw_root: Path) -> tuple[pd.DataFrame, float]:
    base = {m["key"]: collect_predictions(raw_root, [m], cases).set_index("case_id") for m in cfg["models"]}
    ver = {(vk, tk): _verify_scores(cfg, raw_root, cases, base[tk], vk, tk) for vk, tk in VERIFY_PAIRS}
    hb = base["haiku-5-5"]
    gate = float(np.median(hb.loc[hb.split == "dev", "score"]))
    return assemble(base, ver, gate), gate


def cost_bootstrap(cfg: dict, frame: pd.DataFrame, Y: float = 0.90, a_h: float = 1.0, n_boot: int = 1000):
    unit = review_cost_per_case(cfg)
    fx = cfg["cost"]["usd_jpy"]
    level = cfg["analysis"]["confidence_level"]
    keys = [p for p in PIPELINES if p in set(frame.model_key)]
    D, T, C = {}, {}, {}
    for k in keys:
        D[k] = _arrays(frame, k, "dev")[:2]
        c, s, d = _arrays(frame, k, "test")
        T[k] = (c, s)
        C[k] = d.cost_usd.to_numpy() * fx
    n_dev, n_test = len(D[keys[0]][0]), len(T[keys[0]][0])
    rng = np.random.default_rng(cfg["seed"] + 707)
    rec = []
    for b in range(n_boot):
        idev, itst = rng.integers(0, n_dev, n_dev), rng.integers(0, n_test, n_test)
        for k in keys:
            sel = M.select_threshold(D[k][0][idev], D[k][1][idev], Y, a_h, 0.05)
            q = M.qualify(T[k][0][itst], T[k][1][itst], sel["tau"], Y, a_h, level)
            rr = q["test_review_rate"]
            api = float(C[k][itst].mean())
            rec.append({"b": b, "pipeline": k, "review_rate": rr, "api_jpy": api,
                        "total_jpy": api + rr * unit if not np.isnan(rr) else np.nan, "qualified": q["qualified"]})
    boot = pd.DataFrame(rec)
    pct = lambda p: (lambda x: np.nanpercentile(x, p))  # noqa: E731
    summ = boot.groupby("pipeline").agg(review_rate_mean=("review_rate", "mean"), review_rate_lo=("review_rate", pct(2.5)),
                                        review_rate_hi=("review_rate", pct(97.5)), total_mean=("total_jpy", "mean"),
                                        total_lo=("total_jpy", pct(2.5)), total_hi=("total_jpy", pct(97.5)),
                                        api_mean=("api_jpy", "mean"), share_qualified=("qualified", "mean")).reset_index()
    w_rr = boot.pivot(index="b", columns="pipeline", values="review_rate")
    w_tc = boot.pivot(index="b", columns="pipeline", values="total_jpy")
    diffs = []
    for a, bb, what in (("P_hv", "P_opus", "total"), ("P_hg", "P_hv", "review"), ("P_hg", "P_hv", "total"),
                        ("P_hv", "P_haiku", "review"), ("P_sv", "P_sonnet", "review"), ("P_sv", "P_opus", "total"),
                        ("P_hg", "P_opus", "total"), ("P_hv", "P_opus", "review")):
        x = ((w_tc if what == "total" else w_rr)[a] - (w_tc if what == "total" else w_rr)[bb]).dropna()
        diffs.append({"a": a, "b": bb, "metric": what, "mean": x.mean(), "lo": np.percentile(x, 2.5), "hi": np.percentile(x, 97.5)})
    return summ, pd.DataFrame(diffs)


def point_table(cfg: dict, frame: pd.DataFrame, Y: float = 0.90, a_h: float = 1.0) -> pd.DataFrame:
    unit = review_cost_per_case(cfg)
    fx = cfg["cost"]["usd_jpy"]
    level = cfg["analysis"]["confidence_level"]
    rows = []
    for k in [p for p in PIPELINES if p in set(frame.model_key)]:
        cd, sd, _ = _arrays(frame, k, "dev")
        ct, st, d = _arrays(frame, k, "test")
        sel = M.select_threshold(cd, sd, Y, a_h, 0.05)
        q = M.qualify(ct, st, sel["tau"], Y, a_h, level)
        api = float(d.cost_usd.mean()) * fx
        rows.append({"pipeline": k, "label": PIPE_LABEL[k], "ai_accuracy": float(ct.mean()), "tau": sel["tau"],
                     "review_rate": q["test_review_rate"], "reliability": q["test_reliability"], "lcb": q["lcb"],
                     "qualified": q["qualified"], "api_jpy_per_case": api,
                     "review_jpy_per_case": q["test_review_rate"] * unit, "total_jpy_per_case": api + q["test_review_rate"] * unit,
                     "share_verified": float((st <= 1.0).mean()) if k == "P_hg" else (1.0 if k in ("P_hv", "P_sv") else 0.0)})
    return pd.DataFrame(rows)


def sensitivity(cfg: dict, point: pd.DataFrame) -> pd.DataFrame:
    sens = cfg["cost"]["sensitivity"]
    rows = []
    for w in sens["wage_jpy_per_hour"]:
        for mi in sens["review_minutes"]:
            unit = review_cost_per_case(cfg, w, mi)
            g = point.copy()
            g["total"] = g.api_jpy_per_case + g.review_rate * unit
            g["rank"] = g.total.rank(method="min")
            for _, r in g.iterrows():
                rows.append({"wage_jpy_per_hour": w, "review_minutes": mi, "pipeline": r.pipeline,
                             "total_jpy_per_case": r.total, "rank": r["rank"]})
    return pd.DataFrame(rows)


def _row(hid, hyp, verdict, summary, details):
    return {"id": hid, "hypothesis": hyp, "verdict": verdict, "summary": summary,
            "details": json.dumps(details, ensure_ascii=False, default=float)}


def evaluate(diffs: pd.DataFrame, summ: pd.DataFrame, point: pd.DataFrame) -> pd.DataFrame:
    def get(a, b, m):
        r = diffs[(diffs.a == a) & (diffs.b == b) & (diffs.metric == m)].iloc[0]
        return float(r["mean"]), float(r.lo), float(r.hi)

    def v_neg(lo, hi):
        return "支持" if hi < 0 else ("不支持" if lo > 0 else "判定保留")

    out = []
    d, lo, hi = get("P_hv", "P_opus", "total")
    p = point.set_index("pipeline")
    out.append(_row("H1", "総コスト：Haiku＋Opus 検証 < Opus 単独", v_neg(lo, hi),
                    f"差 {d:+.1f}円（{lo:+.1f}〜{hi:+.1f}）。点推定 {p.loc['P_hv','total_jpy_per_case']:.1f}円 対 {p.loc['P_opus','total_jpy_per_case']:.1f}円",
                    {"mean": d, "lo": lo, "hi": hi}))
    q_ok = bool(p.loc["P_hv", "qualified"])
    share = float(summ.set_index("pipeline").loc["P_hv", "share_qualified"])
    v2 = "支持" if q_ok and share >= 0.90 else ("不支持" if not q_ok else "判定保留")
    out.append(_row("H2", "Haiku＋Opus 検証は適格で、適格率90%以上", v2,
                    f"最終評価データの下限 {p.loc['P_hv','lcb']:.3f}（{'適格' if q_ok else '不適格'}）、ブートストラップでの適格率 {share:.0%}",
                    {"qualified": q_ok, "share_qualified": share}))
    d, lo, hi = get("P_hg", "P_hv", "review")
    tc_ok = p.loc["P_hg", "total_jpy_per_case"] <= p.loc["P_hv", "total_jpy_per_case"]
    v3 = "支持" if hi < 0.05 and tc_ok else ("不支持" if lo > 0.05 else "判定保留")
    out.append(_row("H3", "検証を一部に絞っても、レビュー率の増加は+5pt未満で総コストも増えない", v3,
                    f"レビュー率の差 {d*100:+.1f}pt（{lo*100:+.1f}〜{hi*100:+.1f}）、総コスト {p.loc['P_hg','total_jpy_per_case']:.1f}円 対 {p.loc['P_hv','total_jpy_per_case']:.1f}円、"
                    f"検証した割合 {p.loc['P_hg','share_verified']:.0%}", {"mean": d, "lo": lo, "hi": hi, "total_ok": bool(tc_ok)}))
    d, lo, hi = get("P_hv", "P_haiku", "review")
    out.append(_row("H4", "v4 の再現：Haiku＋Opus 検証は Haiku 単独よりレビュー率が低い", v_neg(lo, hi),
                    f"差 {d*100:+.1f}pt（{lo*100:+.1f}〜{hi*100:+.1f}）", {"mean": d, "lo": lo, "hi": hi}))
    d, lo, hi = get("P_sv", "P_sonnet", "review")
    out.append(_row("H5", "Sonnet＋Opus 検証は Sonnet 単独よりレビュー率が低い", v_neg(lo, hi),
                    f"差 {d*100:+.1f}pt（{lo*100:+.1f}〜{hi*100:+.1f}）", {"mean": d, "lo": lo, "hi": hi}))
    return pd.DataFrame(out)


def simulate(cfg: dict, cases: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """API を使わない動作確認用（架空の値。実測ではない）。"""
    rng = np.random.default_rng(cfg["seed"] + 13)
    n = len(cases)
    diff = rng.normal(0, 1.5, n)
    acc = {"opus-5-5": 0.87, "sonnet-5-5": 0.84, "haiku-5-5": 0.77}
    price = {"opus-5-5": 0.0016, "sonnet-5-5": 0.0008, "haiku-5-5": 0.00004}
    base = {}
    for k, a in acc.items():
        c = (rng.random(n) < 1 / (1 + np.exp(-(np.log(a / (1 - a)) - diff)))).astype(int)
        f = cases.copy().set_index("case_id")
        f["status"], f["correct"] = "ok", c
        f["predicted_id"] = np.where(c == 1, f.label_id, (f.label_id + 1) % 77)
        f["score"] = np.clip(0.75 + 0.15 * c + rng.normal(0, 0.12, n), 0, 1).round(2)
        f["cost_usd"] = price[k]
        f["model_id"] = f"simulation:{k}"
        for col in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            f[col] = np.nan
        base[k] = f
    ver = {}
    for vk, tk in VERIFY_PAIRS:
        c = base[tk].correct.to_numpy()
        ver[(vk, tk)] = (np.clip(0.5 + 0.4 * c + rng.normal(0, 0.12, n), 0, 1).round(2), np.full(n, 0.0008))
    hb = base["haiku-5-5"]
    gate = float(np.median(hb.loc[hb.split == "dev", "score"]))
    return assemble(base, ver, gate), gate
