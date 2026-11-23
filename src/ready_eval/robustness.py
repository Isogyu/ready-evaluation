"""v5 の結論の頑健さの確認（API を使わない、事後の分析で事前登録外）。

1. a_h 別の主表：a_h = 0.90（論文の主仮定）・0.95・1.00 と、目標 85%・90%・95% の組で、v5 の各構成を同じ手順
   （dev で R ≥ Y + δ、レビュー率最小、test で片側下限）で判定し直す。
2. 人間が AI の誤答を見逃す場合：レビューで AI が正しい案件を正しく保つ確率 p_c と、AI の誤答を正しく直せる確率
   p_w を分ける（主分析は p_c = p_w = 1）。期待値の信頼性で dev からしきい値を選び直す。
3. 開発データの入れ替え：v5 の dev は v2〜v4 と同じ500件なので、v4 の test 500件（v5 と重複なし）を dev とし、
   v5 の test を test として比べ直す。v4 の test には Opus の検証が付いているので、P_hg 以外の5構成を比べられる。

実行：python -m ready_eval --config config_v5.yaml robustness
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .analysis import _arrays, review_cost_per_case
from .pipelines import PIPE_LABEL, PIPELINES

V4_KEY = {"P_opus": "opus-5-5:self", "P_sonnet": "sonnet-5-5:self", "P_haiku": "haiku-5-5:self",
          "P_hv": "haiku-5-5:xver_opus-5-5", "P_sv": "sonnet-5-5:xver_opus-5-5"}


def _keys(frame):
    return [p for p in PIPELINES if p in set(frame.model_key)]


def ah_grid(cfg: dict, frame: pd.DataFrame, a_hs=(0.90, 0.95, 1.00), targets=(0.85, 0.90, 0.95)) -> pd.DataFrame:
    unit, fx, level = review_cost_per_case(cfg), cfg["cost"]["usd_jpy"], cfg["analysis"]["confidence_level"]
    rows = []
    for k in _keys(frame):
        cd, sd, _ = _arrays(frame, k, "dev")
        ct, st, d = _arrays(frame, k, "test")
        api = float(d.cost_usd.mean()) * fx
        for a_h in a_hs:
            for Y in targets:
                sel = M.select_threshold(cd, sd, Y, a_h, 0.05)
                q = M.qualify(ct, st, sel["tau"], Y, a_h, level)
                rr = q["test_review_rate"]
                rows.append({"pipeline": k, "label": PIPE_LABEL[k], "a_h": a_h, "Y": Y, "dev_feasible": sel["feasible"],
                             "tau": sel["tau"], "review_rate": rr, "reliability": q["test_reliability"], "lcb": q["lcb"],
                             "qualified": q["qualified"], "total_jpy_per_case": api + rr * unit if not np.isnan(rr) else np.nan})
    return pd.DataFrame(rows)


def _rel(c, s, tau, pc, pw):
    rev = s < tau
    return (c[~rev].sum() + (pc * c[rev] + pw * (1 - c[rev])).sum()) / len(c)


def _select_pc_pw(c, s, Y, delta, pc, pw):
    for tau in M.candidate_thresholds(s):  # 小さい順 ＝ レビュー率の小さい順
        if _rel(c, s, tau, pc, pw) >= Y + delta - 1e-12:
            return float(tau)
    return np.nan


def human_error(cfg: dict, frame: pd.DataFrame, settings=((1.0, 1.0), (1.0, 0.9), (1.0, 0.8), (1.0, 0.7), (0.99, 0.8)),
                targets=(0.85, 0.90)) -> pd.DataFrame:
    unit, fx = review_cost_per_case(cfg), cfg["cost"]["usd_jpy"]
    rows = []
    for pc, pw in settings:
        for k in [p for p in _keys(frame) if p != "P_hg"]:
            cd, sd, _ = _arrays(frame, k, "dev")
            ct, st, d = _arrays(frame, k, "test")
            api = float(d.cost_usd.mean()) * fx
            for Y in targets:
                tau = _select_pc_pw(cd, sd, Y, 0.05, pc, pw)
                ok = not np.isnan(tau)
                rr = float((st < tau).mean()) if ok else np.nan
                rows.append({"p_correct_kept": pc, "p_wrong_fixed": pw, "Y": Y, "pipeline": k, "label": PIPE_LABEL[k],
                             "tau": tau, "review_rate": rr, "reliability": _rel(ct, st, tau, pc, pw) if ok else np.nan,
                             "total_jpy_per_case": api + rr * unit if ok else np.nan})
    return pd.DataFrame(rows)


def external_dev(cfg: dict, frame5: pd.DataFrame, frame4: pd.DataFrame, targets=(0.85, 0.90, 0.95),
                 n_boot: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """dev ＝ v4 の test、test ＝ v5 の test。点推定（a_h=0.95・1.00）と、a_h=1.00 での選定込みブートストラップ。"""
    unit, fx, level = review_cost_per_case(cfg), cfg["cost"]["usd_jpy"], cfg["analysis"]["confidence_level"]
    D, T = {}, {}
    for k, k4 in V4_KEY.items():
        D[k] = _arrays(frame4, k4, "test")[:2]
        c, s, d = _arrays(frame5, k, "test")
        T[k] = (c, s, d.cost_usd.to_numpy() * fx)
    rows = []
    for Y in targets:
        for a_h in (0.95, 1.00):
            for k in V4_KEY:
                sel = M.select_threshold(*D[k], Y, a_h, 0.05)
                q = M.qualify(T[k][0], T[k][1], sel["tau"], Y, a_h, level)
                rr = q["test_review_rate"]
                rows.append({"Y": Y, "a_h": a_h, "pipeline": k, "label": PIPE_LABEL[k], "tau": sel["tau"], "review_rate": rr,
                             "reliability": q["test_reliability"], "lcb": q["lcb"], "qualified": q["qualified"],
                             "total_jpy_per_case": T[k][2].mean() + rr * unit if not np.isnan(rr) else np.nan})
    nd, nt = len(D["P_opus"][0]), len(T["P_opus"][0])
    rng = np.random.default_rng(cfg["seed"] + 909)
    rec = []
    for b in range(n_boot):
        i, j = rng.integers(0, nd, nd), rng.integers(0, nt, nt)
        for Y in targets:
            r = {}
            for k in V4_KEY:
                sel = M.select_threshold(D[k][0][i], D[k][1][i], Y, 1.0, 0.05)
                rr = float(M.threshold_review_mask(T[k][1][j], sel["tau"]).mean()) if sel["feasible"] else 1.0
                r[k] = (rr, float(T[k][2][j].mean()) + rr * unit)
            rec.append({"b": b, "Y": Y, "P_hv-P_haiku:review": r["P_hv"][0] - r["P_haiku"][0],
                        "P_sv-P_sonnet:review": r["P_sv"][0] - r["P_sonnet"][0],
                        "P_hv-P_opus:total": r["P_hv"][1] - r["P_opus"][1],
                        "P_sv-P_opus:total": r["P_sv"][1] - r["P_opus"][1]})
    bt = pd.DataFrame(rec)
    summ = []
    for Y, g in bt.groupby("Y"):
        for col in bt.columns[2:]:
            a, bb_m = col.split(":")
            x = g[col].to_numpy()
            summ.append({"Y": Y, "pair": a, "metric": bb_m, "mean": x.mean(),
                         "lo": np.percentile(x, 2.5), "hi": np.percentile(x, 97.5)})
    return pd.DataFrame(rows), pd.DataFrame(summ)
