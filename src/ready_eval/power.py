"""必要件数の見積もり（API を使わない）。

v5 の予測を母集団の代わりとみなし、開発・最終評価の件数 n を変えて再標本化する。毎回しきい値を選び直し、
構成どうしの1件当たり総コストの差（と必要レビュー率の差）の標準偏差 SD(n) を測る。
ある真の差 Δ を、両側 α=0.05・検出力80% で見分けるには、おおよそ 2.80 × SD(n) ≤ Δ が必要（2.80 = 1.96 + 0.84）。
そこで各 n で「見分けられる最小の差」MDD(n) = 2.80 × SD(n) を出し、Δ 以下になる最小の n を読み取る。
SD が √n に反比例して減るとは限らない（確信度の値が粗く、しきい値が二つの段の間で入れ替わると、件数を増やしても
ばらつきが減らない）ので、当てはめの傾きも報告し、外挿はしない。

注意：再標本化の母集団は500件の観測データなので、真の分布の形（特に確信度の細かさ）は500件で見えた範囲に限られる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .analysis import _arrays, review_cost_per_case

Z = 1.959964 + 0.841621  # 両側5%・検出力80%

PAIRS = [("P_hv", "P_opus", "total"), ("P_sv", "P_opus", "total"), ("P_sv", "P_hv", "total"),
         ("P_sonnet", "P_opus", "total"), ("P_hv", "P_haiku", "total"),
         ("P_sv", "P_sonnet", "review"), ("P_hv", "P_opus", "review")]


def simulate_sd(cfg: dict, frame: pd.DataFrame, sizes=(250, 500, 1000, 2000, 4000, 8000, 16000), reps: int = 400,
                Y: float = 0.90, a_h: float = 1.0) -> pd.DataFrame:
    unit = review_cost_per_case(cfg)
    fx = cfg["cost"]["usd_jpy"]
    keys = sorted({k for a, b, _ in PAIRS for k in (a, b)})
    D = {k: _arrays(frame, k, "dev")[:2] for k in keys}
    T = {}
    for k in keys:
        c, s, d = _arrays(frame, k, "test")
        T[k] = (c, s, d.cost_usd.to_numpy() * fx)
    nd, nt = len(D[keys[0]][0]), len(T[keys[0]][0])
    rng = np.random.default_rng(cfg["seed"] + 808)
    rows = []
    for n in sizes:
        vals = {p: [] for p in PAIRS}
        for _ in range(reps):
            idev, itst = rng.integers(0, nd, n), rng.integers(0, nt, n)
            rr, tc = {}, {}
            for k in keys:
                sel = M.select_threshold(D[k][0][idev], D[k][1][idev], Y, a_h, 0.05)
                if not sel["feasible"]:
                    rr[k] = 1.0
                else:
                    rr[k] = float(M.threshold_review_mask(T[k][1][itst], sel["tau"]).mean())
                tc[k] = float(T[k][2][itst].mean()) + rr[k] * unit
            for p in PAIRS:
                a, b, m = p
                vals[p].append((tc[a] - tc[b]) if m == "total" else (rr[a] - rr[b]))
        for (a, b, m), v in vals.items():
            v = np.array(v)
            rows.append({"a": a, "b": b, "metric": m, "n": n, "mean": v.mean(), "sd": v.std(ddof=1)})
    return pd.DataFrame(rows)


def required_n(sd: pd.DataFrame, deltas_total=(2, 5, 10), deltas_review=(0.02, 0.05, 0.10)) -> pd.DataFrame:
    rows = []
    for (a, b, m), g in sd.groupby(["a", "b", "metric"]):
        g = g.sort_values("n")
        slope = float(np.polyfit(np.log(g.n), np.log(g.sd), 1)[0])
        row = {"a": a, "b": b, "metric": m, "fitted_slope": slope,
               "sd_at_500": float(g.loc[g.n == 500, "sd"].iloc[0]),
               "mdd_at_500": Z * float(g.loc[g.n == 500, "sd"].iloc[0]),
               "mdd_at_max_n": Z * float(g.sd.iloc[-1]), "max_n": int(g.n.iloc[-1]),
               "mean_diff_at_max_n": float(g["mean"].iloc[-1])}
        for d in (deltas_total if m == "total" else deltas_review):
            ok = g[Z * g.sd <= d]
            row[f"n_req_delta_{d}"] = int(ok.n.iloc[0]) if len(ok) else None
        obs = abs(row["mean_diff_at_max_n"])
        ok = g[Z * g.sd <= obs] if obs > 0 else g.iloc[0:0]
        row["n_req_observed_diff"] = int(ok.n.iloc[0]) if len(ok) else None
        rows.append(row)
    return pd.DataFrame(rows)
