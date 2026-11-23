"""予測結果から指標・方針比較・適格判定・コストの表を作る。"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import metrics as M


def _arrays(df: pd.DataFrame, model: str, split: str):
    d = df[(df.model_key == model) & (df.split == split)].sort_values("case_id")
    return d["correct"].to_numpy(), d["score"].to_numpy(), d


def review_cost_per_case(cfg: dict, wage: float | None = None, minutes: float | None = None) -> float:
    c = cfg["cost"]
    wage = c["wage_jpy_per_hour"] if wage is None else wage
    minutes = c["review_minutes"] if minutes is None else minutes
    return wage / 60.0 * minutes


def api_cost_per_case_jpy(cfg: dict, d: pd.DataFrame) -> float:
    return float(d["cost_usd"].fillna(0).mean()) * cfg["cost"]["usd_jpy"]


def nocache_cost_jpy(cfg: dict, model: str, d: pd.DataFrame) -> float:
    """キャッシュが一切効かなかった場合の1件当たりAPI費用（感度確認用）。"""
    price = {m["key"]: m for m in cfg.get("models", [])}.get(model)
    if price is None or d["input_tokens"].isna().all():
        return np.nan
    tok_in = sum(pd.to_numeric(d[c]).fillna(0) for c in
                 ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    usd = tok_in * price["price_in"] / 1e6 + pd.to_numeric(d["output_tokens"]).fillna(0) * price["price_out"] / 1e6
    return float(usd.mean()) * cfg["cost"]["usd_jpy"]


def model_metrics(cfg: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = cfg["analysis"]
    rows, bins = [], []
    for model in df.model_key.unique():
        for split in ("dev", "test"):
            c, s, d = _arrays(df, model, split)
            n, k = len(c), int(c.sum())
            lo, hi = M.wilson_ci(k, n)
            n_err = n - k
            stable = n_err >= a["min_errors_for_auroc"]
            au = M.auroc_error_detection(c, s)
            au_lo, au_hi = M.bootstrap_auroc(c, s, a["bootstrap_n"], cfg["seed"]) if not np.isnan(au) else (np.nan, np.nan)
            row = {
                "model_key": model, "model_id": d["model_id"].iloc[0], "split": split, "n": n,
                "accuracy": k / n, "accuracy_ci_lo": lo, "accuracy_ci_hi": hi, "n_errors": n_err,
                "brier": M.brier(c, s), f"ece_{a['ece_bins']}bins": M.ece(c, s, a["ece_bins"]),
                "ece_15bins": M.ece(c, s, 15), "auroc_error_detection": au,
                "auroc_ci_lo": au_lo, "auroc_ci_hi": au_hi, "aurc": M.aurc(c, s),
                "auroc_stable": bool(stable and not np.isnan(au)),
                "mean_confidence_ok": float(pd.to_numeric(d.loc[d.status == "ok", "confidence"]).mean()),
                "n_distinct_confidence": int(d.loc[d.status == "ok", "confidence"].nunique()),
                "api_cost_usd_total": float(d["cost_usd"].fillna(0).sum()),
                "api_cost_jpy_per_case": api_cost_per_case_jpy(cfg, d),
                "api_cost_jpy_per_case_nocache": nocache_cost_jpy(cfg, model, d),
                "mean_input_tokens": float(pd.to_numeric(d["input_tokens"]).mean()),
                "mean_output_tokens": float(pd.to_numeric(d["output_tokens"]).mean()),
            }
            for st in ("ok", "invalid_output", "invalid_label", "api_error", "timeout", "refusal", "max_tokens", "not_run"):
                row[f"n_{st}"] = int((d.status == st).sum())
            rows.append(row)
            for b in M.calibration_table(c, s, a["ece_bins"]):
                bins.append({"model_key": model, "split": split, **b})
    return pd.DataFrame(rows), pd.DataFrame(bins)


def policy_curves(cfg: dict, df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    a = cfg["analysis"]
    rates = np.round(np.arange(0, 1 + 1e-9, a["review_rate_grid_step"]), 4)
    rows = []
    for model in df.model_key.unique():
        c, s, _ = _arrays(df, model, split)
        for a_h in a["human_success_rates"]:
            for r in M.confidence_policy_curve(c, s, a_h):
                rows.append({"model_key": model, "split": split, "a_h": a_h, "policy": "D_confidence", **r})
            for r in M.random_policy_curve(c, rates, a_h, a["random_review_seeds"], cfg["seed"]):
                rows.append({"model_key": model, "split": split, "a_h": a_h, "policy": "C_random", **r})
    return pd.DataFrame(rows)


def ready_table(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    a = cfg["analysis"]
    rows = []
    for model in df.model_key.unique():
        cd, sd, _ = _arrays(df, model, "dev")
        ct, st, dt = _arrays(df, model, "test")
        api_jpy = api_cost_per_case_jpy(cfg, dt)
        for a_h in a["human_success_rates"]:
            for margin in a["margins"]:
                for Y in a["target_reliabilities"]:
                    sel = M.select_threshold(cd, sd, Y, a_h, margin)
                    q = M.qualify(ct, st, sel["tau"], Y, a_h, a["confidence_level"])
                    boot = (M.bootstrap_fixed_tau(ct, st, sel["tau"], a_h, a["bootstrap_n"], cfg["seed"])
                            if sel["feasible"] else {})
                    oracle = M.select_threshold(ct, st, Y, a_h, 0.0)  # 参考：最終評価データ上での事後的な最小値
                    rr = q["test_review_rate"]
                    rows.append({
                        "model_key": model, "a_h": a_h, "margin": margin, "target": Y, **sel, **q, **boot,
                        "test_oracle_review_rate": oracle["dev_review_rate"],
                        "api_cost_jpy_per_case": api_jpy,
                        "review_cost_jpy_per_case": rr * review_cost_per_case(cfg) if not np.isnan(rr) else np.nan,
                        "total_cost_jpy_per_case": api_jpy + rr * review_cost_per_case(cfg) if not np.isnan(rr) else np.nan,
                    })
    return pd.DataFrame(rows)


def policy_summary(cfg: dict, df: pd.DataFrame, ready: pd.DataFrame) -> pd.DataFrame:
    """A/B/C/D を同じ表で比較（最終評価データ、a_h=主仮定、δ=0.05）。C は D と同じレビュー率で比べる。"""
    a = cfg["analysis"]
    a_h = a["primary_human_success"]
    unit = review_cost_per_case(cfg)
    rows = []
    for model in df.model_key.unique():
        c, s, d = _arrays(df, model, "test")
        api = api_cost_per_case_jpy(cfg, d)
        acc = float(c.mean())

        def add(policy, target, rr, rel, extra=None):
            rows.append({"model_key": model, "policy": policy, "target": target, "a_h": a_h, "ai_accuracy": acc,
                         "review_rate": rr, "reliability_after_review": rel, "api_cost_jpy_per_case": api,
                         "review_cost_jpy_per_case": rr * unit, "total_cost_jpy_per_case": api + rr * unit,
                         "total_cost_jpy_all_cases": (api + rr * unit) * len(c), **(extra or {})})

        add("A_ai_only", None, 0.0, acc)
        add("B_full_review", None, 1.0, a_h)
        sub = ready[(ready.model_key == model) & (ready.a_h == a_h) & (ready.margin == 0.05)]
        for _, r in sub.iterrows():
            if not r["feasible"]:
                add("D_confidence", r["target"], np.nan, np.nan, {"note": "dev_infeasible"})
                continue
            add("D_confidence", r["target"], r["test_review_rate"], r["test_reliability"],
                {"lcb": r["lcb"], "qualified": r["qualified"]})
            rr = r["test_review_rate"]
            rc = M.random_policy_curve(c, [rr], a_h, a["random_review_seeds"], cfg["seed"])[0]
            add("C_random", r["target"], rc["review_rate"], rc["reliability"],
                {"rel_p2_5": rc["rel_p2_5"], "rel_p97_5": rc["rel_p97_5"], "note": "Dと同じレビュー率"})
    return pd.DataFrame(rows)


def cost_sensitivity(cfg: dict, ready: pd.DataFrame) -> pd.DataFrame:
    """人件費×レビュー時間の格子で、各目標信頼性での1件当たり総コストと順位を出す。"""
    a, sens = cfg["analysis"], cfg["cost"]["sensitivity"]
    sub = ready[(ready.margin == 0.05)]
    rows = []
    for wage in sens["wage_jpy_per_hour"]:
        for minutes in sens["review_minutes"]:
            unit = review_cost_per_case(cfg, wage, minutes)
            for (a_h, Y), g in sub.groupby(["a_h", "target"]):
                g = g.copy()
                g["total"] = g["api_cost_jpy_per_case"] + g["test_review_rate"] * unit
                g["rank_by_total_cost"] = g["total"].rank(method="min")
                for _, r in g.iterrows():
                    rows.append({"wage_jpy_per_hour": wage, "review_minutes": minutes, "a_h": a_h, "target": Y,
                                 "model_key": r["model_key"], "review_rate": r["test_review_rate"],
                                 "api_cost_jpy_per_case": r["api_cost_jpy_per_case"],
                                 "review_cost_jpy_per_case": r["test_review_rate"] * unit if r["feasible"] else np.nan,
                                 "total_cost_jpy_per_case": r["total"] if r["feasible"] else np.nan,
                                 "rank_by_total_cost": r["rank_by_total_cost"] if r["feasible"] else np.nan,
                                 "qualified": r["qualified"]})
    return pd.DataFrame(rows)


def pairwise_tests(cfg: dict, df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    a = cfg["analysis"]
    models = list(df.model_key.unique())
    rows = []
    for m1, m2 in combinations(models, 2):
        c1, _, _ = _arrays(df, m1, split)
        c2, _, _ = _arrays(df, m2, split)
        mc = M.mcnemar_exact(c1, c2)
        diff, lo, hi = M.paired_bootstrap_diff(c1, c2, a["bootstrap_n"], cfg["seed"])
        rows.append({"model_1": m1, "model_2": m2, "split": split, "accuracy_diff": diff,
                     "diff_ci_lo": lo, "diff_ci_hi": hi, **mc})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = M.holm(out["p_value"].tolist())
    return out


def selection_bootstrap(cfg: dict, df: pd.DataFrame, n_boot: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """しきい値選定まで含めた不確実性。開発・最終評価の両データをケース単位で再標本化し、
    毎回しきい値を選び直して最終評価データでのレビュー率・判定を記録する。
    同じ再標本を全モデルに使うので、モデル間の差は対応ありで比べられる。
    """
    a = cfg["analysis"]
    n_boot = n_boot or max(200, a["bootstrap_n"] // 2)
    models = list(df.model_key.unique())
    dev = {m: _arrays(df, m, "dev")[:2] for m in models}
    tst = {m: _arrays(df, m, "test")[:2] for m in models}
    n_dev, n_test = len(next(iter(dev.values()))[0]), len(next(iter(tst.values()))[0])
    rng = np.random.default_rng(cfg["seed"] + 101)
    rec = []
    for b in range(n_boot):
        idev, itst = rng.integers(0, n_dev, n_dev), rng.integers(0, n_test, n_test)
        for a_h in a["human_success_rates"]:
            for Y in a["target_reliabilities"]:
                for m in models:
                    cd, sd = dev[m][0][idev], dev[m][1][idev]
                    ct, st = tst[m][0][itst], tst[m][1][itst]
                    sel = M.select_threshold(cd, sd, Y, a_h, 0.05)
                    q = M.qualify(ct, st, sel["tau"], Y, a_h, a["confidence_level"])
                    rec.append({"b": b, "a_h": a_h, "target": Y, "model_key": m, "feasible": sel["feasible"],
                                "review_rate": q["test_review_rate"], "qualified": q["qualified"]})
    boot = pd.DataFrame(rec)
    summ = (boot.groupby(["a_h", "target", "model_key"])
            .agg(review_rate_mean=("review_rate", "mean"),
                 review_rate_lo=("review_rate", lambda x: np.nanpercentile(x, 2.5) if x.notna().any() else np.nan),
                 review_rate_hi=("review_rate", lambda x: np.nanpercentile(x, 97.5) if x.notna().any() else np.nan),
                 p_dev_feasible=("feasible", "mean"), p_qualified=("qualified", "mean"))
            .reset_index())
    diffs = []
    wide = boot.pivot_table(index=["b", "a_h", "target"], columns="model_key", values="review_rate")
    for m1, m2 in combinations(models, 2):
        d = (wide[m1] - wide[m2]).groupby(level=["a_h", "target"])
        for (a_h, Y), x in d:
            x = x.dropna()
            if len(x) < 0.5 * n_boot:
                diffs.append({"a_h": a_h, "target": Y, "model_1": m1, "model_2": m2, "n_valid": len(x)})
                continue
            diffs.append({"a_h": a_h, "target": Y, "model_1": m1, "model_2": m2, "n_valid": len(x),
                          "review_rate_diff_mean": x.mean(), "diff_lo": np.percentile(x, 2.5),
                          "diff_hi": np.percentile(x, 97.5), "diff_lo90": np.percentile(x, 5),
                          "diff_hi90": np.percentile(x, 95), "share_model1_less_review": float((x < 0).mean())})
    return summ, pd.DataFrame(diffs)


def run_all(cfg: dict, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    incomplete = df[df.status == "not_run"]
    if len(incomplete):
        raise RuntimeError(f"未実行のケースが {len(incomplete)} 件あります。実行を完了してから分析してください。")
    met, bins = model_metrics(cfg, df)
    ready = ready_table(cfg, df)
    sel_summ, sel_diff = selection_bootstrap(cfg, df)
    return {
        "selection_bootstrap": sel_summ, "selection_bootstrap_diffs": sel_diff,
        "metrics": met, "calibration_bins": bins, "policy_curves": policy_curves(cfg, df),
        "ready_qualification": ready, "policy_summary": policy_summary(cfg, df, ready),
        "cost_sensitivity": cost_sensitivity(cfg, ready), "pairwise_tests": pairwise_tests(cfg, df),
    }
