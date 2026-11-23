"""追加実験（v2）の事前登録仮説 H1〜H6 の判定。基準は docs/hypotheses_v2.md と同じ。

モデルは役割で指定する（top=Opus, mid=Sonnet, low=Haiku）。シミュレーションでは simulation.roles を使う。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import metrics as M
from .analysis import _arrays, review_cost_per_case


def _roles(cfg: dict, df: pd.DataFrame) -> dict:
    roles = cfg["hypotheses"]["roles"]
    if not set(roles.values()) <= set(df.model_key):
        roles = cfg["simulation"]["roles"]
    return roles


def _row(hid, hypothesis, verdict, summary, details):
    return {"id": hid, "hypothesis": hypothesis, "verdict": verdict, "summary": summary,
            "details": json.dumps(details, ensure_ascii=False, default=float)}


def _diff(diffs: pd.DataFrame, a_h, Y, m1, m2, col_lo="diff_lo", col_hi="diff_hi"):
    """m1−m2 のレビュー率差（平均、区間）。表の並びが逆なら符号を反転する。"""
    sel = diffs[(diffs.a_h == a_h) & (diffs.target == Y)]
    r = sel[(sel.model_1 == m1) & (sel.model_2 == m2)]
    if len(r):
        r = r.iloc[0]
        return r.get("review_rate_diff_mean", np.nan), r.get(col_lo, np.nan), r.get(col_hi, np.nan)
    r = sel[(sel.model_1 == m2) & (sel.model_2 == m1)].iloc[0]
    return -r.get("review_rate_diff_mean", np.nan), -r.get(col_hi, np.nan), -r.get(col_lo, np.nan)


def h1(cfg, df, t, R):
    pw = t["pairwise_tests"]
    out, ok_dir, ok_sig = {}, True, True
    for a, b in (("top", "mid"), ("top", "low"), ("mid", "low")):
        m1, m2 = R[a], R[b]
        r = pw[((pw.model_1 == m1) & (pw.model_2 == m2)) | ((pw.model_1 == m2) & (pw.model_2 == m1))].iloc[0]
        d = r.accuracy_diff if r.model_1 == m1 else -r.accuracy_diff
        out[f"{m1}-{m2}"] = {"diff": d, "p_holm": r.p_holm}
        ok_dir &= d > 0
        ok_sig &= r.p_holm < 0.05
    v = "支持" if ok_dir and ok_sig else ("不支持" if not ok_dir else "判定保留")
    s = "、".join(f"{k}: {x['diff']*100:+.1f}pt（p={x['p_holm']:.3g}）" for k, x in out.items())
    return _row("H1", "正答率は top > mid > low で3組とも有意", v, s, out)


def h2(cfg, df, t, R):
    ps = t["policy_summary"]
    out, ok = {}, True
    for m in R.values():
        d = ps[(ps.model_key == m) & (ps.policy == "D_confidence") & (ps.target == 0.90)].iloc[0]
        c = ps[(ps.model_key == m) & (ps.policy == "C_random") & (ps.target == 0.90)]
        if c.empty or pd.isna(d.reliability_after_review):
            ok = False
            out[m] = "比較不能"
            continue
        c = c.iloc[0]
        out[m] = {"review_rate": d.review_rate, "D": d.reliability_after_review, "C_mean": c.reliability_after_review,
                  "C_p97_5": c.rel_p97_5}
        ok &= d.reliability_after_review > c.rel_p97_5
    s = "、".join(f"{m}: D {x['D']:.1%} 対 C {x['C_mean']:.1%}（97.5%点 {x['C_p97_5']:.1%}）"
                 for m, x in out.items() if isinstance(x, dict))
    return _row("H2", "同じレビュー率で確信度順 > ランダム（全モデル）", "支持" if ok else "不支持", s, out)


def h3a(cfg, df, t, R):
    d = t["selection_bootstrap_diffs"]
    mean, lo90, hi90 = _diff(d, 1.0, 0.90, R["top"], R["mid"], "diff_lo90", "diff_hi90")
    _, lo95, hi95 = _diff(d, 1.0, 0.90, R["top"], R["mid"])
    if lo90 >= -0.10 and hi90 <= 0.10:
        v = "支持"
    elif lo95 > 0 or hi95 < 0:
        v = "不支持"
    else:
        v = "判定保留"
    s = f"{R['top']}−{R['mid']}: {mean*100:+.1f}pt（90%区間 {lo90*100:+.1f}〜{hi90*100:+.1f}、95%区間 {lo95*100:+.1f}〜{hi95*100:+.1f}）"
    return _row("H3a", "目標90%で top と mid の必要レビュー率の差は ±10pt 以内", v, s,
                {"mean": mean, "lo90": lo90, "hi90": hi90, "lo95": lo95, "hi95": hi95})


def h3b(cfg, df, t, R):
    d = t["selection_bootstrap_diffs"]
    res, verdicts = {}, []
    for other in ("mid", "top"):
        mean, lo, hi = _diff(d, 1.0, 0.95, R["low"], R[other])
        res[f"{R['low']}-{R[other]}"] = {"mean": mean, "lo": lo, "hi": hi}
        verdicts.append("pos" if lo > 0 else ("neg" if hi < 0 else "na"))
    v = "支持" if all(x == "pos" for x in verdicts) else ("不支持" if "neg" in verdicts else "判定保留")
    s = "、".join(f"{k}: {x['mean']*100:+.1f}pt（95%区間 {x['lo']*100:+.1f}〜{x['hi']*100:+.1f}）" for k, x in res.items())
    return _row("H3b", "目標95%で low は mid・top より多くのレビューを要する", v, s, res)


def _boot(cfg, fn, arrays, n_boot):
    rng = np.random.default_rng(cfg["seed"] + 202)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = fn(*[a[i] for a in arrays])
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def h4a(cfg, df, t, R):
    n_boot = cfg["analysis"]["bootstrap_n"]
    out, ok = {}, True
    for role in ("top", "mid"):
        c, s, _ = _arrays(df, R[role], "test")
        gap = float(s.mean() - c.mean())
        lo, hi = _boot(cfg, lambda cc, ss: ss.mean() - cc.mean(), [c, s], n_boot)
        out[R[role]] = {"gap": gap, "lo": lo, "hi": hi}
        ok &= hi < 0
    s = "、".join(f"{m}: {x['gap']*100:+.1f}pt（95%区間 {x['lo']*100:+.1f}〜{x['hi']*100:+.1f}）" for m, x in out.items())
    return _row("H4a", "top と mid は確信度を低めに申告する（平均確信度−正答率 < 0）", "支持" if ok else "不支持", s, out)


def h4b(cfg, df, t, R):
    n_boot = cfg["analysis"]["bootstrap_n"]
    bins = cfg["analysis"]["ece_bins"]
    cl, sl, _ = _arrays(df, R["low"], "test")
    ct, st, _ = _arrays(df, R["top"], "test")
    ece_d = M.ece(cl, sl, bins) - M.ece(ct, st, bins)
    au_d = M.auroc_error_detection(ct, st) - M.auroc_error_detection(cl, sl)
    e_lo, e_hi = _boot(cfg, lambda a, b, c, d: M.ece(a, b, bins) - M.ece(c, d, bins), [cl, sl, ct, st], n_boot)
    a_lo, a_hi = _boot(cfg, lambda a, b, c, d: M.auroc_error_detection(c, d) - M.auroc_error_detection(a, b),
                       [cl, sl, ct, st], n_boot)
    ece_ok, au_ok = e_hi < 0, a_lo > 0
    ece_bad, au_bad = e_lo > 0, a_hi < 0
    v = "支持" if ece_ok and au_ok else ("不支持" if ece_bad or au_bad else "判定保留")
    s = (f"ECE {R['low']}−{R['top']}: {ece_d:+.3f}（95%区間 {e_lo:+.3f}〜{e_hi:+.3f}）、"
         f"AUROC {R['top']}−{R['low']}: {au_d:+.3f}（95%区間 {a_lo:+.3f}〜{a_hi:+.3f}）")
    return _row("H4b", "low は top より ECE が小さいが、AUROC は低い", v, s,
                {"ece_diff": ece_d, "ece_lo": e_lo, "ece_hi": e_hi, "auroc_diff": au_d, "auroc_lo": a_lo, "auroc_hi": a_hi})


def h5(cfg, df, t, R):
    met, ready = t["metrics"], t["ready_qualification"]
    sens = cfg["cost"]["sensitivity"]
    worst, out, ok = 0.0, {}, True
    for m in R.values():
        mm = met[(met.model_key == m) & (met.split == "test")].iloc[0]
        api = mm.api_cost_jpy_per_case_nocache
        if pd.isna(api):
            api = mm.api_cost_jpy_per_case
        r = ready[(ready.model_key == m) & (ready.a_h == 1.0) & (ready.margin == 0.05) & (ready.target == 0.90)].iloc[0]
        rr = r.test_review_rate
        if pd.isna(rr) or rr == 0:
            out[m] = {"review_rate": rr, "note": "レビュー率0または選定不能のため対象外"}
            continue
        shares = [api / (api + rr * review_cost_per_case(cfg, w, mi))
                  for w in sens["wage_jpy_per_hour"] for mi in sens["review_minutes"]]
        out[m] = {"api_jpy_nocache": api, "review_rate": rr, "max_share": max(shares)}
        worst = max(worst, max(shares))
        ok &= max(shares) < 0.05
    s = f"API費用の割合の最大値 {worst:.2%}（" + "、".join(
        f"{m}: {x['max_share']:.2%}" for m, x in out.items() if "max_share" in x) + "）"
    return _row("H5", "目標90%の総コストに占めるAPI費用は全条件で5%未満", "支持" if ok else "不支持", s, out)


def h6(cfg, df, t, R, n_rep: int = 1000):
    rng = np.random.default_rng(cfg["seed"] + 303)
    Y, a_h, margin = 0.90, 1.0, 0.05
    out, ok = {}, True
    for m in R.values():
        cd, sd, _ = _arrays(df, m, "dev")
        ct, st, _ = _arrays(df, m, "test")
        oracle = M.select_threshold(ct, st, Y, a_h, 0.0)["dev_review_rate"]
        res = {}
        for label, size, replace in (("dev100", 100, False), ("dev500", len(cd), True)):
            dev_ = []
            for _ in range(n_rep):
                i = rng.choice(len(cd), size=size, replace=replace)
                sel = M.select_threshold(cd[i], sd[i], Y, a_h, margin)
                rr = M.reliability_after_review(ct, M.threshold_review_mask(st, sel["tau"]), a_h)["review_rate"] \
                    if sel["feasible"] else 1.0
                dev_.append(rr - oracle)
            dev_ = np.array(dev_)
            res[label] = {"mean_overshoot": float(dev_.mean()), "mean_abs": float(np.abs(dev_).mean()),
                          "sd": float(dev_.std())}
        out[m] = {"oracle": oracle, **res}
        ok &= res["dev100"]["mean_abs"] > res["dev500"]["mean_abs"] and res["dev100"]["sd"] > res["dev500"]["sd"]
    s = "、".join(f"{m}: 平均絶対ずれ {x['dev100']['mean_abs']*100:.1f}pt→{x['dev500']['mean_abs']*100:.1f}pt、"
                 f"SD {x['dev100']['sd']*100:.1f}→{x['dev500']['sd']*100:.1f}（オラクル {x['oracle']*100:.1f}%）"
                 for m, x in out.items())
    return _row("H6", "dev 100件で選ぶと dev 500件より過剰レビューのずれ・ばらつきが大きい", "支持" if ok else "不支持", s, out)


def evaluate(cfg: dict, df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    R = _roles(cfg, df)
    rows = [f(cfg, df, tables, R) for f in (h1, h2, h3a, h3b, h4a, h4b, h5, h6)]
    return pd.DataFrame(rows)


def exploratory_margin(cfg: dict, df: pd.DataFrame, n_rep: int = 1000) -> pd.DataFrame:
    """探索的分析（事前登録外）：安全余裕 δ と開発データ件数が、選定レビュー率のずれ（対オラクル）に与える影響。
    目標90%・a_h=1.00。偏り（平均のずれ）・平均絶対ずれ・標準偏差・最終評価データでの適格率を出す。"""
    rng = np.random.default_rng(cfg["seed"] + 404)
    rows = []
    for m in df.model_key.unique():
        cd, sd, _ = _arrays(df, m, "dev")
        ct, st, _ = _arrays(df, m, "test")
        oracle = M.select_threshold(ct, st, 0.90, 1.0, 0.0)["dev_review_rate"]
        for margin in (0.0, 0.025, 0.05):
            for size, replace in ((100, False), (len(cd), True)):
                dev_, qual = [], []
                for _ in range(n_rep):
                    i = rng.choice(len(cd), size=size, replace=replace)
                    sel = M.select_threshold(cd[i], sd[i], 0.90, 1.0, margin)
                    q = M.qualify(ct, st, sel["tau"], 0.90, 1.0, cfg["analysis"]["confidence_level"])
                    rr = q["test_review_rate"] if sel["feasible"] else 1.0
                    dev_.append(rr - oracle)
                    qual.append(q["qualified"] if sel["feasible"] else True)
                dev_ = np.array(dev_)
                rows.append({"model_key": m, "margin": margin, "dev_size": size, "oracle_review_rate": oracle,
                             "bias": dev_.mean(), "mean_abs_dev": np.abs(dev_).mean(), "sd": dev_.std(),
                             "share_qualified": float(np.mean(qual))})
    return pd.DataFrame(rows)
