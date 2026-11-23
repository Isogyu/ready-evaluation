"""評価指標・レビュー方針・READY型の適格判定・統計検定。

入力は1モデル分の配列：
  correct: 0/1（失敗は0）
  score:   レビュー振り分けに使う確信度（失敗は0）
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------ 基本
def wilson_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - level) / 2)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (c - h, c + h)


def clopper_pearson_lower(k: int, n: int, level: float = 0.95) -> float:
    """片側の Clopper–Pearson 下限（level は片側の信頼水準）。"""
    if n == 0:
        return np.nan
    if k == 0:
        return 0.0
    return float(stats.beta.ppf(1 - level, k, n - k + 1))


# ------------------------------------------------------------------ 確信度
def brier(correct, score) -> float:
    c, s = np.asarray(correct, float), np.asarray(score, float)
    return float(np.mean((s - c) ** 2))


def calibration_table(correct, score, n_bins: int = 10) -> list[dict]:
    """等幅ビン。区間は [lo, hi)、最後のビンのみ [lo, 1.0]。"""
    c, s = np.asarray(correct, float), np.asarray(score, float)
    # np.linspace では 0.3 が 0.30000000000000004 になり、境界ちょうどの値が一つ下のビンに入る。整数比で作る。
    edges = np.arange(n_bins + 1) / n_bins
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append({
            "bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(mask.sum()),
            "mean_conf": float(s[mask].mean()) if mask.any() else np.nan,
            "accuracy": float(c[mask].mean()) if mask.any() else np.nan,
        })
    return rows


def ece(correct, score, n_bins: int = 10) -> float:
    n = len(correct)
    return float(sum(r["n"] / n * abs(r["accuracy"] - r["mean_conf"])
                     for r in calibration_table(correct, score, n_bins) if r["n"] > 0))


def auroc_error_detection(correct, score) -> float:
    """確信度の低さで誤答を見分けられるか。誤答=正例、スコア=1-確信度。片方のクラスしかなければ NaN。"""
    c = np.asarray(correct, int)
    if c.min() == c.max():
        return np.nan
    return float(roc_auc_score(1 - c, 1 - np.asarray(score, float)))


def risk_coverage(correct, score) -> tuple[np.ndarray, np.ndarray]:
    """しきい値ごと（同値は同時に採用）のカバレッジと採用分の誤答率。"""
    c, s = np.asarray(correct, float), np.asarray(score, float)
    order = np.argsort(-s, kind="stable")
    c, s = c[order], s[order]
    n = len(c)
    last_of_group = np.r_[s[1:] != s[:-1], True]
    k = np.arange(1, n + 1)[last_of_group]
    errs = np.cumsum(1 - c)[last_of_group]
    return k / n, errs / k


def aurc(correct, score) -> float:
    """リスク・カバレッジ曲線の下側面積（階段近似。低いほど良い）。"""
    cov, risk = risk_coverage(correct, score)
    widths = np.diff(np.r_[0.0, cov])
    return float(np.sum(widths * risk))


# ------------------------------------------------------------------ レビュー方針
def reliability_after_review(correct, review_mask, a_h: float) -> dict:
    """レビュー対象は確率 a_h で正しく処理される（理想化した仮定）。期待値で計算する。"""
    c = np.asarray(correct, float)
    r = np.asarray(review_mask, bool)
    n = len(c)
    n_rev = int(r.sum())
    acc_part = c[~r].sum()
    return {
        "review_rate": n_rev / n,
        "reliability": (acc_part + a_h * n_rev) / n,
        "accepted_accuracy": float(c[~r].mean()) if n_rev < n else np.nan,
        "reviewed_ai_accuracy": float(c[r].mean()) if n_rev > 0 else np.nan,  # レビュー対象のAI正答率
        "errors_caught": int((1 - c[r]).sum()),
        "errors_missed": int((1 - c[~r]).sum()),
    }


def threshold_review_mask(score, tau: float) -> np.ndarray:
    """確信度が tau 未満ならレビュー（tau=inf で全件レビュー、tau=-inf でレビューなし）。"""
    return np.asarray(score, float) < tau


def candidate_thresholds(score) -> np.ndarray:
    s = np.unique(np.asarray(score, float))
    return np.r_[-np.inf, s, np.inf]


def confidence_policy_curve(correct, score, a_h: float) -> list[dict]:
    out = []
    for tau in candidate_thresholds(score):
        r = reliability_after_review(correct, threshold_review_mask(score, tau), a_h)
        out.append({"tau": tau, **r})
    return out


def random_policy_curve(correct, rates, a_h: float, seeds: int, seed0: int = 0) -> list[dict]:
    c = np.asarray(correct, float)
    n = len(c)
    out = []
    for rate in rates:
        k = int(round(rate * n))
        vals = []
        for s in range(seeds):
            rng = np.random.default_rng(seed0 + s)
            mask = np.zeros(n, bool)
            mask[rng.choice(n, size=k, replace=False)] = True
            vals.append(reliability_after_review(c, mask, a_h)["reliability"])
        vals = np.array(vals)
        out.append({"review_rate": k / n, "reliability": float(vals.mean()),
                    "rel_p2_5": float(np.percentile(vals, 2.5)), "rel_p97_5": float(np.percentile(vals, 97.5))})
    return out


# ------------------------------------------------------------------ READY型の選定と検証
def _threshold_table(correct, score, a_h: float):
    """候補しきい値ごとのレビュー件数・採用件数・採用分の正解件数・信頼性を一度に計算する（ベクトル化）。
    候補は -inf（レビューなし）、データに現れた確信度の値、+inf（全件レビュー）。確信度が tau 未満ならレビュー。"""
    c = np.asarray(correct, float)
    s = np.asarray(score, float)
    n = len(c)
    order = np.argsort(s, kind="stable")
    s_sorted, c_sorted = s[order], c[order]
    cands = np.r_[-np.inf, np.unique(s), np.inf]
    n_rev = np.searchsorted(s_sorted, cands, side="left")
    c_cum = np.r_[0.0, np.cumsum(c_sorted)]
    k_rev = c_cum[n_rev]
    k_acc = c.sum() - k_rev
    n_acc = n - n_rev
    rel = (k_acc + a_h * n_rev) / n
    return cands, n_rev, n_acc, k_acc, rel


def _pick(cands, n_rev, rel, feasible, n):
    if not feasible.any():
        return {"tau": np.nan, "dev_review_rate": np.nan, "dev_reliability": np.nan, "feasible": False}
    j = int(np.argmax(feasible))  # しきい値が小さいほどレビューが少ないので、最初に条件を満たす候補
    return {"tau": float(cands[j]), "dev_review_rate": n_rev[j] / n, "dev_reliability": float(rel[j]), "feasible": True}


def select_threshold(correct_dev, score_dev, target: float, a_h: float, margin: float) -> dict:
    """開発データで R_dev >= target + margin を満たす候補のうち、レビュー率が最小のしきい値を選ぶ。

    論文 Eq.45 の簡略版（コストが一定なので、コスト最小 = レビュー率最小）。
    """
    cands, n_rev, _, _, rel = _threshold_table(correct_dev, score_dev, a_h)
    return _pick(cands, n_rev, rel, rel >= target + margin - 1e-12, len(np.asarray(correct_dev)))


def cp_lower_vec(k: np.ndarray, n: np.ndarray, level: float) -> np.ndarray:
    k, n = np.asarray(k, float), np.asarray(n, float)
    out = np.zeros_like(k)
    ok = (n > 0) & (k > 0)
    out[ok] = stats.beta.ppf(1 - level, k[ok], n[ok] - k[ok] + 1)
    return out


def select_threshold_lcb(correct_dev, score_dev, target: float, a_h: float, level: float = 0.95) -> dict:
    """下限法：開発データ上の片側下限（採用分の Clopper–Pearson 下限＋レビュー分 a_h）が target 以上になる
    候補のうち、レビュー率が最小のしきい値を選ぶ。固定の安全余裕の代わりに、件数に応じた余裕が自動で付く。"""
    cands, n_rev, n_acc, k_acc, rel = _threshold_table(correct_dev, score_dev, a_h)
    n = len(np.asarray(correct_dev))
    lcb = (n_acc / n) * cp_lower_vec(k_acc, n_acc, level) + (n_rev / n) * a_h
    out = _pick(cands, n_rev, rel, lcb >= target - 1e-12, n)
    return out


def qualify(correct_test, score_test, tau: float, target: float, a_h: float, level: float = 0.95) -> dict:
    """固定したしきい値を最終評価データに適用し、片側下限で判定する（論文 §5.5 の簡略版）。

    下限 = (採用件数/N)×採用分正答率の Clopper–Pearson 下限 + (レビュー件数/N)×a_h。
    a_h には信頼区間を付けない（仮定値として固定）。
    """
    c = np.asarray(correct_test, int)
    n = len(c)
    if np.isnan(tau):
        return {"test_review_rate": np.nan, "test_reliability": np.nan, "lcb": np.nan,
                "point_meets_target": False, "qualified": False, "n_accepted": 0, "status": "dev_infeasible"}
    mask = threshold_review_mask(score_test, tau)
    r = reliability_after_review(c, mask, a_h)
    n_acc = int((~mask).sum())
    k_acc = int(c[~mask].sum())
    lcb_acc = clopper_pearson_lower(k_acc, n_acc, level) if n_acc > 0 else 0.0
    lcb = (n_acc / n) * lcb_acc + ((n - n_acc) / n) * a_h
    qualified = bool(lcb >= target - 1e-12)
    status = "qualified" if qualified else "not_qualified"
    return {"test_review_rate": r["review_rate"], "test_reliability": r["reliability"], "lcb": lcb,
            "point_meets_target": bool(r["reliability"] >= target - 1e-12), "qualified": qualified,
            "n_accepted": n_acc, "accepted_accuracy": r["accepted_accuracy"],
            "reviewed_ai_accuracy": r["reviewed_ai_accuracy"], "status": status}


def bootstrap_fixed_tau(correct, score, tau: float, a_h: float, n_boot: int, seed: int) -> dict:
    """しきい値を固定したまま最終評価データをケース単位で再標本化し、レビュー率と信頼性の95%区間を出す。"""
    c, s = np.asarray(correct, float), np.asarray(score, float)
    n = len(c)
    rng = np.random.default_rng(seed)
    rr, rel = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        r = reliability_after_review(c[i], threshold_review_mask(s[i], tau), a_h)
        rr[b], rel[b] = r["review_rate"], r["reliability"]
    return {"review_rate_lo": float(np.percentile(rr, 2.5)), "review_rate_hi": float(np.percentile(rr, 97.5)),
            "reliability_lo": float(np.percentile(rel, 2.5)), "reliability_hi": float(np.percentile(rel, 97.5))}


# ------------------------------------------------------------------ モデル間比較
def mcnemar_exact(c1, c2) -> dict:
    """同じ問題での正誤の対応ありデータ。不一致ペアに対する正確二項検定（両側）。"""
    c1, c2 = np.asarray(c1, int), np.asarray(c2, int)
    b = int(((c1 == 1) & (c2 == 0)).sum())
    c = int(((c1 == 0) & (c2 == 1)).sum())
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"only_first_correct": b, "only_second_correct": c, "p_value": p}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_bootstrap_diff(c1, c2, n_boot: int, seed: int) -> tuple[float, float, float]:
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    n = len(c1)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        d[b] = c1[i].mean() - c2[i].mean()
    return float(c1.mean() - c2.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def bootstrap_auroc(correct, score, n_boot: int, seed: int) -> tuple[float, float]:
    c, s = np.asarray(correct, int), np.asarray(score, float)
    n = len(c)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = auroc_error_detection(c[i], s[i])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return (np.nan, np.nan)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
