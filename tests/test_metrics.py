import numpy as np
import pytest

from ready_eval import metrics as M


def test_wilson_and_cp():
    lo, hi = M.wilson_ci(90, 100)
    assert 0.82 < lo < 0.84 and 0.94 < hi < 0.95
    assert M.clopper_pearson_lower(100, 100, 0.95) == pytest.approx(0.05 ** (1 / 100))
    assert M.clopper_pearson_lower(0, 10) == 0.0


def test_ece_perfect_and_bad():
    c = np.array([1, 1, 0, 0])
    assert M.ece(c, np.array([1.0, 1.0, 0.0, 0.0]), 10) == pytest.approx(0.0)
    assert M.ece(c, np.array([0.9, 0.9, 0.9, 0.9]), 10) == pytest.approx(0.4)


def test_brier():
    assert M.brier([1, 0], [1.0, 0.0]) == 0.0
    assert M.brier([1, 0], [0.5, 0.5]) == 0.25


def test_auroc_error_detection():
    c = np.array([1, 1, 1, 0, 0])
    assert M.auroc_error_detection(c, np.array([0.9, 0.8, 0.7, 0.2, 0.1])) == 1.0
    assert M.auroc_error_detection(c, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 0.0
    assert np.isnan(M.auroc_error_detection(np.ones(5), np.linspace(0, 1, 5)))


def test_risk_coverage_ties_grouped():
    cov, risk = M.risk_coverage([1, 0, 1, 1], [0.9, 0.9, 0.5, 0.5])
    assert cov.tolist() == [0.5, 1.0]
    assert risk.tolist() == [0.5, 0.25]
    assert M.aurc([1, 0, 1, 1], [0.9, 0.9, 0.5, 0.5]) == pytest.approx(0.5 * 0.5 + 0.5 * 0.25)


def test_reliability_after_review():
    c = np.array([1, 0, 1, 0])
    r = M.reliability_after_review(c, np.array([False, True, False, False]), a_h=1.0)
    assert r["review_rate"] == 0.25 and r["reliability"] == 0.75
    assert r["reviewed_ai_accuracy"] == 0.0 and r["errors_missed"] == 1
    r = M.reliability_after_review(c, np.ones(4, bool), a_h=0.9)
    assert r["reliability"] == pytest.approx(0.9)


def test_select_threshold_uses_min_review_and_margin():
    c = np.array([1, 1, 1, 1, 0, 0])
    s = np.array([0.9, 0.9, 0.8, 0.8, 0.3, 0.2])
    sel = M.select_threshold(c, s, target=0.8, a_h=1.0, margin=0.0)
    # 0.3未満をレビュー → 1件の誤答が残る（5/6=0.833 >= 0.8）
    assert sel["feasible"] and sel["dev_review_rate"] == pytest.approx(1 / 6)
    sel2 = M.select_threshold(c, s, target=0.8, a_h=1.0, margin=0.05)
    assert sel2["dev_review_rate"] == pytest.approx(2 / 6)


def test_select_threshold_infeasible_when_ah_below_target():
    c = np.array([1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6])
    sel = M.select_threshold(c, s, target=0.95, a_h=0.9, margin=0.0)
    assert not sel["feasible"]
    assert M.qualify(c, s, sel["tau"], 0.95, 0.9)["status"] == "dev_infeasible"


def test_qualify_lcb_below_point():
    rng = np.random.default_rng(0)
    c = (rng.random(200) < 0.9).astype(int)
    s = rng.random(200)
    q = M.qualify(c, s, -np.inf, target=0.85, a_h=1.0)
    assert q["test_review_rate"] == 0.0
    assert q["lcb"] < q["test_reliability"]
    q_all = M.qualify(c, s, np.inf, target=0.85, a_h=1.0)
    assert q_all["lcb"] == 1.0 and q_all["qualified"]


def test_mcnemar_and_holm():
    r = M.mcnemar_exact([1, 1, 1, 0], [0, 0, 1, 0])
    assert r["only_first_correct"] == 2 and r["only_second_correct"] == 0
    assert M.mcnemar_exact([1, 0], [1, 0])["p_value"] == 1.0
    assert M.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_random_policy_endpoints():
    c = np.array([1, 0, 1, 0])
    out = M.random_policy_curve(c, [0.0, 1.0], a_h=1.0, seeds=3)
    assert out[0]["reliability"] == 0.5 and out[1]["reliability"] == 1.0


def _brute_select(c, s, target, a_h, margin):
    best = None
    for tau in M.candidate_thresholds(s):
        r = M.reliability_after_review(c, M.threshold_review_mask(s, tau), a_h)
        if r["reliability"] >= target + margin - 1e-12 and (best is None or r["review_rate"] < best[1] - 1e-12):
            best = (tau, r["review_rate"])
    return best


def test_vectorized_select_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(5, 60))
        c = (rng.random(n) < rng.uniform(0.5, 0.95)).astype(int)
        s = np.round(rng.random(n), 1)
        Y, a_h, d = rng.choice([0.8, 0.9, 0.95]), rng.choice([0.9, 1.0]), rng.choice([0.0, 0.05])
        b = _brute_select(c, s, Y, a_h, d)
        v = M.select_threshold(c, s, Y, a_h, d)
        if b is None:
            assert not v["feasible"]
        else:
            assert v["tau"] == b[0] and v["dev_review_rate"] == pytest.approx(b[1])


def test_lcb_selection_is_conservative():
    rng = np.random.default_rng(5)
    c = (rng.random(500) < 0.85).astype(int)
    s = np.clip(c * 0.3 + rng.random(500) * 0.7, 0, 1).round(2)
    lcb = M.select_threshold_lcb(c, s, 0.90, 1.0)
    pt = M.select_threshold(c, s, 0.90, 1.0, 0.0)
    assert lcb["feasible"] and lcb["dev_review_rate"] >= pt["dev_review_rate"]
    q = M.qualify(c, s, lcb["tau"], 0.90, 1.0)
    assert q["qualified"]  # 同じデータに当てれば下限は必ず目標以上


def test_ece_bin_edges_exact():
    # 0.7 は [0.7, 0.8) のビンに入る（np.linspace の端点誤差で一つ下のビンに入らない）
    assert M.ece([1, 0], [0.7, 0.65], 10) == pytest.approx(0.475)
    rows = M.calibration_table([1, 1, 1], [0.3, 0.6, 0.7], 10)
    assert [r["n"] for r in rows if r["n"]] == [1, 1, 1]
    assert [r["bin"] for r in rows if r["n"]] == [3, 6, 7]


def test_human_error_selection_matches_ah1_when_perfect():
    from ready_eval.robustness import _select_pc_pw
    rng = np.random.default_rng(0)
    s = np.round(rng.uniform(0, 1, 300), 2)
    c = (rng.uniform(0, 1, 300) < s).astype(int)
    for Y in (0.85, 0.9):
        sel = M.select_threshold(c, s, Y, 1.0, 0.05)
        assert _select_pc_pw(c, s, Y, 0.05, 1.0, 1.0) == sel["tau"]
    # 誤答を見逃すほど、同じ目標に必要なレビューは増える（しきい値は下がらない）
    t1 = _select_pc_pw(c, s, 0.85, 0.05, 1.0, 1.0)
    t2 = _select_pc_pw(c, s, 0.85, 0.05, 1.0, 0.8)
    assert np.isnan(t2) or t2 >= t1
