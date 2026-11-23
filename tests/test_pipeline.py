import copy

import numpy as np
import pandas as pd
import pytest

from ready_eval import analysis, data
from ready_eval.config import load_config
from ready_eval.runner import collect_predictions, run_models, simulate_predictions

LABELS = [f"label_{i}" for i in range(77)]


def _cases(n_dev=10, n_test=20):
    rows = []
    for split, n in (("dev", n_dev), ("test", n_test)):
        for i in range(n):
            rows.append({"case_id": f"{split}-{i:05d}", "split": split, "text": f"msg {split} {i}",
                         "label_id": i % 77, "label": LABELS[i % 77]})
    return pd.DataFrame(rows)


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["run"]["concurrency"] = 2
    c["analysis"]["bootstrap_n"] = 50
    c["analysis"]["random_review_seeds"] = 3
    return c


def test_stratified_sample_balanced():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(77 * 5)], "label": np.repeat(range(77), 5)})
    out = data.stratified_sample(df, 200, np.random.default_rng(0))
    counts = out["label"].value_counts()
    assert len(out) == 200 and counts.min() == 2 and counts.max() == 3 and out.index.is_unique
    out2 = data.stratified_sample(df, 200, np.random.default_rng(0))
    assert out.index.tolist() == out2.index.tolist()  # シード固定で再現


def test_parse_label_names():
    text = "names:\n" + "\n".join(f"          '{i}': name_{i}" for i in range(77))
    assert data.parse_label_names(text)[76] == "name_76"


class FakeCaller:
    def __init__(self):
        self.calls = 0

    def __call__(self, m, case, labels):
        self.calls += 1
        ok = case["label_id"] % 3 != 0
        return {"model_key": m["key"], "model_id": m["model_id"], "case_id": case["case_id"],
                "status": "ok" if case["label_id"] != 5 else "api_error",
                "predicted_id": case["label_id"] if ok else 0, "confidence": 0.9 if ok else 0.4,
                "cost_usd": 0.01, "usage": {"input_tokens": 100, "output_tokens": 10}}


def test_runner_caches_and_respects_budget(tmp_path, cfg):
    cases = _cases()
    m = {"key": "fake", "provider": "anthropic", "model_id": "fake-1", "price_in": 1.0, "price_out": 1.0,
         "max_tokens": 10, "effort": "low"}
    fake = FakeCaller()
    cfg["run"]["budget_usd"] = 100
    st = run_models(cfg, cases, LABELS, [m], tmp_path, lambda _m: fake, log=lambda *_: None)
    assert st["done"] == 30 and fake.calls == 30
    st2 = run_models(cfg, cases, LABELS, [m], tmp_path, lambda _m: fake, log=lambda *_: None)
    assert st2["cached"] == 30 and fake.calls == 30  # 再実行しない
    df = collect_predictions(tmp_path, [m], cases)
    assert (df.status == "api_error").sum() == 2  # label_id=5 は dev と test に1件ずつ
    assert (df.loc[df.status == "api_error", "correct"] == 0).all()  # 失敗は誤答として分母に残る
    assert (df.loc[df.status == "api_error", "score"] == 0.0).all()

    # 予算ぎりぎりでは新規実行しない
    m2 = dict(m, key="fake2", model_id="fake-2")
    cfg["run"]["budget_usd"] = 0.301  # 既存の支出 0.30 + 1件分の予約（約0.0019）で超える
    fake2 = FakeCaller()
    st3 = run_models(cfg, cases, LABELS, [m2], tmp_path, lambda _m: fake2, log=lambda *_: None)
    assert st3["stopped"] and fake2.calls == 0
    df2 = collect_predictions(tmp_path, [m2], cases)
    assert (df2.status == "not_run").all()
    with pytest.raises(RuntimeError):
        analysis.run_all(cfg, df2)


def test_simulation_end_to_end(tmp_path, cfg):
    cases = _cases(30, 60)
    df = simulate_predictions(cfg, cases)
    tables = analysis.run_all(cfg, df)
    assert set(tables["metrics"].model_key) == {"sim-A", "sim-B", "sim-C"}
    r = tables["ready_qualification"]
    assert len(r) == 3 * 3 * 2 * 4
    assert ((r.test_reliability.dropna() >= 0) & (r.test_reliability.dropna() <= 1)).all()
    ps = tables["policy_summary"]
    assert (ps.loc[ps.policy == "B_full_review", "reliability_after_review"] == cfg["analysis"]["primary_human_success"]).all()
    assert {"A_ai_only", "B_full_review", "C_random", "D_confidence"} <= set(tables["policy_summary"].policy)
    assert len(tables["pairwise_tests"]) == 3


def test_extend_splits_disjoint(tmp_path, cfg):
    raw = data.RAW_DIR
    base = tmp_path / "v1.csv"
    c1 = copy.deepcopy(cfg)
    data.prepare_splits(c1, raw, base)
    c2 = copy.deepcopy(cfg)
    c2["seed"] = 1
    c2["data"].update({"extend_from": str(base), "n_dev_new": 154, "n_test": 154})
    v2 = data.prepare_splits(c2, raw, tmp_path / "v2.csv")
    v1 = pd.read_csv(base)
    assert set(v1[v1.split == "dev"].case_id) <= set(v2[v2.split == "dev"].case_id)
    assert not set(v1[v1.split == "test"].case_id) & set(v2[v2.split == "test"].case_id)
    assert (v2.split == "dev").sum() == 100 + 154 and (v2.split == "test").sum() == 154


def test_hypotheses_on_simulation(cfg):
    from ready_eval.hypotheses import evaluate
    c = copy.deepcopy(load_config(data.resolve("config_v2.yaml")))
    c["analysis"]["bootstrap_n"] = 50
    c["analysis"]["random_review_seeds"] = 10
    df = simulate_predictions(c, _cases(100, 150))
    res = evaluate(c, df, analysis.run_all(c, df))
    assert res.id.tolist() == ["H1", "H2", "H3a", "H3b", "H4a", "H4b", "H5", "H6"]
    assert set(res.verdict) <= {"支持", "不支持", "判定保留"}


def test_cache_key_stable_for_default_and_distinct_for_variants():
    from ready_eval.runner import cache_key
    m = {"key": "x", "provider": "anthropic", "model_id": "m", "max_tokens": 10, "effort": "low"}
    case = {"case_id": "test-00001", "text": "hello"}
    k0 = cache_key(m, case)
    assert cache_key(dict(m, task="classify", sample=0), case) == k0  # v1・v2 のキーを変えない
    assert cache_key(dict(m, sample=1), case) != k0
    assert cache_key(dict(m, sample=1), case) != cache_key(dict(m, sample=2), case)
    kv = cache_key(dict(m, task="verify"), dict(case, proposed_label="a"))
    assert kv != k0 and kv != cache_key(dict(m, task="verify"), dict(case, proposed_label="b"))


def test_parse_verify():
    from ready_eval.prompt import parse_verify
    assert parse_verify('{"probability_correct": 0.7}') == ("ok", 0.7, "")
    assert parse_verify('{"probability_correct": 70}')[1] == 0.7
    assert parse_verify("nope")[0] == "invalid_output"
