"""コマンドライン：python -m ready_eval <command>

  prepare              データ取得と分割（data/splits.csv）
  check                APIキーとモデルIDの確認（Models API のみ。課金なし）
  run [--pilot N]      モデル実行（費用が発生する）。--pilot N は開発用データの先頭N件だけ
  analyze              results/ の実測結果から表とグラフを作る
  simulate             APIを使わずに仮想モデルで全工程を通す（results/simulation/、charts/simulation/）
  robustness           v5 の結論の頑健さ（a_h 別・人間の見逃し・開発データの入れ替え）。--config config_v5.yaml で使う
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version

from dotenv import load_dotenv

from . import analysis, charts, data
from .config import ROOT, load_config, resolve
from .prompt import PROMPT_VERSION
from .runner import collect_predictions, run_models, simulate_predictions, spent_usd


def _models(cfg, keys):
    ms = cfg["models"]
    return [m for m in ms if m["key"] in keys] if keys else ms


def _caller_factory(cfg):
    from .clients import AnthropicCaller, OpenAICaller
    cache = {}

    def f(m):
        if m["provider"] not in cache:
            cls = {"anthropic": AnthropicCaller, "openai": OpenAICaller}[m["provider"]]
            cache[m["provider"]] = cls(cfg["run"]["timeout_s"], cfg["run"]["max_retries"])
        return cache[m["provider"]]
    return f


def cmd_prepare(cfg, args):
    s = data.prepare_splits(cfg)
    print(s.groupby("split").agg(n=("case_id", "size"), classes=("label_id", "nunique")))


def cmd_check(cfg, args):
    from .clients import AnthropicCaller
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY が見つかりません（.env を確認してください）。")
        sys.exit(1)
    available = AnthropicCaller(30, 1).list_models()
    for m in cfg["models"]:
        if m["provider"] == "anthropic":
            print(f"{m['model_id']:<24} {'利用可' if m['model_id'] in available else '見つからない'}")
    print("OPENAI_API_KEY:", "あり" if os.environ.get("OPENAI_API_KEY") else "なし")
    (resolve(cfg["run"]["results_dir"])).mkdir(parents=True, exist_ok=True)
    (resolve(cfg["run"]["results_dir"]) / "available_models.json").write_text(
        json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(), "models": available}, indent=1))


def _paths(cfg):
    results_dir = resolve(cfg["run"]["results_dir"])
    raw_root = resolve(cfg["run"].get("raw_dir", f"{cfg['run']['results_dir']}/raw"))
    charts_dir = resolve(cfg["run"].get("charts_dir", "charts"))
    splits_path = cfg["data"].get("splits_path", "data/splits.csv")
    return results_dir, raw_root, charts_dir, splits_path


def _doc_hashes(cfg):
    import hashlib
    out = {}
    for d in cfg.get("preregistered_docs", []):
        p = resolve(d)
        if p.exists():
            out[d] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _write_meta(cfg, results_dir, models, extra):
    results_dir.mkdir(parents=True, exist_ok=True)
    meta_path = results_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"runs": []}
    meta["runs"].append({
        "finished_at": datetime.now(timezone.utc).isoformat(), "prompt_version": PROMPT_VERSION,
        "models": models, "budget_usd": cfg["run"]["budget_usd"], "seed": cfg["seed"],
        "python": platform.python_version(), "anthropic_sdk": version("anthropic"),
        "preregistered_doc_sha256": _doc_hashes(cfg), **extra,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def cmd_run(cfg, args):
    results_dir, raw_root, _, splits_path = _paths(cfg)
    splits = data.load_splits(splits_path)
    labels = data.load_labels()
    cases = splits[splits.split == "dev"].head(args.pilot) if args.pilot else splits
    models = _models(cfg, args.models)
    state = run_models(cfg, cases, labels, models, raw_root, _caller_factory(cfg), args.retry_errors)
    _write_meta(cfg, results_dir, models, {"mode": "pilot" if args.pilot else "full",
                                           "n_cases": len(cases), "new_calls": state["done"],
                                           "stopped_by_budget": state["stopped"], "spent_usd": state["spent"]})
    df = collect_predictions(raw_root, models, cases)
    done = df[df.status != "not_run"]
    print(done.groupby("model_key").agg(
        n=("case_id", "size"), acc=("correct", "mean"), cost_usd=("cost_usd", "sum"),
        in_tok=("input_tokens", "mean"), out_tok=("output_tokens", "mean"),
        fails=("status", lambda s: int((s != "ok").sum()))))


def _analyze(cfg, df, results_dir, charts_dir, mode_note):
    tables = analysis.run_all(cfg, df)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "predictions.csv", index=False)
    tables["metrics"].to_csv(results_dir / "metrics.csv", index=False)
    tables["policy_summary"].to_csv(results_dir / "cost_analysis.csv", index=False)
    for name in ("calibration_bins", "policy_curves", "ready_qualification", "cost_sensitivity", "pairwise_tests",
                 "selection_bootstrap", "selection_bootstrap_diffs"):
        tables[name].to_csv(results_dir / f"{name}.csv", index=False)
    paths = charts.make_all(tables, cfg, charts_dir, mode_note, analysis.review_cost_per_case(cfg))
    print(f"表: {results_dir}\nグラフ: {charts_dir}（{len(paths)}枚）")
    return tables


def cmd_analyze(cfg, args):
    results_dir, raw_root, charts_dir, splits_path = _paths(cfg)
    splits = data.load_splits(splits_path)
    df = collect_predictions(raw_root, cfg["models"], splits)
    tables = _analyze(cfg, df, results_dir, charts_dir, cfg["run"].get("chart_note", "（実測：API実行結果）"))
    if cfg.get("hypotheses"):
        from .hypotheses import evaluate
        res = evaluate(cfg, df, tables)
        res.to_csv(results_dir / "hypotheses.csv", index=False)
        print(res[["id", "verdict", "summary"]].to_string(index=False))
        from .hypotheses import exploratory_margin
        exploratory_margin(cfg, df).to_csv(results_dir / "exploratory_margin.csv", index=False)
    print(f"累積API費用（全実験）: ${spent_usd(raw_root):.4f}")


def cmd_simulate(cfg, args):
    _, _, charts_dir, splits_path = _paths(cfg)
    splits = data.load_splits(splits_path)
    df = simulate_predictions(cfg, splits)
    sim_dir = resolve(cfg["run"]["sim_results_dir"])
    tables = _analyze(cfg, df, sim_dir, charts_dir / "simulation", "【シミュレーション値：実測ではない】")
    if cfg.get("hypotheses"):
        from .hypotheses import evaluate
        res = evaluate(cfg, df, tables)
        res.to_csv(sim_dir / "hypotheses.csv", index=False)
        print(res[["id", "verdict", "summary"]].to_string(index=False))


def _signals_analyze(cfg, frame, var, results_dir, charts_dir, note):
    from . import signals as SG
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    tables = analysis.run_all(cfg, frame)
    frame.to_csv(results_dir / "predictions_signals.csv", index=False)
    for name, t in tables.items():
        if name == "pairwise_tests":  # 同じ予測どうしの比較は意味がないので保存しない
            continue
        t.to_csv(results_dir / f"{name}.csv", index=False)
    var.to_csv(results_dir / "sample_variability.csv", index=False)
    msumm, mdiff = SG.method_bootstrap(cfg, frame)
    msumm.to_csv(results_dir / "method_bootstrap.csv", index=False)
    mdiff.to_csv(results_dir / "method_bootstrap_diffs.csv", index=False)
    SG.point_methods(cfg, frame).to_csv(results_dir / "method_point.csv", index=False)
    res = SG.evaluate(cfg, frame, tables, msumm, mdiff)
    res.to_csv(results_dir / "hypotheses.csv", index=False)
    models = list(dict.fromkeys(frame.base_model))
    charts.v3_auroc(tables["metrics"], models, charts_dir / "graph7_signal_auroc.png", note)
    charts.v3_required_review(tables["selection_bootstrap"], tables["ready_qualification"], models,
                              charts_dir / "graph8_signal_required_review.png", note)
    charts.v3_curves(tables["policy_curves"], models, charts_dir / "graph9_signal_curves.png", note)
    charts.v3_methods(msumm, models, charts_dir / "graph10_selection_methods.png", note)
    print(res[["id", "verdict", "summary"]].to_string(index=False))


def cmd_signals_run(cfg, args):
    from . import signals as SG
    results_dir, raw_root, _, splits_path = _paths(cfg)
    splits = data.load_splits(splits_path)
    cases = splits[splits.split == "dev"].head(args.pilot) if args.pilot else splits
    st = SG.run_signals(cfg, cases, data.load_labels(), raw_root, _caller_factory(cfg))
    _write_meta(cfg, results_dir, cfg["models"], {"mode": "signals-pilot" if args.pilot else "signals-full",
                                                   "n_cases": len(cases), "stopped_by_budget": st["stopped"],
                                                   "spent_usd_total": spent_usd(raw_root)})
    print(f"累積API費用（全実験）: ${spent_usd(raw_root):.4f}")


def cmd_signals_analyze(cfg, args):
    from . import signals as SG
    results_dir, raw_root, charts_dir, splits_path = _paths(cfg)
    splits = data.load_splits(splits_path)
    frame, var = SG.build_signal_frame(cfg, splits, raw_root)
    _signals_analyze(cfg, frame, var, results_dir, charts_dir, cfg["run"].get("chart_note", ""))
    print(f"累積API費用（全実験）: ${spent_usd(raw_root):.4f}")


def cmd_signals_simulate(cfg, args):
    from . import signals as SG
    _, _, charts_dir, splits_path = _paths(cfg)
    frame, var = SG.simulate_signal_frame(cfg, data.load_splits(splits_path))
    _signals_analyze(cfg, frame, var, resolve(cfg["run"]["sim_results_dir"]), charts_dir / "simulation",
                     "【シミュレーション値：実測ではない】")


V4_ORDER = ["self", "xagree", "xver_opus-5-5", "xver_sonnet-5-5", "xver_haiku-5-5", "combx"]
V4_LABEL = {"self": "自己申告", "xagree": "他モデル一致", "xver_opus-5-5": "Opus が検証",
            "xver_sonnet-5-5": "Sonnet が検証", "xver_haiku-5-5": "Haiku が検証", "combx": "組み合わせ"}


def _cross_analyze(cfg, frame, results_dir, charts_dir, note):
    from . import signals as SG
    from . import signals_v4 as V4
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    tables = analysis.run_all(cfg, frame)
    frame.to_csv(results_dir / "predictions_signals.csv", index=False)
    for name, t in tables.items():
        if name != "pairwise_tests":  # 同じ予測どうしの比較は意味がないので保存しない
            t.to_csv(results_dir / f"{name}.csv", index=False)
    sub = frame[frame.signal.isin(["self", "combx"])]
    msumm, mdiff = SG.method_bootstrap(cfg, sub)
    msumm.to_csv(results_dir / "method_bootstrap.csv", index=False)
    mdiff.to_csv(results_dir / "method_bootstrap_diffs.csv", index=False)
    SG.point_methods(cfg, frame).to_csv(results_dir / "method_point.csv", index=False)
    res = V4.evaluate(cfg, frame, tables)
    res.to_csv(results_dir / "hypotheses.csv", index=False)
    models = list(dict.fromkeys(frame.base_model))
    charts.v4_panels(tables, models, V4_ORDER, V4_LABEL, charts_dir, note)
    charts.v3_methods(msumm, models, charts_dir / "graph14_cross_selection_methods.png", note,
                      title="グラフ14　しきい値の選び方：δ=0.05 と下限法（自己申告と組み合わせ）",
                      sigs=("self", "combx"), labels=V4_LABEL)
    print(res[["id", "verdict", "summary"]].to_string(index=False))


def cmd_cross_run(cfg, args):
    from . import signals_v4 as V4
    results_dir, raw_root, _, splits_path = _paths(cfg)
    splits = data.load_splits(splits_path)
    cases = splits[splits.split == "dev"].head(args.pilot) if args.pilot else splits
    st = V4.run_cross(cfg, cases, data.load_labels(), raw_root, _caller_factory(cfg))
    _write_meta(cfg, results_dir, cfg["models"], {"mode": "cross-pilot" if args.pilot else "cross-full",
                                                   "n_cases": len(cases), "stopped_by_budget": st["stopped"],
                                                   "spent_usd_total": spent_usd(raw_root)})
    print(f"累積API費用（全実験）: ${spent_usd(raw_root):.4f}")


def cmd_cross_analyze(cfg, args):
    from . import signals_v4 as V4
    results_dir, raw_root, charts_dir, splits_path = _paths(cfg)
    frame = V4.build_frame(cfg, data.load_splits(splits_path), raw_root)
    _cross_analyze(cfg, frame, results_dir, charts_dir, cfg["run"].get("chart_note", ""))
    print(f"累積API費用（全実験）: ${spent_usd(raw_root):.4f}")


def cmd_cross_simulate(cfg, args):
    from . import signals_v4 as V4
    _, _, charts_dir, splits_path = _paths(cfg)
    frame = V4.simulate_frame(cfg, data.load_splits(splits_path))
    _cross_analyze(cfg, frame, resolve(cfg["run"]["sim_results_dir"]), charts_dir / "simulation",
                   "【シミュレーション値：実測ではない】")


def _pipelines_analyze(cfg, frame, gate, results_dir, charts_dir, note):
    from . import pipelines as PL
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    tables = analysis.run_all(cfg, frame)
    frame.to_csv(results_dir / "predictions_pipelines.csv", index=False)
    for name, t in tables.items():
        if name != "pairwise_tests":
            t.to_csv(results_dir / f"{name}.csv", index=False)
    point = PL.point_table(cfg, frame)
    summ, diffs = PL.cost_bootstrap(cfg, frame)
    sens = PL.sensitivity(cfg, point)
    point.to_csv(results_dir / "pipeline_point.csv", index=False)
    summ.to_csv(results_dir / "pipeline_bootstrap.csv", index=False)
    diffs.to_csv(results_dir / "pipeline_bootstrap_diffs.csv", index=False)
    sens.to_csv(results_dir / "pipeline_sensitivity.csv", index=False)
    (results_dir / "gate.json").write_text(json.dumps({"gate_haiku_confidence": gate}))
    res = PL.evaluate(diffs, summ, point)
    res.to_csv(results_dir / "hypotheses.csv", index=False)
    charts.v5_costs(summ, point, PL.PIPELINES, PL.PIPE_LABEL, charts_dir / "graph15_pipeline_costs.png", note)
    charts.v5_sensitivity(sens, PL.PIPELINES, PL.PIPE_LABEL, charts_dir / "graph16_pipeline_sensitivity.png", note)
    print(f"ゲート値 g = {gate}")
    print(point[["label", "ai_accuracy", "review_rate", "lcb", "qualified", "api_jpy_per_case", "total_jpy_per_case"]].round(3).to_string(index=False))
    print(res[["id", "verdict", "summary"]].to_string(index=False))


def cmd_pipelines_run(cfg, args):
    from . import pipelines as PL
    results_dir, raw_root, _, splits_path = _paths(cfg)
    splits = data.load_splits(splits_path)
    cases = splits[splits.split == "dev"].head(args.pilot) if args.pilot else splits
    st = PL.run(cfg, cases, data.load_labels(), raw_root, _caller_factory(cfg))
    _write_meta(cfg, results_dir, cfg["models"], {"mode": "pipelines-pilot" if args.pilot else "pipelines-full",
                                                   "n_cases": len(cases), "stopped_by_budget": st["stopped"],
                                                   "spent_usd_total": spent_usd(raw_root)})
    print(f"累積API費用（全実験）: ${spent_usd(raw_root):.4f}")


def cmd_pipelines_analyze(cfg, args):
    from . import pipelines as PL
    results_dir, raw_root, charts_dir, splits_path = _paths(cfg)
    frame, gate = PL.build_frame(cfg, data.load_splits(splits_path), raw_root)
    _pipelines_analyze(cfg, frame, gate, results_dir, charts_dir, cfg["run"].get("chart_note", ""))
    print(f"累積API費用（全実験）: ${spent_usd(raw_root):.4f}")


def cmd_pipelines_simulate(cfg, args):
    from . import pipelines as PL
    _, _, charts_dir, splits_path = _paths(cfg)
    frame, gate = PL.simulate(cfg, data.load_splits(splits_path))
    _pipelines_analyze(cfg, frame, gate, resolve(cfg["run"]["sim_results_dir"]), charts_dir / "simulation",
                       "【シミュレーション値：実測ではない】")


def cmd_power(cfg, args):
    from . import pipelines as PL
    from . import power as PW
    results_dir, raw_root, charts_dir, splits_path = _paths(cfg)
    frame, _ = PL.build_frame(cfg, data.load_splits(splits_path), raw_root)
    sd = PW.simulate_sd(cfg, frame, reps=args.reps)
    req = PW.required_n(sd)
    sd.to_csv(results_dir / "power_sd.csv", index=False)
    req.to_csv(results_dir / "power_required_n.csv", index=False)
    charts.power_chart(sd, req, PL.PIPE_LABEL, charts_dir / "graph17_required_n.png", cfg["run"].get("chart_note", ""))
    print(req.round(3).to_string(index=False))


def cmd_robustness(cfg, args):
    """v5 の結論の頑健さ（a_h 別・人間の見逃し・dev の入れ替え）。API なし、事前登録外。"""
    from . import pipelines as PL
    from . import robustness as RB
    from . import signals_v4 as V4
    results_dir, raw_root, _, splits_path = _paths(cfg)
    frame, _ = PL.build_frame(cfg, data.load_splits(splits_path), raw_root)
    cfg4 = load_config(args.v4_config)
    frame4 = V4.build_frame(cfg4, data.load_splits(_paths(cfg4)[3]), _paths(cfg4)[1])
    grid = RB.ah_grid(cfg, frame)
    he = RB.human_error(cfg, frame)
    ext_pt, ext_bt = RB.external_dev(cfg, frame, frame4)
    grid.to_csv(results_dir / "robust_ah_grid.csv", index=False)
    he.to_csv(results_dir / "robust_human_error.csv", index=False)
    ext_pt.to_csv(results_dir / "robust_external_dev_point.csv", index=False)
    ext_bt.to_csv(results_dir / "robust_external_dev_bootstrap.csv", index=False)
    tb = []
    for Y in (0.85, 0.90, 0.95):  # 元の dev・a_h=1.00 で目標だけを変えた選定込みブートストラップ（90% は主分析と同じ）
        _, dd = PL.cost_bootstrap(cfg, frame, Y=Y)
        tb.append(dd.assign(Y=Y))
    import pandas as pd
    tb = pd.concat(tb, ignore_index=True)
    tb.to_csv(results_dir / "robust_target_bootstrap.csv", index=False)
    print(tb.round(3).to_string(index=False))
    print(grid.pivot_table(index=["a_h", "Y"], columns="pipeline", values="review_rate").round(3).to_string())
    print(he.pivot_table(index=["p_correct_kept", "p_wrong_fixed", "Y"], columns="pipeline", values="total_jpy_per_case").round(1).to_string())
    print(ext_bt.round(3).to_string(index=False))


def main(argv=None):
    load_dotenv(ROOT / ".env", override=True)
    p = argparse.ArgumentParser(prog="ready_eval")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--pilot", type=int, default=None)
    r.add_argument("--models", nargs="*", default=None)
    r.add_argument("--retry-errors", action="store_true")
    sub.add_parser("analyze")
    sub.add_parser("simulate")
    sr = sub.add_parser("signals-run")
    sr.add_argument("--pilot", type=int, default=None)
    sub.add_parser("signals-analyze")
    sub.add_parser("signals-simulate")
    cr = sub.add_parser("cross-run")
    cr.add_argument("--pilot", type=int, default=None)
    sub.add_parser("cross-analyze")
    sub.add_parser("cross-simulate")
    pr = sub.add_parser("pipelines-run")
    pr.add_argument("--pilot", type=int, default=None)
    sub.add_parser("pipelines-analyze")
    sub.add_parser("pipelines-simulate")
    pw = sub.add_parser("power")
    pw.add_argument("--reps", type=int, default=400)
    rb = sub.add_parser("robustness")
    rb.add_argument("--v4-config", default="config_v4.yaml")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    {"prepare": cmd_prepare, "check": cmd_check, "run": cmd_run,
     "analyze": cmd_analyze, "simulate": cmd_simulate, "signals-run": cmd_signals_run,
     "signals-analyze": cmd_signals_analyze, "signals-simulate": cmd_signals_simulate,
     "cross-run": cmd_cross_run, "cross-analyze": cmd_cross_analyze, "cross-simulate": cmd_cross_simulate,
     "pipelines-run": cmd_pipelines_run, "pipelines-analyze": cmd_pipelines_analyze,
     "pipelines-simulate": cmd_pipelines_simulate, "power": cmd_power,
     "robustness": cmd_robustness}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
