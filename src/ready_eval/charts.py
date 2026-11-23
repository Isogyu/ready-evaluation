"""グラフ1〜6（matplotlib、PNG）。モデルの色はモデルの並び順で固定する（順位で塗り替えない）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 検証済み既定パレットの先頭3色（全組み合わせで色覚多様性に対して判別可能）。4モデル以上は4色目以降を使う。
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8985"
POLICY_MARKERS = {"A_ai_only": "o", "B_full_review": "s", "C_random": "^", "D_confidence": "D"}
POLICY_NAMES = {"A_ai_only": "A: AIのみ", "B_full_review": "B: 全件レビュー",
                "C_random": "C: ランダム", "D_confidence": "D: 確信度順"}

plt.rcParams.update({
    "font.family": ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "DejaVu Sans"],
    "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 110,
    "savefig.dpi": 160, "savefig.bbox": "tight", "lines.linewidth": 2,
})


def _colors(models: list[str]) -> dict[str, str]:
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(models)}


def _subtitle(fig, text: str):
    fig.text(0.01, -0.02, text, fontsize=8, color=INK2, ha="left", va="top", wrap=True)


def g1_accuracy(met: pd.DataFrame, models, out: Path, note: str):
    t = met[met.split == "test"].set_index("model_key").loc[models]
    col = _colors(models)
    fig, ax = plt.subplots(figsize=(6.5, 0.7 + 0.6 * len(models)))
    for i, m in enumerate(models):
        r = t.loc[m]
        ax.plot([r.accuracy_ci_lo, r.accuracy_ci_hi], [i, i], color=col[m], lw=2)
        ax.plot(r.accuracy, i, "o", color=col[m], ms=9, mec="white", mew=2)
        ax.text(r.accuracy_ci_hi + 0.005, i, f"{r.accuracy:.1%}（{int(r.n - r.n_errors)}/{int(r.n)}）",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(models)), models)
    ax.invert_yaxis()
    ax.set_xlabel("AI単体の正答率（最終評価データ、95% Wilson区間）")
    ax.set_title("グラフ1　モデル別の正答率")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    lo = max(0.0, t.accuracy_ci_lo.min() - 0.05)
    ax.set_xlim(lo, min(1.0, t.accuracy_ci_hi.max() + 0.12))
    _subtitle(fig, note)
    fig.savefig(out)
    plt.close(fig)


def g2_calibration(bins: pd.DataFrame, met: pd.DataFrame, models, out: Path, note: str):
    col = _colors(models)
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        b = bins[(bins.model_key == m) & (bins.split == "test") & (bins.n > 0)]
        ax.plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=1)
        ax.plot(b.mean_conf, b.accuracy, "-o", color=col[m], ms=8, mec="white", mew=2)
        for _, r in b.iterrows():
            ax.annotate(f"n={int(r.n)}", (r.mean_conf, r.accuracy), textcoords="offset points",
                        xytext=(4, -12), fontsize=7, color=INK2)
        mm = met[(met.model_key == m) & (met.split == "test")].iloc[0]
        ax.set_title(m)
        ax.text(0.03, 0.97, f"ECE(10ビン)={mm['ece_10bins']:.3f}\nBrier={mm.brier:.3f}\n"
                f"AUROC={mm.auroc_error_detection:.2f}" + ("" if mm.auroc_stable else "（不安定）"),
                transform=ax.transAxes, va="top", fontsize=8, color=INK)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("自己申告の確信度（ビン平均）")
    axes[0][0].set_ylabel("実際の正答率")
    fig.suptitle("グラフ2　確信度と実際の正答率（最終評価データ、10等幅ビン）", fontweight="bold")
    _subtitle(fig, note)
    fig.savefig(out)
    plt.close(fig)


def g3_review_vs_reliability(curves: pd.DataFrame, cfg, models, out: Path, note: str):
    col = _colors(models)
    a_hs = [1.0, 0.9] if {1.0, 0.9} <= set(curves.a_h) else sorted(curves.a_h.unique())[::-1]
    fig, axes = plt.subplots(1, len(a_hs), figsize=(6.2 * len(a_hs), 4.8), sharey=True, squeeze=False)
    for ax, a_h in zip(axes[0], a_hs):
        for m in models:
            d = curves[(curves.model_key == m) & (curves.a_h == a_h) & (curves.policy == "D_confidence")]
            d = d.sort_values("review_rate")
            ax.step(d.review_rate, d.reliability, where="post", color=col[m], label=f"{m}（D: 確信度順）")
            r = curves[(curves.model_key == m) & (curves.a_h == a_h) & (curves.policy == "C_random")]
            ax.plot(r.review_rate, r.reliability, ls=":", color=col[m], lw=1.5, label=f"{m}（C: ランダム平均）")
        for y in cfg["analysis"]["target_reliabilities"]:
            ax.axhline(y, color=NEUTRAL, lw=0.8, ls="--")
            ax.text(1.0, y, f" {y:.0%}", va="center", fontsize=7, color=INK2)
        ax.set_title(f"人間のレビュー成功率 a_h = {a_h:.0%}（仮定）")
        ax.set_xlabel("レビュー率（人間が確認する割合）")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0][0].set_ylabel("レビュー後の信頼性（期待値）")
    lo = curves[curves.a_h.isin(a_hs)].reliability.min()
    axes[0][0].set_ylim(max(0, lo - 0.03), 1.01)
    axes[0][-1].legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle("グラフ3　レビュー率とレビュー後信頼性（最終評価データ）", fontweight="bold")
    _subtitle(fig, note)
    fig.savefig(out)
    plt.close(fig)


def g4_review_vs_cost(ready: pd.DataFrame, met: pd.DataFrame, cfg, models, out: Path, note: str, unit: float):
    col = _colors(models)
    a_h = cfg["analysis"]["primary_human_success"]
    targets = cfg["analysis"]["target_reliabilities"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.linspace(0, 1, 101)
    apis = {m: met[(met.model_key == m) & (met.split == "test")].api_cost_jpy_per_case.iloc[0] for m in models}
    for m in models:
        ax.plot(x, apis[m] + x * unit, color=col[m], lw=1.5, label=f"{m}（API {apis[m]:.2f}円/件）")
    for m in models:
        d = ready[(ready.model_key == m) & (ready.a_h == a_h) & (ready.margin == 0.05) & ready.feasible]
        ax.plot(d.test_review_rate, d.total_cost_jpy_per_case, "o", color=col[m], ms=8, mec="white", mew=2)
    ax.text(0.98, 0.04, f"総コスト ＝ API費用 ＋ レビュー率 × {unit:.0f}円\nAPI費用が小さく3本の線はほぼ重なる。\n点＝各目標の選定結果",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xlabel("レビュー率")
    ax.set_ylabel("1件当たり総コスト（円）")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("レビュー率と総コスト")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    w = 0.6 / max(1, len(models))
    for j, m in enumerate(models):
        d = ready[(ready.model_key == m) & (ready.a_h == a_h) & (ready.margin == 0.05)].set_index("target")
        for i, Y in enumerate(targets):
            r = d.loc[Y]
            xpos = i + (j - (len(models) - 1) / 2) * w
            if not r.feasible:
                continue
            lo, hi = apis[m] + r.review_rate_lo * unit, apis[m] + r.review_rate_hi * unit
            ax2.plot([xpos, xpos], [lo, hi], color=col[m], lw=1.5)
            ax2.plot(xpos, r.total_cost_jpy_per_case, "o", ms=8, mec=col[m], mew=2,
                     color=col[m] if r.qualified else "white")
    for m in models:
        ax2.plot([], [], "o", ms=8, color=col[m], label=m)
    ax2.set_xticks(range(len(targets)), [f"{t:.0%}" for t in targets])
    ax2.set_xlim(-0.5, len(targets) - 0.5)
    ax2.set_xlabel("目標信頼性 Y")
    ax2.set_title("目標ごとの1件当たり総コスト（縦線：ブートストラップ95%区間）")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"グラフ4　レビュー率と1件当たり総運用コスト（a_h={a_h:.0%}、δ=0.05）", fontweight="bold")
    _subtitle(fig, "白抜き＝片側95%下限が目標未満。" + note)
    fig.savefig(out)
    plt.close(fig)


def g5_required_review(ready: pd.DataFrame, cfg, models, out: Path, note: str):
    col = _colors(models)
    a_hs = cfg["analysis"]["human_success_rates"]
    targets = cfg["analysis"]["target_reliabilities"]
    fig, axes = plt.subplots(1, len(a_hs), figsize=(4.6 * len(a_hs), 4.4), sharey=True, squeeze=False)
    w = 0.6 / max(1, len(models))
    for ax, a_h in zip(axes[0], a_hs):
        for j, m in enumerate(models):
            d = ready[(ready.model_key == m) & (ready.a_h == a_h) & (ready.margin == 0.05)].set_index("target")
            for i, Y in enumerate(targets):
                xpos = i + (j - (len(models) - 1) / 2) * w
                r = d.loc[Y]
                if not r.feasible:
                    ax.text(xpos, 1.02, "×", ha="center", color=col[m], fontsize=10)
                    continue
                ax.plot([xpos, xpos], [r.review_rate_lo, r.review_rate_hi], color=col[m], lw=1.5)
                ax.plot(xpos, r.test_review_rate, "o", ms=8, mec=col[m], mew=2,
                        color=col[m] if r.qualified else "white")
        ax.set_xticks(range(len(targets)), [f"{t:.0%}" for t in targets])
        ax.set_xlim(-0.5, len(targets) - 0.5)
        ax.set_title(f"a_h = {a_h:.0%}")
        ax.set_xlabel("目標信頼性 Y")
        ax.set_ylim(-0.03, 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0][0].set_ylabel("必要レビュー率（最終評価データ）")
    for m in models:
        axes[0][0].plot([], [], "o", ms=8, color=col[m], label=m)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("グラフ5　目標信頼性ごとの必要レビュー率（しきい値は開発データで決定、δ=0.05）", fontweight="bold")
    _subtitle(fig, "塗りつぶし＝片側95%下限が目標以上（適格）、白抜き＝下限が目標未満（保証できない）、"
              "×＝開発データで達成できるしきい値なし。縦線はブートストラップ95%区間。" + note)
    fig.savefig(out)
    plt.close(fig)


def g6_cost_effectiveness(summary: pd.DataFrame, models, out: Path, note: str, target: float):
    col = _colors(models)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    s = summary[(summary.policy.isin(["A_ai_only", "B_full_review"])) | (summary.target == target)]
    for _, r in s.iterrows():
        if pd.isna(r.total_cost_jpy_per_case):
            continue
        ax.plot(r.total_cost_jpy_per_case, r.reliability_after_review, POLICY_MARKERS[r.policy],
                color=col[r.model_key], ms=9, mec="white", mew=1.5)
    for p, mk in POLICY_MARKERS.items():
        ax.plot([], [], mk, color=NEUTRAL, label=POLICY_NAMES[p])
    for m in models:
        ax.plot([], [], "-", color=col[m], label=m)
    ax.set_xlabel("1件当たり総コスト（円）")
    ax.set_ylabel("レビュー後の信頼性（期待値）")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"グラフ6　各方針の費用対効果（C・D は目標 {target:.0%} 用のレビュー率）")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")
    _subtitle(fig, note)
    fig.savefig(out)
    plt.close(fig)


def make_all(tables: dict, cfg: dict, out_dir: Path, mode_note: str, unit_review_cost: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(tables["metrics"].model_key))
    note = mode_note
    paths = {
        "g1": out_dir / "graph1_accuracy.png", "g2": out_dir / "graph2_calibration.png",
        "g3": out_dir / "graph3_review_vs_reliability.png", "g4": out_dir / "graph4_review_vs_cost.png",
        "g5": out_dir / "graph5_required_review_rate.png", "g6": out_dir / "graph6_cost_effectiveness.png",
    }
    g1_accuracy(tables["metrics"], models, paths["g1"], note)
    g2_calibration(tables["calibration_bins"], tables["metrics"], models, paths["g2"], note)
    g3_review_vs_reliability(tables["policy_curves"], cfg, models, paths["g3"], note)
    cost_note = (f"人件費 {cfg['cost']['wage_jpy_per_hour']:,}円/時・レビュー {cfg['cost']['review_minutes']}分/件・"
                 f"1ドル={cfg['cost']['usd_jpy']}円は仮定。a_h={cfg['analysis']['primary_human_success']:.0%}。" + note)
    g4_review_vs_cost(tables["ready_qualification"], tables["metrics"], cfg, models, paths["g4"], cost_note, unit_review_cost)
    g5_required_review(tables["ready_qualification"], cfg, models, paths["g5"], note)
    g6_cost_effectiveness(tables["policy_summary"], models, paths["g6"], cost_note, 0.90)
    return list(paths.values())


# ------------------------------------------------------------------ v3：振り分け信号の比較
SIG_ORDER = ["self", "cons", "ver", "comb"]
SIG_LABEL = {"self": "自己申告", "cons": "一致度", "ver": "自己検証", "comb": "組み合わせ"}


def _sig_colors(order=None):
    return {s: SERIES[i % len(SERIES)] for i, s in enumerate(order or SIG_ORDER)}


def v3_auroc(met: pd.DataFrame, models, out: Path, note: str, order=None, labels=None,
             title="グラフ7　振り分け信号ごとの誤り検出性能（予測は同じ・最終評価データ）"):
    SIG_ORDER_, SIG_LABEL_ = (order or SIG_ORDER), (labels or SIG_LABEL)
    col = _sig_colors(SIG_ORDER_)
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.8), sharex=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for i, s in enumerate(SIG_ORDER_):
            r = met[(met.model_key == f"{m}:{s}") & (met.split == "test")].iloc[0]
            ax.plot([r.auroc_ci_lo, r.auroc_ci_hi], [i, i], color=col[s], lw=2)
            ax.plot(r.auroc_error_detection, i, "o", color=col[s], ms=9, mec="white", mew=2)
            ax.text(r.auroc_error_detection, i + 0.22, f"{r.auroc_error_detection:.3f}", ha="center", va="top", fontsize=8, color=INK)
        ax.set_yticks(range(len(SIG_ORDER_)), [SIG_LABEL_[s] for s in SIG_ORDER_])
        ax.set_ylim(len(SIG_ORDER_) - 0.3, -0.5)
        ax.set_title(m)
        ax.set_xlabel("誤り検出 AUROC（95%区間）")
    fig.suptitle(title, fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.45)
    _subtitle(fig, note)
    fig.savefig(out)
    plt.close(fig)


def v3_required_review(sel: pd.DataFrame, ready: pd.DataFrame, models, out: Path, note: str, Y: float = 0.90,
                       order=None, labels=None, title="グラフ8　信号ごとの必要レビュー率"):
    SIG_ORDER_, SIG_LABEL_ = (order or SIG_ORDER), (labels or SIG_LABEL)
    col = _sig_colors(SIG_ORDER_)
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.8), sharex=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for i, s in enumerate(SIG_ORDER_):
            k = f"{m}:{s}"
            b = sel[(sel.model_key == k) & (sel.a_h == 1.0) & (sel.target == Y)].iloc[0]
            r = ready[(ready.model_key == k) & (ready.a_h == 1.0) & (ready.margin == 0.05) & (ready.target == Y)].iloc[0]
            ax.plot([b.review_rate_lo, b.review_rate_hi], [i, i], color=col[s], lw=2)
            ax.plot(r.test_review_rate, i, "o", color=col[s], ms=9, mec="white", mew=2)
            ax.text(r.test_review_rate, i + 0.22, f"{r.test_review_rate:.1%}", ha="center", va="top", fontsize=8, color=INK)
        ax.set_yticks(range(len(SIG_ORDER_)), [SIG_LABEL_[s] for s in SIG_ORDER_])
        ax.set_ylim(len(SIG_ORDER_) - 0.3, -0.5)
        ax.set_title(m)
        ax.set_xlabel(f"目標{Y:.0%}の必要レビュー率")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(f"{title}（目標{Y:.0%}、a_h=100%、δ=0.05）", fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.45)
    _subtitle(fig, "点＝実際の最終評価データでの値、横線＝しきい値選定込みのブートストラップ95%区間。" + note)
    fig.savefig(out)
    plt.close(fig)


def v3_curves(curves: pd.DataFrame, models, out: Path, note: str, order=None, labels=None,
              title="グラフ9　信号ごとのレビュー率と信頼性（同じ予測、最終評価データ）"):
    SIG_ORDER_, SIG_LABEL_ = (order or SIG_ORDER), (labels or SIG_LABEL)
    col = _sig_colors(SIG_ORDER_)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for s in SIG_ORDER_:
            d = curves[(curves.model_key == f"{m}:{s}") & (curves.a_h == 1.0) & (curves.policy == "D_confidence")]
            d = d.sort_values("review_rate")
            ax.step(d.review_rate, d.reliability, where="post", color=col[s], label=SIG_LABEL_[s], lw=1.8)
        r = curves[(curves.model_key == f"{m}:self") & (curves.a_h == 1.0) & (curves.policy == "C_random")]
        ax.plot(r.review_rate, r.reliability, ls=":", color=NEUTRAL, lw=1.5, label="ランダム")
        ax.axhline(0.90, color=NEUTRAL, lw=0.8, ls="--")
        ax.set_title(m)
        ax.set_xlabel("レビュー率")
        ax.set_xlim(0, 0.8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0][0].set_ylabel("レビュー後の信頼性（a_h=100%）")
    axes[0][-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(title, fontweight="bold", y=1.03)
    _subtitle(fig, note)
    fig.savefig(out)
    plt.close(fig)


def v3_methods(msumm: pd.DataFrame, models, out: Path, note: str, title="グラフ10　しきい値の選び方：固定の安全余裕 δ=0.05 と下限法（a_h=100%）",
               sigs=("self", "comb"), labels=None):
    labels = labels or SIG_LABEL
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), squeeze=False)
    ax, ax2 = axes[0]
    col = {"delta": NEUTRAL, "lcb": SERIES[0]}
    name = {"delta": "δ=0.05 法", "lcb": "下限法"}
    rows = [(m, s) for m in models for s in sigs]
    for i, (m, s) in enumerate(rows):
        for j, meth in enumerate(("delta", "lcb")):
            r = msumm[(msumm.model_key == f"{m}:{s}") & (msumm.method == meth)].iloc[0]
            y = i + (j - 0.5) * 0.3
            ax.plot([r.review_rate_lo, r.review_rate_hi], [y, y], color=col[meth], lw=2)
            ax.plot(r.review_rate_mean, y, "o", color=col[meth], ms=8, mec="white", mew=1.5,
                    label=name[meth] if i == 0 else None)
            ax2.barh(y, r.share_qualified, height=0.28, color=col[meth])
    ylabels = [f"{m}・{labels[s]}" for m, s in rows]
    for a in (ax, ax2):
        a.set_yticks(range(len(rows)), ylabels)
        a.invert_yaxis()
    ax2.set_yticklabels([])
    ax.set_xlabel("目標90%のレビュー率（ブートストラップ平均と95%区間）")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_xlabel("最終評価データで適格になった割合")
    ax2.set_xlim(0, 1)
    ax2.axvline(0.9, color=INK2, lw=0.8, ls="--")
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(title, fontweight="bold")
    _subtitle(fig, note)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ v4：別モデルの情報を使った信号
def v4_panels(tables: dict, models, order, labels, out_dir: Path, note: str, Y: float = 0.90):
    """モデルごとに使える信号が違う（自分自身による検証は無い）ため、パネルごとに並びを作る。色は信号で固定。"""
    met, sel, ready, curves = (tables["metrics"], tables["selection_bootstrap"], tables["ready_qualification"],
                               tables["policy_curves"])
    col = _sig_colors(order)
    have = set(met.model_key)

    def avail(m):
        return [s for s in order if f"{m}:{s}" in have]

    # グラフ11：AUROC
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.0), sharex=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        sigs = avail(m)
        for i, s in enumerate(sigs):
            r = met[(met.model_key == f"{m}:{s}") & (met.split == "test")].iloc[0]
            ax.plot([r.auroc_ci_lo, r.auroc_ci_hi], [i, i], color=col[s], lw=2)
            ax.plot(r.auroc_error_detection, i, "o", color=col[s], ms=9, mec="white", mew=2)
            ax.text(r.auroc_error_detection, i + 0.22, f"{r.auroc_error_detection:.3f}", ha="center", va="top", fontsize=8, color=INK)
        ax.set_yticks(range(len(sigs)), [labels[s] for s in sigs])
        ax.set_ylim(len(sigs) - 0.3, -0.5)
        ax.set_title(m)
        ax.set_xlabel("誤り検出 AUROC（95%区間）")
    fig.suptitle("グラフ11　別モデルの情報を使った信号の誤り検出性能（予測は同じ・最終評価データ）", fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.5)
    _subtitle(fig, note)
    fig.savefig(out_dir / "graph11_cross_auroc.png")
    plt.close(fig)

    # グラフ12：必要レビュー率
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.0), sharex=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        sigs = avail(m)
        for i, s in enumerate(sigs):
            k = f"{m}:{s}"
            b = sel[(sel.model_key == k) & (sel.a_h == 1.0) & (sel.target == Y)].iloc[0]
            r = ready[(ready.model_key == k) & (ready.a_h == 1.0) & (ready.margin == 0.05) & (ready.target == Y)].iloc[0]
            ax.plot([b.review_rate_lo, b.review_rate_hi], [i, i], color=col[s], lw=2)
            ax.plot(r.test_review_rate, i, "o", color=col[s], ms=9, mec="white", mew=2)
            ax.text(r.test_review_rate, i + 0.22, f"{r.test_review_rate:.1%}", ha="center", va="top", fontsize=8, color=INK)
        ax.set_yticks(range(len(sigs)), [labels[s] for s in sigs])
        ax.set_ylim(len(sigs) - 0.3, -0.5)
        ax.set_title(m)
        ax.set_xlabel(f"目標{Y:.0%}の必要レビュー率")
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(f"グラフ12　信号ごとの必要レビュー率（目標{Y:.0%}、a_h=100%、δ=0.05）", fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.5)
    _subtitle(fig, "点＝最終評価データでの値、横線＝しきい値選定込みのブートストラップ95%区間。" + note)
    fig.savefig(out_dir / "graph12_cross_required_review.png")
    plt.close(fig)

    # グラフ13：レビュー率と信頼性
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for s in avail(m):
            if s.startswith("xver_") and s != f"xver_{'opus-5-5' if m != 'opus-5-5' else 'sonnet-5-5'}":
                continue  # 見やすさのため、検証は「強い検証モデル」だけ描く
            d = curves[(curves.model_key == f"{m}:{s}") & (curves.a_h == 1.0) & (curves.policy == "D_confidence")]
            d = d.sort_values("review_rate")
            ax.step(d.review_rate, d.reliability, where="post", color=col[s], label=labels[s], lw=1.8)
        r = curves[(curves.model_key == f"{m}:self") & (curves.a_h == 1.0) & (curves.policy == "C_random")]
        ax.plot(r.review_rate, r.reliability, ls=":", color=NEUTRAL, lw=1.5, label="ランダム")
        ax.axhline(Y, color=NEUTRAL, lw=0.8, ls="--")
        ax.set_title(m)
        ax.set_xlabel("レビュー率")
        ax.set_xlim(0, 0.8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0][0].set_ylabel("レビュー後の信頼性（a_h=100%）")
    fig.suptitle("グラフ13　信号ごとのレビュー率と信頼性（同じ予測、最終評価データ）", fontweight="bold", y=1.03)
    _subtitle(fig, note)
    fig.savefig(out_dir / "graph13_cross_curves.png")
    plt.close(fig)


# ------------------------------------------------------------------ v5：運用構成の総コスト比較
def v5_costs(summ: pd.DataFrame, point: pd.DataFrame, order, labels, out: Path, note: str):
    s, p = summ.set_index("pipeline"), point.set_index("pipeline")
    order = [k for k in order if k in s.index]
    colors = {"P_opus": SERIES[0], "P_sonnet": SERIES[1], "P_haiku": SERIES[2], "P_hv": SERIES[3],
              "P_hg": SERIES[4], "P_sv": SERIES[6]}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    for i, k in enumerate(order):
        ax.plot([s.loc[k, "total_lo"], s.loc[k, "total_hi"]], [i, i], color=colors[k], lw=2)
        ax.plot(p.loc[k, "total_jpy_per_case"], i, "o", color=colors[k], ms=9, mec="white", mew=2)
        ax.text(p.loc[k, "total_jpy_per_case"], i + 0.22, f"{p.loc[k, 'total_jpy_per_case']:.1f}円", ha="center", va="top", fontsize=8, color=INK)
        ax2.plot([s.loc[k, "review_rate_lo"], s.loc[k, "review_rate_hi"]], [i, i], color=colors[k], lw=2)
        ax2.plot(p.loc[k, "review_rate"], i, "o", color=colors[k], ms=9, mec="white", mew=2)
        ax2.text(p.loc[k, "review_rate"], i + 0.22, f"{p.loc[k, 'review_rate']:.1%}", ha="center", va="top", fontsize=8, color=INK)
    for a in (ax, ax2):
        a.set_yticks(range(len(order)), [labels[k] for k in order])
        a.set_ylim(len(order) - 0.3, -0.5)
    ax2.set_yticklabels([])
    ax.set_xlabel("1件当たり総コスト（円、API＋レビュー）")
    ax2.set_xlabel("必要レビュー率")
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("グラフ15　運用構成ごとの総コストと必要レビュー率（目標90%、a_h=100%、δ=0.05）", fontweight="bold", y=1.03)
    _subtitle(fig, "点＝最終評価データでの値、横線＝しきい値選定込みのブートストラップ95%区間。人件費4,000円/時・2分/件は仮定。" + note)
    fig.savefig(out)
    plt.close(fig)


def v5_sensitivity(sens: pd.DataFrame, order, labels, out: Path, note: str):
    colors = {"P_opus": SERIES[0], "P_sonnet": SERIES[1], "P_haiku": SERIES[2], "P_hv": SERIES[3],
              "P_hg": SERIES[4], "P_sv": SERIES[6]}
    wages = sorted(sens.wage_jpy_per_hour.unique())
    fig, axes = plt.subplots(1, len(wages), figsize=(4.4 * len(wages), 3.8), sharey=False, squeeze=False)
    for ax, w in zip(axes[0], wages):
        for k in [o for o in order if o in set(sens.pipeline)]:
            d = sens[(sens.wage_jpy_per_hour == w) & (sens.pipeline == k)].sort_values("review_minutes")
            ax.plot(d.review_minutes, d.total_jpy_per_case, "-o", color=colors[k], ms=5, label=labels[k])
        ax.set_title(f"人件費 {w:,}円/時")
        ax.set_xlabel("1件当たりレビュー時間（分）")
    axes[0][0].set_ylabel("1件当たり総コスト（円）")
    axes[0][-1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("グラフ16　人件費とレビュー時間を変えたときの総コスト（目標90%）", fontweight="bold", y=1.03)
    _subtitle(fig, note)
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ 必要件数の見積もり
def power_chart(sd: pd.DataFrame, req: pd.DataFrame, labels, out: Path, note: str):
    tot = sd[sd.metric == "total"]
    pairs = list(dict.fromkeys(zip(tot.a, tot.b)))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (a, b) in enumerate(pairs):
        g = tot[(tot.a == a) & (tot.b == b)].sort_values("n")
        ax.plot(g.n, 2.8016 * g.sd, "-o", color=SERIES[i % len(SERIES)], ms=5, label=f"{labels[a]} − {labels[b]}")
    ax.set_xscale("log")
    ticks = sorted(tot.n.unique())
    ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax.minorticks_off()
    for y in (2, 5, 10):
        ax.axhline(y, color=NEUTRAL, ls="--", lw=0.8)
        ax.text(ticks[-1], y, f" {y}円", va="center", fontsize=8, color=INK2)
    ax.axvline(880, color=NEUTRAL, ls=":", lw=0.8)
    ax.text(880, 1.0, " BANKING77 の未使用 test 約880件", fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel("開発・最終評価それぞれの件数")
    ax.set_ylabel("見分けられる最小の総コスト差（円/件）")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.suptitle("グラフ17　件数と、見分けられる最小の総コスト差（両側5%・検出力80%）", fontweight="bold", y=1.02)
    fig.text(0.01, -0.22, "v5 の予測を母集団とみなした再標本化（各件数400回）。Opus・Sonnet 単独が絡む比較は、件数を増やしてもほとんど下がらない。"
             "人件費4,000円/時・2分/件は仮定。" + note, fontsize=8, color=INK2, ha="left", va="top")
    fig.savefig(out)
    plt.close(fig)
