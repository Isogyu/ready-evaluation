# READY論文に基づくレビュー振り分けの検証（BANKING77）

「AIの正答率だけでモデルを選んでいないか？」という問いを、READY 論文（arXiv:2609.02095）の評価の考え方に沿って、銀行問い合わせの77クラス分類（BANKING77）で小規模に検証した。**論文の完全再現ではなく予備実験**である。

個人で行った検証で、特定の組織とは関係ない。

- 結果の要約と考察：[report.html](report.html)（レポート）、[presentation.md](presentation.md)（発表用）
- 論文の整理：[docs/paper_review.md](docs/paper_review.md)
- 方法：[docs/methodology.md](docs/methodology.md)　限界：[docs/limitations.md](docs/limitations.md)　出典：[docs/references.md](docs/references.md)
- 実行前の計画と費用見積もり：[docs/experiment_plan.md](docs/experiment_plan.md)
- 構成の差を見分けるのに必要な件数の見積もり（API 費用なし）：[docs/power_analysis.md](docs/power_analysis.md)
- 追加実験の仮説と判定基準（実行前に固定）・判定結果：[v2](docs/hypotheses_v2.md)、[v3](docs/hypotheses_v3.md)、[v4](docs/hypotheses_v4.md)、[v5](docs/hypotheses_v5.md)

## 実施した実験（2026-11-23）

| 項目 | 予備実験 v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| 内容 | モデル比較（探索） | モデル比較 | 同じ予測で、同一モデル内の信号を比較 | 同じ予測で、別モデルの検証・一致を比較 | 運用構成どうしの総コスト比較 |
| 設定ファイル | `config.yaml` | `config_v2.yaml` | `config_v3.yaml` | `config_v4.yaml` | `config_v5.yaml` |
| test | 200件 | 500件（新規） | 500件（新規） | 500件（新規） | 500件（新規） |
| 仮説 | なし | H1〜H6 | H1〜H6 | H1〜H5 | H1〜H5 |
| 新規 API 呼び出し | 900件 | 2,694件 | 約16,500件 | 7,500件 | 2,500件 |
| API 費用 | $0.77 | $2.10 | $12.56 | $5.37 | $2.81（累計 $23.62） |
| 出力 | `results/` | `results_v2/` | `results_v3/` | `results_v4/` | `results_v5/` |

すべての test は互いに重複しない。dev は v2〜v4 で同じ500件。累積費用の上限は v3 から $30（利用者の承認による）。

モデルは両方とも claude-opus-5-5 / claude-sonnet-5-5 / claude-haiku-5-5（effort=low）。

## 必要なもの

- macOS（Apple Silicon で確認）、Python 3.11 以上。GPU 不要。
- 実測するなら Anthropic API キー（Claude Code の契約とは別）。キーが無くてもシミュレーションと、保存済み結果からの分析はできる。

## 再現手順

```bash
cd ready-evaluation
```

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e ".[dev]"
```

API キーを使う場合は `.env` に書く（git には含めない）：

```
ANTHROPIC_API_KEY=sk-ant-...
```

テスト：

```bash
.venv/bin/python -m pytest -q
```

データ取得と分割（`data/splits.csv` を作る。シード固定なので同じ case_id になる）：

```bash
.venv/bin/python -m ready_eval prepare
```

キーとモデルIDの確認（課金なし）：

```bash
.venv/bin/python -m ready_eval check
```

試走（dev の先頭5件 × 全モデル）と本番実行（**費用が発生する**。`config.yaml` の `budget_usd` で上限を設定）：

```bash
.venv/bin/python -m ready_eval run --pilot 5
```

```bash
.venv/bin/python -m ready_eval run
```

分析とグラフ（`results/` と `charts/` に出力）：

```bash
.venv/bin/python -m ready_eval analyze
```

API を使わずに全工程を試す（出力は `results/simulation/` と `charts/simulation/`。**実測ではない**）：

```bash
.venv/bin/python -m ready_eval simulate
```

追加実験 v2 を再現する（v1 の `data/splits.csv` が先に必要。分析時に仮説 H1〜H6 の判定 `results_v2/hypotheses.csv` も出力する）：

```bash
.venv/bin/python -m ready_eval --config config_v2.yaml prepare
```

```bash
.venv/bin/python -m ready_eval --config config_v2.yaml run
```

```bash
.venv/bin/python -m ready_eval --config config_v2.yaml analyze
```

追加実験 v3（v2 の `data/splits_v2.csv` が先に必要）。`signals-run` は元の分類 → 追加分類4回 → 自己検証の順に実行し、`signals-analyze` が信号ごとの表・仮説判定（`results_v3/hypotheses.csv`）・グラフ7〜10を出力する：

```bash
.venv/bin/python -m ready_eval --config config_v3.yaml prepare
```

```bash
.venv/bin/python -m ready_eval --config config_v3.yaml signals-run
```

```bash
.venv/bin/python -m ready_eval --config config_v3.yaml signals-analyze
```

追加実験 v4（v3 の `data/splits_v3.csv` が先に必要）。`cross-run` は3モデルの分類と、互いの予測を検証する6通りを実行し、`cross-analyze` が表・仮説判定・グラフ11〜14を出力する：

```bash
.venv/bin/python -m ready_eval --config config_v4.yaml prepare
```

```bash
.venv/bin/python -m ready_eval --config config_v4.yaml cross-run
```

```bash
.venv/bin/python -m ready_eval --config config_v4.yaml cross-analyze
```

追加実験 v5（v4 の `data/splits_v4.csv` が先に必要）：

```bash
.venv/bin/python -m ready_eval --config config_v5.yaml prepare
```

```bash
.venv/bin/python -m ready_eval --config config_v5.yaml pipelines-run
```

```bash
.venv/bin/python -m ready_eval --config config_v5.yaml pipelines-analyze
```

必要件数の見積もり（v5 の結果から。API を呼ばない）：

```bash
.venv/bin/python -m ready_eval --config config_v5.yaml power
```

### キャッシュと中断・再開

- 応答は1件ごとに `results/raw/<model>/<hash>.json` に保存する（v1・v2 で共有。費用上限もここに保存された全応答の合計で判定する）。hash はモデルID・case_id・問い合わせ文・プロンプト版・生成設定から作る。
- 同じ条件の結果があれば API を呼ばない。途中で止まっても `run` を再実行すれば続きから進む。
- APIエラー・タイムアウトだったケースだけをやり直すには `run --retry-errors`。
- 累積費用（保存済みの全応答の合計）と実行中の予約分が上限を超えそうになると、新しい呼び出しを止める。

### 設定の変更

`config.yaml` で、件数・モデル・料金・費用上限・目標信頼性・a_h・δ・ECE のビン数・人件費・レビュー時間・為替を変えられる。件数を減らすと統計的な不確実性が大きくなる。

## ディレクトリ

```
ready-evaluation/
├── config.yaml          実験設定
├── data/                取得したデータ（raw/）、抽出結果（splits.csv）、ラベル一覧
├── src/ready_eval/      実装（data, prompt, clients, runner, metrics, analysis, charts, hypotheses, signals, signals_v4, pipelines, power）
├── config_v2.yaml       追加実験 v2 の設定
├── config_v3.yaml       追加実験 v3 の設定
├── config_v4.yaml       追加実験 v4 の設定
├── config_v5.yaml       追加実験 v5 の設定
├── tests/               pytest（25件）
├── results/
│   ├── predictions.csv          全予測（モデル×ケース）
│   ├── metrics.csv              正答率・Brier・ECE・AUROC・AURC・API費用
│   ├── cost_analysis.csv        方針A〜Dの比較（レビュー率・信頼性・1件当たりコスト）
│   ├── ready_qualification.csv  目標×a_h×δ ごとのしきい値・レビュー率・下限・判定
│   ├── selection_bootstrap*.csv しきい値選定込みのブートストラップ
│   ├── cost_sensitivity.csv     人件費×レビュー時間の感度分析
│   ├── pairwise_tests.csv       McNemar 検定（Holm 補正）
│   ├── policy_curves.csv, calibration_bins.csv
│   ├── run_meta.json            実行日時・モデル・設定・SDK版
│   └── raw/                     API の生の応答
├── results_v2/          追加実験 v2 の結果（hypotheses.csv、exploratory_margin.csv を含む）
├── results_v3/          追加実験 v3 の結果（predictions_signals.csv、method_bootstrap.csv、hypotheses.csv など）
├── results_v4/          追加実験 v4 の結果（同上）
├── results_v5/          追加実験 v5 の結果（pipeline_point.csv、pipeline_bootstrap*.csv、hypotheses.csv など）
├── charts/              グラフ1〜6（v2/ は追加実験 v2、v3/ はグラフ7〜10、v4/ はグラフ11〜14、v5/ はグラフ15〜17、simulation/ はシミュレーション値）
├── docs/                論文レビュー・方法・限界・出典・実験計画
├── report.html          レポート
└── presentation.md      発表用資料
```

## ライセンス・出典

BANKING77（PolyAI、CC BY 4.0）を使用。論文は CC BY 4.0。詳細は [docs/references.md](docs/references.md)。
