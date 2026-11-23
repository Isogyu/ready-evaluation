# READY論文に基づくレビュー振り分けの検証（BANKING77）

「AIの正答率だけでモデルを選んでいないか？」という問いを、READY 論文（arXiv:2609.02095）の評価の考え方に沿って、銀行問い合わせの77クラス分類（BANKING77）で小規模に検証した。**論文の完全再現ではなく予備実験**である。

個人で行った検証で、特定の組織とは関係ない。

## 要点

**問い**：品質目標（レビュー後の信頼性）を満たすのに人間が何%を確認すればよいか、またそのときの総コストは、正答率や API 単価の比較だけで分かるか。

**分かったこと**（Claude 3モデル・BANKING77・1回の呼び出しで終わる分類。いずれも a_h=100% の仮定つき）

| 結果 | 数値（95%区間） | 根拠 | 位置づけ |
|---|---|---|---|
| 同じ Haiku の回答のまま、Opus に検証させた値で振り分けると、目標90%に必要なレビュー率が約半分になった | v5：48.4%→23.4%（しきい値選定込みブートストラップでの差の平均 −23.7pt、−32.4〜−7.4）。v4：48.6%→24.8%（同 −22.9pt、−31.6〜−8.6） | `results_v5/pipeline_point.csv`、`pipeline_bootstrap_diffs.csv` | v5 の H4（事前登録）支持。v4 の主仮説は判定保留で、弱い形だけ成立 |
| API 単価が最も安い Haiku が、目標90%では総コストが最も高かった | 必要レビュー率 Opus 30.8%・Sonnet 29.4%・Haiku 52.6%、1件当たり 41・39・70円（人件費4,000円/時・2分/件は仮定）。Haiku−Opus のレビュー率の差 +26pt（+5〜+39） | `results_v2/cost_analysis.csv`、`selection_bootstrap_diffs.csv` | 探索的（v2 の事前登録仮説ではない） |
| 3モデルが一致しても正しいとは限らない | 全員一致408件のうち41件（10%）が同じ誤り | `results_v4/predictions_signals.csv`（test の3モデルの予測から集計） | 探索的 |
| 確信度の校正の良さと、誤りを見分ける力（AUROC）は一致しなかった | Haiku は ECE が最も小さいが AUROC は Opus より有意に低い | `results_v2/hypotheses.csv`（H4b） | 事前登録・支持 |

**分からなかったこと**

- 「安いモデル＋強い検証」と強いモデル単独のどちらが安いか。差は +1.9円（−10.4〜+13.8）で、目標や開発データを変えると大小が入れ替わった。
- 論文の観察（正答率がほぼ同じ別々のモデルで、レビュー率がはっきり違う）は、事前登録した形では再現できなかった。
- 実際の AI エージェント、実際の人間レビュー、他社のモデル、他のデータセットでの結果。

**重要な制約**

- 人間のレビューは実測していない。a_h（人間の成功率）を 90%（論文の主仮定）にすると、目標90%はどの構成でも達成できない。人間が AI の誤答の20%を見逃すと、最も安い構成が入れ替わる（`results_v5/robust_*.csv`、事前登録外）。
- 「適格」は1つの test での点ごとの判定で、複数の構成をまたぐ同時の保証ではない。事前登録仮説27件に多重性の調整はしていない。
- 事前登録の文書は API 実行前に作成し、ハッシュを記録したが、Git への登録は全実験の後である。作成時刻を第三者が独立に確かめる手段はない（下の「検証の確かめ方」）。

## 資料

- 結果の要約と考察：[report.html](report.html)（レポート）、[presentation.md](presentation.md)（発表用）
- 論文の整理：[docs/paper_review.md](docs/paper_review.md)
- 方法：[docs/methodology.md](docs/methodology.md)　限界：[docs/limitations.md](docs/limitations.md)　出典：[docs/references.md](docs/references.md)
- 実行前の計画と費用見積もり：[docs/experiment_plan.md](docs/experiment_plan.md)
- 構成の差を見分けるのに必要な件数の見積もり（API 費用なし）：[docs/power_analysis.md](docs/power_analysis.md)
- **読むときの注意**：主分析は「人間のレビューは常に正しい（a_h=100%）」を仮定している。論文の主仮定 a_h=90% では目標90%はどの構成でも達成できず、人間が AI の誤答を見逃すと最も安い構成が入れ替わる。適格判定と総コストの順位はこの仮定つきで読む（report.html 第4章「仮定を変えると結論は変わるか」）。
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

分析に使った版で揃えるなら、代わりにロックファイルを使う（macOS・Python 3.13.2 で作成。他の環境では未確認）：

```bash
.venv/bin/pip install -r requirements-lock.txt
```

```bash
.venv/bin/pip install -e . --no-deps
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

v5 の結論の頑健さの確認（a_h を 90%・95%・100% に変えた判定、人間が AI の誤答を見逃す場合、目標を変えた選定込みブートストラップ、v4 の test を dev にした入れ替え。API を呼ばない、事前登録外）：

```bash
.venv/bin/python -m ready_eval --config config_v5.yaml robustness
```

### キャッシュと中断・再開

- 応答は1件ごとに `results/raw/<model>/<hash>.json` に保存する（v1・v2 で共有。費用上限もここに保存された全応答の合計で判定する）。hash はモデルID・case_id・問い合わせ文・プロンプト版・生成設定から作る。
- 同じ条件の結果があれば API を呼ばない。途中で止まっても `run` を再実行すれば続きから進む。
- APIエラー・タイムアウトだったケースだけをやり直すには `run --retry-errors`。
- 累積費用（保存済みの全応答の合計）と実行中の予約分が上限を超えそうになると、新しい呼び出しを止める。

### 設定の変更

`config.yaml` で、件数・モデル・料金・費用上限・目標信頼性・a_h・δ・ECE のビン数・人件費・レビュー時間・為替を変えられる。件数を減らすと統計的な不確実性が大きくなる。

## 検証の確かめ方（API 費用なし）

第三者が、公開されている証拠だけで次の3点を確かめられる。

1. **保存済みの応答から結果が再現すること。** `results*/raw/` に API の生の応答 30,100件（費用合計 $23.62）がある。上の各 `analyze` コマンド・`power`・`robustness` を実行すると、`results*/` の CSV が作り直される。2026-11-23 に全版で実行し、Git に登録済みの CSV と差分が出ないことを確認した（`git status` で確かめられる）。
2. **事前登録の本文が、実行時の記録と一致すること。** `docs/hypotheses_v2〜v5.md` は、実行前の本文の末尾にある「（実行後に記入）」を、実行後に「追記日：」で始まる判定結果に置き換えたものである。次のコマンドは追記を外して実行前の文書を復元し、その SHA-256 を `results_v*/run_meta.json` の最初の実行の記録と比べる（4版とも一致）。

   ```bash
   .venv/bin/python -m ready_eval verify-prereg
   ```

   確かめられるのは「追記より前の本文が、`run_meta.json` を書いた時点から変わっていないこと」までである。`run_meta.json` 自体も全実験の後に Git へ登録したため、文書が API 実行より前に書かれたことは、このリポジトリだけでは証明できない。なお v2 の3回目の記録（08:01）は、追記の作業中に分析を再実行したときのもので、ハッシュが異なる。
3. **テストが通ること。** `.venv/bin/python -m pytest -q`（29件。上の 2 も含む）。

API を呼び直した場合の再現（出力のばらつき）は確かめていない。Temperature を指定できないため、同じ結果になるとは限らない。

## ディレクトリ

```
ready-evaluation/
├── config.yaml          実験設定
├── data/                取得したデータ（raw/）、抽出結果（splits.csv）、ラベル一覧
├── src/ready_eval/      実装（data, prompt, clients, runner, metrics, analysis, charts, hypotheses, signals, signals_v4, pipelines, power, robustness, prereg）
├── config_v2.yaml       追加実験 v2 の設定
├── config_v3.yaml       追加実験 v3 の設定
├── config_v4.yaml       追加実験 v4 の設定
├── config_v5.yaml       追加実験 v5 の設定
├── requirements-lock.txt 分析に使った環境の版（pip freeze）
├── tests/               pytest（29件）
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
├── results_v5/          追加実験 v5 の結果（pipeline_point.csv、pipeline_bootstrap*.csv、hypotheses.csv、robust_*.csv など）
├── charts/              グラフ1〜6（v2/ は追加実験 v2、v3/ はグラフ7〜10、v4/ はグラフ11〜14、v5/ はグラフ15〜17、simulation/ はシミュレーション値）
├── docs/                論文レビュー・方法・限界・出典・実験計画
├── report.html          レポート
└── presentation.md      発表用資料
```

## ライセンス・出典

BANKING77（PolyAI、CC BY 4.0）を使用。論文は CC BY 4.0。詳細は [docs/references.md](docs/references.md)。
