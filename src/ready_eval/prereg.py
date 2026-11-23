"""事前登録文書（docs/hypotheses_v*.md）の本文が API 実行時から変わっていないかを確かめる。

各文書は、実行前は判定結果の欄が「（実行後に記入）」の1行だった。実行後、その行を
「追記日：…」で始まる追記に置き換えた。そこで、現在の文書の「追記日：」より前の本文に
「（実行後に記入）」の行を戻して元の文書を復元し、その SHA-256 を、実行時に
results_v*/run_meta.json へ記録した値（最初の実行の値）と比べる。

これで確かめられるのは「追記より前の本文が、run_meta.json に記録した時点の本文と一致すること」
だけである。run_meta.json 自体は全実験の後に Git へ登録したので、作成時刻は第三者には確かめられない。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PLACEHOLDER = "\n（実行後に記入）\n"
APPENDIX_MARKER = "\n追記日："


def reconstruct(text: str) -> str:
    """追記を外し、実行前の文書を復元する。追記がなければ ValueError。"""
    i = text.find(APPENDIX_MARKER)
    if i < 0:
        raise ValueError("追記（「追記日：」で始まる行）が見つからない")
    return text[:i] + PLACEHOLDER


def check(root: Path, versions=(2, 3, 4, 5)) -> list[dict]:
    rows = []
    for v in versions:
        doc = f"docs/hypotheses_v{v}.md"
        meta = json.loads((root / f"results_v{v}/run_meta.json").read_text(encoding="utf-8"))
        recorded = meta["runs"][0]["preregistered_doc_sha256"][doc]
        text = (root / doc).read_text(encoding="utf-8")
        actual = hashlib.sha256(reconstruct(text).encode("utf-8")).hexdigest()
        rows.append({"doc": doc, "first_run_finished_at": meta["runs"][0]["finished_at"],
                     "recorded_sha256": recorded, "reconstructed_sha256": actual,
                     "match": actual == recorded})
    return rows
