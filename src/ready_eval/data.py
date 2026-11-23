"""BANKING77 の取得と、開発用・最終評価用の層別抽出。

- 元の train/test 分割を維持する（開発用は train、最終評価用は test から抽出）。
- 77クラスからほぼ均等に抽出する（n // 77 件ずつ＋余りをランダムなクラスに1件ずつ）。
- 抽出結果は data/splits.csv に case_id 付きで保存する。
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve

RAW_DIR = resolve("data/raw")
SPLITS_PATH = resolve("data/splits.csv")
LABELS_PATH = resolve("data/labels.csv")
FILES = {
    "README.md": "README.md",
    "train.parquet": "data/train-00000-of-00001.parquet",
    "test.parquet": "data/test-00000-of-00001.parquet",
}


def download(base_url: str, raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for local, remote in FILES.items():
        dest = raw_dir / local
        if dest.exists() and dest.stat().st_size > 0:
            continue
        urllib.request.urlretrieve(f"{base_url}/{remote}", dest)


def parse_label_names(readme_text: str) -> list[str]:
    """データセットカードの class_label 定義からラベル名を ID 順に取り出す。"""
    pairs = re.findall(r"^\s+'(\d+)':\s*(\S+)\s*$", readme_text, flags=re.M)
    names = {int(i): n for i, n in pairs}
    if sorted(names) != list(range(77)):
        raise ValueError(f"ラベル定義が77件そろっていません（{len(names)}件）")
    return [names[i] for i in range(77)]


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    labels = parse_label_names((raw_dir / "README.md").read_text(encoding="utf-8"))
    train = pd.read_parquet(raw_dir / "train.parquet")
    test = pd.read_parquet(raw_dir / "test.parquet")
    return train, test, labels


def stratified_sample(df: pd.DataFrame, n: int, rng: np.random.Generator, n_classes: int = 77) -> pd.DataFrame:
    """各クラスから floor(n/K) 件、余りはランダムに選んだクラスから1件ずつ追加する。"""
    base, rem = divmod(n, n_classes)
    extra = set(rng.choice(n_classes, size=rem, replace=False).tolist()) if rem else set()
    picks = []
    for c in range(n_classes):
        k = base + (1 if c in extra else 0)
        if k == 0:
            continue
        idx = df.index[df["label"] == c].to_numpy()
        if len(idx) < k:
            raise ValueError(f"クラス {c} の件数が不足しています")
        picks.extend(rng.choice(idx, size=k, replace=False).tolist())
    out = df.loc[sorted(picks)].copy()
    return out


def prepare_splits(cfg: dict, raw_dir: Path = RAW_DIR, out_path: Path | None = None) -> pd.DataFrame:
    """通常は train/test から層別抽出する。data.extend_from があれば（追加実験）：
    dev ＝ 元の dev ＋ train の残りから n_dev_new 件、test ＝ 元の test と重ならない n_test 件。"""
    out_path = resolve(cfg["data"].get("splits_path", "data/splits.csv")) if out_path is None else out_path
    download(cfg["data"]["hf_base_url"], raw_dir)
    train, test, labels = load_raw(raw_dir)
    rng = np.random.default_rng(cfg["seed"])
    base = cfg["data"].get("extend_from")
    if base:
        prev = pd.read_csv(resolve(base))
        prev_dev = prev[prev.split == "dev"].source_index.tolist()
        prev_test = prev[prev.split == "test"].source_index.tolist()
        for extra in cfg["data"].get("exclude_test_from", []):  # さらに古い実験の test も除外する
            e = pd.read_csv(resolve(extra))
            prev_test += e[e.split == "test"].source_index.tolist()
        prev_test = sorted(set(prev_test))
        new_dev = stratified_sample(train.drop(index=prev_dev), cfg["data"].get("n_dev_new", 0), rng)
        dev = pd.concat([train.loc[prev_dev], new_dev]).sort_index()
        tst = stratified_sample(test.drop(index=prev_test), cfg["data"]["n_test"], rng)
    else:
        dev = stratified_sample(train, cfg["data"]["n_dev"], rng)
        tst = stratified_sample(test, cfg["data"]["n_test"], rng)
    rows = []
    for split_name, src_split, df in (("dev", "train", dev), ("test", "test", tst)):
        for i, r in df.iterrows():
            rows.append({
                "case_id": f"{src_split}-{i:05d}",
                "split": split_name,
                "source_split": src_split,
                "source_index": int(i),
                "text": r["text"],
                "label_id": int(r["label"]),
                "label": labels[int(r["label"])],
            })
    splits = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_path, index=False)
    pd.DataFrame({"label_id": range(77), "label": labels}).to_csv(out_path.parent / "labels.csv", index=False)
    return splits


def load_splits(path: Path | str = SPLITS_PATH) -> pd.DataFrame:
    return pd.read_csv(resolve(path))


def load_labels(path: Path = LABELS_PATH) -> list[str]:
    return pd.read_csv(path).sort_values("label_id")["label"].tolist()
