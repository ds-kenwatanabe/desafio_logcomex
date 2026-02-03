from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import joblib


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_joblib(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, str(p))


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.int64, np.int32, np.int16, np.int8)):
        return int(x)
    if isinstance(x, (np.float64, np.float32, np.float16)):
        return float(x)
    if isinstance(x, (Path,)):
        return str(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


def fingerprint_df(df: pd.DataFrame, sample_rows: int = 200) -> str:
    """
    Fingerprint para rastreio (shape + colunas + amostra determinística)
    """
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(("|".join(map(str, df.columns))).encode())
    sample = df.head(sample_rows).to_csv(index=False).encode()
    h.update(sample)
    return h.hexdigest()[:16]
