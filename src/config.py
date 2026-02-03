from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """
    Carrega config em YAML ou JSON.
    - YAML: .yml/.yaml
    - JSON: .json
    Retorna dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    suf = p.suffix.lower()
    if suf in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Config YAML requer PyYAML. Instale com: pip install pyyaml"
            ) from e
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if suf == ".json":
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config extension: {suf} (use .yaml/.yml ou .json)")
