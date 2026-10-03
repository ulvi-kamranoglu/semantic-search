"""config.yaml + (varsa) local.yaml birləşdirir. Hər skript konfiqi buradan yükləyir."""
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parent.parent
LOCAL_ALLOWED = {"runtime"}   # local.yaml yalnız bu bölməni dəyişə bilər


def _merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config() -> dict:
    cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    local = ROOT / "configs/local.yaml"
    if local.exists():
        override = yaml.safe_load(local.read_text()) or {}
        bad = set(override) - LOCAL_ALLOWED
        if bad:
            raise ValueError(f"local.yaml yalnız {LOCAL_ALLOWED} dəyişə bilər, tapıldı: {bad}")
        _merge(cfg, override)

    rt = cfg["runtime"]
    if rt["num_threads"] == "auto":
        rt["num_threads"] = psutil.cpu_count(logical=False) or 1
    if rt["device"] == "auto":
        try:
            import torch
            rt["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            rt["device"] = "cpu"

    cfg["paths"] = {k: str(ROOT / v) for k, v in cfg["paths"].items()}
    return cfg


if __name__ == "__main__":
    import json
    print(json.dumps(load_config(), indent=2, ensure_ascii=False))
