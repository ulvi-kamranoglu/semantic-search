"""Maşın və mühit məlumatını toplayır. Hər benchmark nəticəsi ilə birgə saxlanılır."""
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import psutil


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _version(name: str) -> str:
    try:
        return __import__(name).__version__
    except Exception:
        return "not installed"


def get_env_info() -> dict:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_model": _cpu_model(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_threads": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 1),
        "packages": {n: _version(n) for n in
                     ["torch", "faiss", "polars", "numpy", "sentence_transformers"]},
        "gpu": None,
    }
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": p.name, "vram_gb": round(p.total_memory / 1e9, 1)}
    except Exception:
        pass
    return info


def get_device() -> str:
    """cuda varsa cuda, yoxsa cpu."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def save_env_info(path="results/env_info.json") -> dict:
    info = get_env_info()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


if __name__ == "__main__":
    print(json.dumps(save_env_info(), indent=2, ensure_ascii=False))
