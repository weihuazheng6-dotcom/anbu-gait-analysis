"""Redirect project Python scratch files and common caches to this drive."""
import os
from pathlib import Path


def configure():
    root = Path(__file__).resolve().parent
    runtime = root / "tmp" / "runtime"
    cache = root / "tmp" / "cache"
    paths = {
        "TEMP": runtime / "temp",
        "TMP": runtime / "temp",
        "TMPDIR": runtime / "temp",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
        "XDG_CACHE_HOME": cache / "xdg",
        "MPLCONFIGDIR": cache / "matplotlib",
        "TORCH_HOME": cache / "torch",
        "HF_HOME": cache / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "transformers",
        "JOBLIB_TEMP_FOLDER": runtime / "joblib",
        "PIP_CACHE_DIR": cache / "pip",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

