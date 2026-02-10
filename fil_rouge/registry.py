from __future__ import annotations
from pathlib import Path
import json
import joblib

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path: Path):
    return joblib.load(path)

def promote_latest(src_dir: Path, latest_dir: Path) -> None:
    """
    Copie logique: on écrase latest en recopiant les fichiers.
    (Simplifié pour éviter bugs OS. On fait du "write" direct ailleurs dans les scripts.)
    """
    latest_dir.mkdir(parents=True, exist_ok=True)