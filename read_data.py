from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "clean_data"


def _resolve_folder(folder: str | Path | None) -> Path:
    return Path(folder) if folder is not None else DEFAULT_DATA_DIR


def _read_dataset(filename: str, folder: str | Path | None = None) -> pd.DataFrame:
    path = _resolve_folder(folder) / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")

    data = pd.read_csv(path)
    if {"resp", "truth"}.issubset(data.columns):
        data["accuracy"] = data["resp"] == data["truth"]
    return data


def read_human_data(test: bool = False, folder: str | Path | None = None) -> pd.DataFrame:
    filename = "test_human_data.csv" if test else "human_data.csv"
    return _read_dataset(filename, folder=folder)


def read_pigeon_data(test: bool = False, folder: str | Path | None = None) -> pd.DataFrame:
    filename = "test_pigeon_data.csv" if test else "pigeon_data.csv"
    return _read_dataset(filename, folder=folder)


def read_rat_data(test: bool = False, folder: str | Path | None = None) -> pd.DataFrame:
    filename = "test_rat_data.csv" if test else "rat_data.csv"
    return _read_dataset(filename, folder=folder)


def read_data(
    animal: str = "human",
    folder: str | Path | None = None,
    test: bool = False,
) -> pd.DataFrame:
    readers = {
        "human": read_human_data,
        "pigeon": read_pigeon_data,
        "rat": read_rat_data,
    }
    try:
        reader = readers[animal.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(readers))
        raise ValueError(f"Unknown animal {animal!r}. Expected one of: {valid}") from exc
    return reader(test=test, folder=folder)
