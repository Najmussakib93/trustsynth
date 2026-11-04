"""Explicit aggregate disclosure controls; never accepts patient frames for export.

Small event counts, their complements and associated rates are suppressed.
Experimental settings (seeds, epochs, number of features) are not patient counts.
Zero-record facts can be reported as boolean erasure flags; counts remain <11.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results" / "revision"
PRIVATE = ROOT / "results" / "private" / "revision"
MIN_COUNT = 11
SUPPRESSED = "<11"


def count(value):
    value = int(value)
    if value < 0:
        raise ValueError("A count cannot be negative")
    return SUPPRESSED if value < MIN_COUNT else value


def fraction(events, total):
    events, total = int(events), int(total)
    if total < MIN_COUNT or events < MIN_COUNT or total - events < MIN_COUNT:
        return SUPPRESSED
    return events / total


def category_counts(series, categories=None):
    counts = series.fillna("MISSING").astype(str).value_counts()
    if categories is not None:
        counts = counts.reindex(categories, fill_value=0)
    # If only one cell is suppressed, also suppress the smallest remaining cell.
    # This prevents recovery from a separately published total.
    hidden = counts < MIN_COUNT
    secondary = None
    if hidden.sum() == 1 and (~hidden).any():
        secondary = counts[~hidden].idxmin()
    return {str(k): SUPPRESSED if hidden.loc[k] else "suppressed" if k == secondary else int(v)
            for k, v in counts.items()}


def event_summary(events, total):
    safe_count = count(events)
    if events >= MIN_COUNT and total - events < MIN_COUNT:
        safe_count = "suppressed"
    return {"count": safe_count, "fraction": fraction(events, total)}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Unsupported aggregate JSON type")


def write_json(path, aggregate):
    """Caller must construct a schema of aggregates, never pass row records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aggregate, indent=2, default=_json_default,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, aggregates, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(aggregates[0]) if aggregates else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(aggregates)


def private_path(name):
    path = (PRIVATE / name).resolve()
    if PRIVATE.resolve() not in path.parents:
        raise ValueError("Private artifact must remain inside results/private/revision")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
