"""Explicit ARCTERX drogue decisions, keyed by string platform identifiers."""

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReviewDecision:
    decision: str
    loss_time: np.datetime64
    note: str


@dataclass
class ReviewTable:
    decisions: dict[str, ReviewDecision]
    sha256: str = ""

    def validate_platforms(self, platforms: set[str]) -> None:
        unknown = set(self.decisions) - platforms
        if unknown:
            raise ValueError(f"Review table contains unknown platform IDs: {sorted(unknown)}")


def read_review_table(path: Path | None) -> ReviewTable:
    if path is None:
        return ReviewTable({})
    content = path.read_bytes()
    reader = csv.DictReader(StringIO(content.decode("utf-8-sig")))
    expected = {"platform_code", "decision", "drogue_off_time_utc", "note"}
    if set(reader.fieldnames or []) != expected or len(reader.fieldnames or []) != len(expected):
        raise ValueError(f"Review CSV must have exactly these columns: {sorted(expected)}")
    result: dict[str, ReviewDecision] = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed review CSV row {reader.line_num}")
        row = {k: v.strip() for k, v in row.items()}
        platform, decision, timestamp = row["platform_code"], row["decision"], row["drogue_off_time_utc"]
        if not platform or platform in result:
            raise ValueError(f"Missing or duplicate review platform ID: {platform!r}")
        if decision not in {"retained", "lost"}:
            raise ValueError(f"Review {platform}: decision must be retained or lost")
        loss = np.datetime64("NaT", "ns")
        if decision == "retained":
            if timestamp:
                raise ValueError(f"Review {platform}: retained must not include a loss date")
        else:
            try:
                parsed = pd.Timestamp(timestamp)
                if pd.isna(parsed) or parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
                    raise ValueError("explicit UTC timezone required")
                loss = parsed.tz_convert("UTC").tz_localize(None).as_unit("ns").to_datetime64()
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"Review {platform}: lost requires a valid UTC timestamp with Z or +00:00") from exc
        result[platform] = ReviewDecision(decision, loss, row["note"])
    return ReviewTable(result, sha256(content).hexdigest())
