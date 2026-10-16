"""Transactional, resumable state for independent manual trajectory QC."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManualQCState:
    """SQLite-backed decisions, progress, cursor, and undo history."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._closed = False
        self._create_schema()

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise RuntimeError("Manual-QC state is closed")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except Exception:
            if not self._closed:
                self.connection.rollback()
            raise

    def _create_schema(self) -> None:
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    platform_code TEXT PRIMARY KEY,
                    source_sha256 TEXT NOT NULL,
                    drogue_signature TEXT NOT NULL,
                    decision_revision INTEGER NOT NULL DEFAULT 0,
                    finalized_revision INTEGER,
                    finalized_parquet_sha256 TEXT,
                    finalized_utc TEXT,
                    resample_revision INTEGER,
                    resample_frequency_seconds REAL,
                    show_resampled INTEGER NOT NULL DEFAULT 0,
                    first_opened_utc TEXT NOT NULL,
                    last_opened_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rejected_observations (
                    platform_code TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    source_obs_index INTEGER NOT NULL,
                    updated_utc TEXT NOT NULL,
                    PRIMARY KEY (platform_code, source_obs_index),
                    FOREIGN KEY (platform_code) REFERENCES sources(platform_code)
                );
                CREATE TABLE IF NOT EXISTS rejected_sst_observations (
                    platform_code TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    source_obs_index INTEGER NOT NULL,
                    updated_utc TEXT NOT NULL,
                    PRIMARY KEY (platform_code, source_obs_index),
                    FOREIGN KEY (platform_code) REFERENCES sources(platform_code)
                );
                CREATE TABLE IF NOT EXISTS reviewed_chunks (
                    platform_code TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    chunk_key TEXT NOT NULL,
                    start_time_ns INTEGER,
                    end_time_ns INTEGER,
                    completed_utc TEXT NOT NULL,
                    dirty INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (platform_code, chunk_key),
                    FOREIGN KEY (platform_code) REFERENCES sources(platform_code)
                );
                CREATE TABLE IF NOT EXISTS session_cursor (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    platform_code TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    chunk_key TEXT,
                    selected_source_obs_index INTEGER,
                    saved_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform_code TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    action TEXT NOT NULL,
                    before_json TEXT NOT NULL,
                    source_indices_json TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    undone INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) != SCHEMA_VERSION:
                raise ValueError(
                    f"Manual-QC state schema {row['value']} is unsupported; expected {SCHEMA_VERSION}"
                )

    def register_source(
        self, platform_code: str, source_sha256: str, drogue_signature: str,
    ) -> bool:
        """Register an immutable source and return whether drogue scope changed."""
        platform, digest = str(platform_code), str(source_sha256)
        stamp = _now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM sources WHERE platform_code = ?", (platform,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO sources(
                           platform_code, source_sha256, drogue_signature,
                           first_opened_utc, last_opened_utc
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (platform, digest, str(drogue_signature), stamp, stamp),
                )
                return False
            if str(row["source_sha256"]) != digest:
                raise ValueError(
                    f"Raw source changed for platform {platform}; saved manual-QC state was not applied"
                )
            changed = str(row["drogue_signature"]) != str(drogue_signature)
            connection.execute(
                "UPDATE sources SET drogue_signature = ?, last_opened_utc = ? WHERE platform_code = ?",
                (str(drogue_signature), stamp, platform),
            )
            if changed:
                connection.execute(
                    "UPDATE reviewed_chunks SET dirty = 1 WHERE platform_code = ?",
                    (platform,),
                )
                connection.execute(
                    """UPDATE sources SET finalized_revision = NULL,
                           finalized_parquet_sha256 = NULL, finalized_utc = NULL
                       WHERE platform_code = ?""",
                    (platform,),
                )
            return changed

    def rejected_sources(self, platform_code: str, source_sha256: str) -> set[int]:
        rows = self.connection.execute(
            """SELECT source_obs_index FROM rejected_observations
               WHERE platform_code = ? AND source_sha256 = ?""",
            (str(platform_code), str(source_sha256)),
        ).fetchall()
        return {int(row["source_obs_index"]) for row in rows}

    def rejected_sst_sources(
        self, platform_code: str, source_sha256: str,
    ) -> set[int]:
        rows = self.connection.execute(
            """SELECT source_obs_index FROM rejected_sst_observations
               WHERE platform_code = ? AND source_sha256 = ?""",
            (str(platform_code), str(source_sha256)),
        ).fetchall()
        return {int(row["source_obs_index"]) for row in rows}

    def set_rejected(
        self,
        platform_code: str,
        source_sha256: str,
        source_indices: Iterable[int],
        *,
        rejected: bool,
        dirty_chunk_keys: Iterable[str] = (),
    ) -> set[int]:
        platform, digest = str(platform_code), str(source_sha256)
        sources = sorted(set(map(int, source_indices)))
        if not sources:
            return set()
        stamp = _now()
        changed: set[int] = set()
        before: dict[str, bool] = {}
        with self.transaction() as connection:
            registered = connection.execute(
                "SELECT source_sha256 FROM sources WHERE platform_code = ?", (platform,),
            ).fetchone()
            if registered is None or str(registered["source_sha256"]) != digest:
                raise ValueError(f"Platform {platform} is not registered with this source hash")
            for source in sources:
                exists = connection.execute(
                    """SELECT 1 FROM rejected_observations
                       WHERE platform_code = ? AND source_obs_index = ?""",
                    (platform, source),
                ).fetchone() is not None
                before[str(source)] = exists
                if rejected and not exists:
                    connection.execute(
                        """INSERT INTO rejected_observations(
                               platform_code, source_sha256, source_obs_index, updated_utc
                           ) VALUES (?, ?, ?, ?)""",
                        (platform, digest, source, stamp),
                    )
                    changed.add(source)
                elif not rejected and exists:
                    connection.execute(
                        """DELETE FROM rejected_observations
                           WHERE platform_code = ? AND source_obs_index = ?""",
                        (platform, source),
                    )
                    changed.add(source)
            if changed:
                connection.execute(
                    """UPDATE sources SET decision_revision = decision_revision + 1,
                           finalized_revision = NULL,
                           finalized_parquet_sha256 = NULL, finalized_utc = NULL,
                           last_opened_utc = ? WHERE platform_code = ?""",
                    (stamp, platform),
                )
                for key in set(map(str, dirty_chunk_keys)):
                    connection.execute(
                        """UPDATE reviewed_chunks SET dirty = 1
                           WHERE platform_code = ? AND chunk_key = ?""",
                        (platform, key),
                    )
                connection.execute(
                    """INSERT INTO audit_log(
                           platform_code, source_sha256, action, before_json,
                           source_indices_json, created_utc
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        platform, digest, "reject" if rejected else "restore",
                        json.dumps(before, sort_keys=True), json.dumps(sorted(changed)), stamp,
                    ),
                )
        return changed

    def set_sst_rejected(
        self,
        platform_code: str,
        source_sha256: str,
        source_indices: Iterable[int],
        *,
        rejected: bool,
        dirty_chunk_keys: Iterable[str] = (),
    ) -> set[int]:
        platform, digest = str(platform_code), str(source_sha256)
        sources = sorted(set(map(int, source_indices)))
        if not sources:
            return set()
        stamp = _now()
        changed: set[int] = set()
        before: dict[str, bool] = {}
        with self.transaction() as connection:
            registered = connection.execute(
                "SELECT source_sha256 FROM sources WHERE platform_code = ?", (platform,),
            ).fetchone()
            if registered is None or str(registered["source_sha256"]) != digest:
                raise ValueError(f"Platform {platform} is not registered with this source hash")
            for source in sources:
                exists = connection.execute(
                    """SELECT 1 FROM rejected_sst_observations
                       WHERE platform_code = ? AND source_obs_index = ?""",
                    (platform, source),
                ).fetchone() is not None
                before[str(source)] = exists
                if rejected and not exists:
                    connection.execute(
                        """INSERT INTO rejected_sst_observations(
                               platform_code, source_sha256, source_obs_index, updated_utc
                           ) VALUES (?, ?, ?, ?)""",
                        (platform, digest, source, stamp),
                    )
                    changed.add(source)
                elif not rejected and exists:
                    connection.execute(
                        """DELETE FROM rejected_sst_observations
                           WHERE platform_code = ? AND source_obs_index = ?""",
                        (platform, source),
                    )
                    changed.add(source)
            if changed:
                connection.execute(
                    """UPDATE sources SET decision_revision = decision_revision + 1,
                           finalized_revision = NULL,
                           finalized_parquet_sha256 = NULL, finalized_utc = NULL,
                           last_opened_utc = ? WHERE platform_code = ?""",
                    (stamp, platform),
                )
                for key in set(map(str, dirty_chunk_keys)):
                    connection.execute(
                        """UPDATE reviewed_chunks SET dirty = 1
                           WHERE platform_code = ? AND chunk_key = ?""",
                        (platform, key),
                    )
                connection.execute(
                    """INSERT INTO audit_log(
                           platform_code, source_sha256, action, before_json,
                           source_indices_json, created_utc
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        platform, digest,
                        "reject_sst" if rejected else "restore_sst",
                        json.dumps(before, sort_keys=True),
                        json.dumps(sorted(changed)), stamp,
                    ),
                )
        return changed

    def mark_chunk_reviewed(
        self,
        platform_code: str,
        source_sha256: str,
        chunk_key: str,
        *,
        start_time_ns: int | None,
        end_time_ns: int | None,
    ) -> None:
        stamp = _now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO reviewed_chunks(
                       platform_code, source_sha256, chunk_key, start_time_ns,
                       end_time_ns, completed_utc, dirty
                   ) VALUES (?, ?, ?, ?, ?, ?, 0)
                   ON CONFLICT(platform_code, chunk_key) DO UPDATE SET
                       source_sha256 = excluded.source_sha256,
                       start_time_ns = excluded.start_time_ns,
                       end_time_ns = excluded.end_time_ns,
                       completed_utc = excluded.completed_utc,
                       dirty = 0""",
                (
                    str(platform_code), str(source_sha256), str(chunk_key),
                    start_time_ns, end_time_ns, stamp,
                ),
            )

    def chunk_states(self, platform_code: str, source_sha256: str) -> dict[str, str]:
        rows = self.connection.execute(
            """SELECT chunk_key, dirty FROM reviewed_chunks
               WHERE platform_code = ? AND source_sha256 = ?""",
            (str(platform_code), str(source_sha256)),
        ).fetchall()
        return {
            str(row["chunk_key"]): "dirty" if bool(row["dirty"]) else "reviewed"
            for row in rows
        }

    def mark_chunks_dirty(self, platform_code: str, chunk_keys: Iterable[str]) -> None:
        keys = set(map(str, chunk_keys))
        if not keys:
            return
        with self.transaction() as connection:
            for key in keys:
                connection.execute(
                    """UPDATE reviewed_chunks SET dirty = 1
                       WHERE platform_code = ? AND chunk_key = ?""",
                    (str(platform_code), key),
                )

    def rejected_records(self, platform_code: str, source_sha256: str) -> dict[int, str]:
        rows = self.connection.execute(
            """SELECT source_obs_index, updated_utc FROM rejected_observations
               WHERE platform_code = ? AND source_sha256 = ?""",
            (str(platform_code), str(source_sha256)),
        ).fetchall()
        return {
            int(row["source_obs_index"]): str(row["updated_utc"])
            for row in rows
        }

    def rejected_sst_records(
        self, platform_code: str, source_sha256: str,
    ) -> dict[int, str]:
        rows = self.connection.execute(
            """SELECT source_obs_index, updated_utc FROM rejected_sst_observations
               WHERE platform_code = ? AND source_sha256 = ?""",
            (str(platform_code), str(source_sha256)),
        ).fetchall()
        return {
            int(row["source_obs_index"]): str(row["updated_utc"])
            for row in rows
        }

    def reviewed_chunk_keys(self, platform_code: str, source_sha256: str) -> set[str]:
        return {
            key for key, value in self.chunk_states(platform_code, source_sha256).items()
            if value == "reviewed"
        }

    def revisions(self, platform_code: str) -> tuple[int, int | None]:
        row = self.connection.execute(
            """SELECT decision_revision, resample_revision FROM sources
               WHERE platform_code = ?""",
            (str(platform_code),),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown manual-QC platform: {platform_code}")
        return int(row["decision_revision"]), (
            None if row["resample_revision"] is None else int(row["resample_revision"])
        )

    def mark_finalized(self, platform_code: str, parquet_sha256: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """UPDATE sources SET finalized_revision = decision_revision,
                       finalized_parquet_sha256 = ?, finalized_utc = ?
                   WHERE platform_code = ?""",
                (str(parquet_sha256), _now(), str(platform_code)),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Unknown manual-QC platform: {platform_code}")

    def finalization_state(self, platform_code: str) -> dict[str, object]:
        row = self.connection.execute(
            """SELECT source_sha256, drogue_signature, decision_revision, finalized_revision,
                      finalized_parquet_sha256, finalized_utc
               FROM sources WHERE platform_code = ?""",
            (str(platform_code),),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown manual-QC platform: {platform_code}")
        finalized = row["finalized_revision"]
        return {
            "source_sha256": str(row["source_sha256"]),
            "drogue_signature": str(row["drogue_signature"]),
            "decision_revision": int(row["decision_revision"]),
            "finalized_revision": None if finalized is None else int(finalized),
            "parquet_sha256": row["finalized_parquet_sha256"],
            "finalized_utc": row["finalized_utc"],
            "current": finalized is not None and int(finalized) == int(row["decision_revision"]),
        }

    def mark_resampled(self, platform_code: str, frequency_seconds: float, *, shown: bool = True) -> None:
        with self.transaction() as connection:
            connection.execute(
                """UPDATE sources SET resample_revision = decision_revision,
                       resample_frequency_seconds = ?, show_resampled = ?, last_opened_utc = ?
                   WHERE platform_code = ?""",
                (float(frequency_seconds), int(bool(shown)), _now(), str(platform_code)),
            )

    def resample_state(self, platform_code: str) -> dict[str, object]:
        row = self.connection.execute(
            """SELECT decision_revision, resample_revision,
                      resample_frequency_seconds, show_resampled
               FROM sources WHERE platform_code = ?""",
            (str(platform_code),),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown manual-QC platform: {platform_code}")
        revision = int(row["decision_revision"])
        sampled = None if row["resample_revision"] is None else int(row["resample_revision"])
        return {
            "decision_revision": revision,
            "resample_revision": sampled,
            "frequency_seconds": row["resample_frequency_seconds"],
            "shown": bool(row["show_resampled"]),
            "stale": sampled is not None and sampled != revision,
        }

    def set_resample_shown(self, platform_code: str, shown: bool) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE sources SET show_resampled = ? WHERE platform_code = ?",
                (int(bool(shown)), str(platform_code)),
            )

    def save_cursor(
        self,
        platform_code: str,
        source_sha256: str,
        chunk_key: str | None,
        selected_source_obs_index: int | None,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO session_cursor(
                       singleton, platform_code, source_sha256, chunk_key,
                       selected_source_obs_index, saved_utc
                   ) VALUES (1, ?, ?, ?, ?, ?)
                   ON CONFLICT(singleton) DO UPDATE SET
                       platform_code = excluded.platform_code,
                       source_sha256 = excluded.source_sha256,
                       chunk_key = excluded.chunk_key,
                       selected_source_obs_index = excluded.selected_source_obs_index,
                       saved_utc = excluded.saved_utc""",
                (
                    str(platform_code), str(source_sha256), chunk_key,
                    selected_source_obs_index, _now(),
                ),
            )

    def load_cursor(self) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT * FROM session_cursor WHERE singleton = 1"
        ).fetchone()
        return None if row is None else dict(row)

    def undo_last_with_domain(
        self, platform_code: str, source_sha256: str,
    ) -> tuple[str, set[int]]:
        platform, digest = str(platform_code), str(source_sha256)
        with self.transaction() as connection:
            row = connection.execute(
                """SELECT * FROM audit_log
                   WHERE platform_code = ? AND source_sha256 = ? AND undone = 0
                   ORDER BY sequence DESC LIMIT 1""",
                (platform, digest),
            ).fetchone()
            if row is None:
                return "position", set()
            domain = "sst" if str(row["action"]).endswith("_sst") else "position"
            table = (
                "rejected_sst_observations"
                if domain == "sst" else "rejected_observations"
            )
            before = {int(key): bool(value) for key, value in json.loads(row["before_json"]).items()}
            for source, was_rejected in before.items():
                if was_rejected:
                    connection.execute(
                        f"""INSERT INTO {table}(
                               platform_code, source_sha256, source_obs_index, updated_utc
                           ) VALUES (?, ?, ?, ?)
                           ON CONFLICT(platform_code, source_obs_index) DO UPDATE SET
                               source_sha256 = excluded.source_sha256,
                               updated_utc = excluded.updated_utc""",
                        (platform, digest, source, _now()),
                    )
                else:
                    connection.execute(
                        f"""DELETE FROM {table}
                            WHERE platform_code = ? AND source_obs_index = ?""",
                        (platform, source),
                    )
            connection.execute(
                "UPDATE audit_log SET undone = 1 WHERE sequence = ?", (int(row["sequence"]),),
            )
            connection.execute(
                """UPDATE sources SET decision_revision = decision_revision + 1,
                       finalized_revision = NULL, finalized_parquet_sha256 = NULL,
                       finalized_utc = NULL WHERE platform_code = ?""",
                (platform,),
            )
            # Undo changes accepted interpolation. Mark progress conservatively
            # in the same transaction so a crash cannot leave false completion.
            connection.execute(
                "UPDATE reviewed_chunks SET dirty = 1 WHERE platform_code = ?",
                (platform,),
            )
            return domain, set(before)

    def undo_last(self, platform_code: str, source_sha256: str) -> set[int]:
        """Backward-compatible source-only undo result."""
        return self.undo_last_with_domain(platform_code, source_sha256)[1]
