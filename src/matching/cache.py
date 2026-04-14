"""SQLite-backed cache for market match results.

Avoids re-running expensive fuzzy matching on every scan cycle.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class MatchCache:
    """Persistent cache for market matches."""

    def __init__(self, db_path: str | Path = "match_cache.db", ttl_hours: int = 24):
        self.db_path = str(db_path)
        self.ttl_seconds = ttl_hours * 3600
        self._conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                kalshi_id TEXT NOT NULL,
                poly_id TEXT NOT NULL,
                similarity REAL NOT NULL,
                outcome_mapping TEXT NOT NULL DEFAULT '{}',
                settlement_risk TEXT NOT NULL DEFAULT 'similar',
                created_at REAL NOT NULL,
                PRIMARY KEY (kalshi_id, poly_id)
            )
        """)
        self._conn.commit()

    def get(
        self, kalshi_id: str, poly_id: str
    ) -> dict | None:
        """Look up a cached match. Returns None if not found or expired."""
        row = self._conn.execute(
            "SELECT similarity, outcome_mapping, settlement_risk, created_at "
            "FROM matches WHERE kalshi_id = ? AND poly_id = ?",
            (kalshi_id, poly_id),
        ).fetchone()

        if row is None:
            return None

        similarity, mapping_json, risk, created_at = row
        if time.time() - created_at > self.ttl_seconds:
            self.delete(kalshi_id, poly_id)
            return None

        return {
            "similarity": similarity,
            "outcome_mapping": json.loads(mapping_json),
            "settlement_risk": risk,
        }

    def get_all_matches_for(self, kalshi_id: str) -> list[dict]:
        """Get all cached matches for a Kalshi market."""
        rows = self._conn.execute(
            "SELECT poly_id, similarity, outcome_mapping, settlement_risk, created_at "
            "FROM matches WHERE kalshi_id = ?",
            (kalshi_id,),
        ).fetchall()

        results = []
        now = time.time()
        for poly_id, similarity, mapping_json, risk, created_at in rows:
            if now - created_at <= self.ttl_seconds:
                results.append({
                    "poly_id": poly_id,
                    "similarity": similarity,
                    "outcome_mapping": json.loads(mapping_json),
                    "settlement_risk": risk,
                })
        return results

    def put(
        self,
        kalshi_id: str,
        poly_id: str,
        similarity: float,
        outcome_mapping: dict[str, str] | None = None,
        settlement_risk: str = "similar",
    ):
        """Store a match result."""
        self._conn.execute(
            "INSERT OR REPLACE INTO matches "
            "(kalshi_id, poly_id, similarity, outcome_mapping, settlement_risk, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                kalshi_id,
                poly_id,
                similarity,
                json.dumps(outcome_mapping or {}),
                settlement_risk,
                time.time(),
            ),
        )
        self._conn.commit()

    def delete(self, kalshi_id: str, poly_id: str):
        self._conn.execute(
            "DELETE FROM matches WHERE kalshi_id = ? AND poly_id = ?",
            (kalshi_id, poly_id),
        )
        self._conn.commit()

    def purge_expired(self):
        """Remove all expired entries."""
        cutoff = time.time() - self.ttl_seconds
        self._conn.execute("DELETE FROM matches WHERE created_at < ?", (cutoff,))
        self._conn.commit()

    def close(self):
        self._conn.close()
