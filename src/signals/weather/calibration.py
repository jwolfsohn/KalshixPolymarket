"""SQLite store for tracking forecast accuracy over time."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone


class CalibrationStore:
    """Track forecast accuracy to improve ensemble weights over time.

    Records each forecast prediction and the actual outcome so that
    source-level accuracy can be measured and weights adjusted.
    """

    def __init__(self, db_path: str = "weather_calibration.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                source TEXT NOT NULL,
                metric TEXT NOT NULL,
                threshold REAL NOT NULL,
                predicted_prob REAL NOT NULL,
                forecast_value REAL NOT NULL,
                target_date TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                UNIQUE(market_id, source)
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                market_id TEXT PRIMARY KEY,
                actual_value REAL NOT NULL,
                exceeded_threshold INTEGER NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_forecasts_source ON forecasts(source);
            CREATE INDEX IF NOT EXISTS idx_forecasts_metric ON forecasts(metric);
        """)
        self._conn.commit()

    def record_forecast(
        self,
        market_id: str,
        source: str,
        metric: str,
        threshold: float,
        predicted_prob: float,
        forecast_value: float,
        target_date: date,
    ):
        """Record a forecast prediction for a market."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO forecasts
               (market_id, source, metric, threshold, predicted_prob,
                forecast_value, target_date, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (market_id, source, metric, threshold, predicted_prob,
             forecast_value, target_date.isoformat(), now),
        )
        self._conn.commit()

    def record_outcome(
        self, market_id: str, actual_value: float, exceeded_threshold: bool
    ):
        """Record the actual outcome for a market."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO outcomes
               (market_id, actual_value, exceeded_threshold, recorded_at)
               VALUES (?, ?, ?, ?)""",
            (market_id, actual_value, int(exceeded_threshold), now),
        )
        self._conn.commit()

    def get_source_accuracy(
        self, source: str, metric: str | None = None, days: int = 90
    ) -> dict:
        """Get accuracy statistics for a forecast source.

        Returns dict with brier_score, calibration_error, and n_forecasts.
        Lower brier_score is better (0 = perfect).
        """
        query = """
            SELECT f.predicted_prob, o.exceeded_threshold
            FROM forecasts f
            JOIN outcomes o ON f.market_id = o.market_id
            WHERE f.source = ?
              AND f.recorded_at >= date('now', ?)
        """
        params: list = [source, f"-{days} days"]

        if metric:
            query += " AND f.metric = ?"
            params.append(metric)

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return {"brier_score": None, "calibration_error": None, "n_forecasts": 0}

        brier_sum = 0.0
        cal_sum = 0.0
        for row in rows:
            pred = row["predicted_prob"]
            actual = row["exceeded_threshold"]
            brier_sum += (pred - actual) ** 2
            cal_sum += abs(pred - actual)

        n = len(rows)
        return {
            "brier_score": brier_sum / n,
            "calibration_error": cal_sum / n,
            "n_forecasts": n,
        }

    def get_optimal_weights(self, metric: str | None = None) -> dict[str, float] | None:
        """Compute optimal source weights based on historical accuracy.

        Returns None if insufficient data (< 10 forecasts per source).
        Uses inverse Brier score as weight.
        """
        sources = ["open_meteo", "nws", "visual_crossing"]
        scores = {}

        for source in sources:
            acc = self.get_source_accuracy(source, metric)
            if acc["n_forecasts"] < 10 or acc["brier_score"] is None:
                return None
            scores[source] = acc["brier_score"]

        # Inverse Brier score (lower is better, so invert)
        inv_scores = {s: 1.0 / max(bs, 0.01) for s, bs in scores.items()}
        total = sum(inv_scores.values())
        return {s: v / total for s, v in inv_scores.items()}

    def close(self):
        self._conn.close()
