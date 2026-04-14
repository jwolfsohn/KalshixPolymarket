"""SQLite persistence for tracked wallet profiles and activity."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class WalletProfile:
    """Scored wallet profile with trading statistics."""
    address: str
    total_trades: int
    winning_trades: int
    win_rate: float
    total_volume_usd: float
    avg_position_size_usd: float
    total_pnl_usd: float
    categories: dict[str, int]  # category -> trade count
    last_active: datetime
    score: float  # composite insider likelihood score (0-1)
    label: str  # "smart_money", "likely_insider", "bot", "retail"


class WalletDB:
    """Persistent storage for tracked wallet profiles and activity."""

    def __init__(self, db_path: str = "wallet_tracker.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS wallets (
                address TEXT PRIMARY KEY,
                total_trades INTEGER NOT NULL DEFAULT 0,
                winning_trades INTEGER NOT NULL DEFAULT 0,
                win_rate REAL NOT NULL DEFAULT 0,
                total_volume_usd REAL NOT NULL DEFAULT 0,
                avg_position_size_usd REAL NOT NULL DEFAULT 0,
                total_pnl_usd REAL NOT NULL DEFAULT 0,
                categories_json TEXT NOT NULL DEFAULT '{}',
                last_active TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                label TEXT NOT NULL DEFAULT 'unknown',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tracked_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT NOT NULL,
                market_id TEXT NOT NULL,
                market_title TEXT NOT NULL,
                outcome TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT '',
                resolved INTEGER NOT NULL DEFAULT 0,
                resolution TEXT,
                UNIQUE(address, market_id, timestamp)
            );

            CREATE TABLE IF NOT EXISTS tracked_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT NOT NULL,
                market_id TEXT NOT NULL,
                market_title TEXT NOT NULL,
                outcome TEXT NOT NULL,
                size REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                category TEXT NOT NULL DEFAULT '',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                UNIQUE(address, market_id, outcome)
            );

            CREATE INDEX IF NOT EXISTS idx_wallets_score ON wallets(score);
            CREATE INDEX IF NOT EXISTS idx_wallets_label ON wallets(label);
            CREATE INDEX IF NOT EXISTS idx_trades_address ON tracked_trades(address);
            CREATE INDEX IF NOT EXISTS idx_positions_address ON tracked_positions(address);
            CREATE INDEX IF NOT EXISTS idx_positions_last_seen ON tracked_positions(last_seen);
        """)
        self._conn.commit()

    def upsert_wallet(self, profile: WalletProfile):
        """Insert or update a wallet profile."""
        import json
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO wallets
               (address, total_trades, winning_trades, win_rate,
                total_volume_usd, avg_position_size_usd, total_pnl_usd,
                categories_json, last_active, score, label, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (profile.address, profile.total_trades, profile.winning_trades,
             profile.win_rate, profile.total_volume_usd, profile.avg_position_size_usd,
             profile.total_pnl_usd, json.dumps(profile.categories),
             profile.last_active.isoformat(), profile.score, profile.label, now),
        )
        self._conn.commit()

    def get_tracked_wallets(
        self, min_score: float = 0.7, limit: int = 50
    ) -> list[WalletProfile]:
        """Retrieve wallets above the minimum score threshold."""
        import json
        rows = self._conn.execute(
            """SELECT * FROM wallets WHERE score >= ? AND label != 'bot'
               ORDER BY score DESC LIMIT ?""",
            (min_score, limit),
        ).fetchall()

        wallets = []
        for row in rows:
            wallets.append(WalletProfile(
                address=row["address"],
                total_trades=row["total_trades"],
                winning_trades=row["winning_trades"],
                win_rate=row["win_rate"],
                total_volume_usd=row["total_volume_usd"],
                avg_position_size_usd=row["avg_position_size_usd"],
                total_pnl_usd=row["total_pnl_usd"],
                categories=json.loads(row["categories_json"]),
                last_active=datetime.fromisoformat(row["last_active"]),
                score=row["score"],
                label=row["label"],
            ))
        return wallets

    def record_trade(self, address: str, trade_data: dict):
        """Record a trade for a tracked wallet."""
        self._conn.execute(
            """INSERT OR IGNORE INTO tracked_trades
               (address, market_id, market_title, outcome, side, price,
                size, timestamp, category, resolved, resolution)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (address, trade_data.get("market_id", ""),
             trade_data.get("market_title", ""),
             trade_data.get("outcome", ""),
             trade_data.get("side", "buy"),
             trade_data.get("price", 0),
             trade_data.get("size", 0),
             trade_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
             trade_data.get("category", ""),
             int(trade_data.get("resolved", False)),
             trade_data.get("resolution")),
        )
        self._conn.commit()

    def upsert_position(self, address: str, position_data: dict):
        """Insert or update a tracked position."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO tracked_positions
               (address, market_id, market_title, outcome, size,
                avg_entry_price, current_price, category, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(address, market_id, outcome)
               DO UPDATE SET size=excluded.size,
                   current_price=excluded.current_price,
                   last_seen=excluded.last_seen""",
            (address, position_data.get("market_id", ""),
             position_data.get("market_title", ""),
             position_data.get("outcome", ""),
             position_data.get("size", 0),
             position_data.get("avg_entry_price", 0),
             position_data.get("current_price", 0),
             position_data.get("category", ""),
             now, now),
        )
        self._conn.commit()

    def get_recent_positions(
        self, address: str, hours: int = 24
    ) -> list[dict]:
        """Get positions for a wallet seen within the last N hours."""
        rows = self._conn.execute(
            """SELECT * FROM tracked_positions
               WHERE address = ?
                 AND last_seen >= datetime('now', ?)
               ORDER BY last_seen DESC""",
            (address, f"-{hours} hours"),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_all_recent_positions(self, hours: int = 24) -> list[dict]:
        """Get all positions across tracked wallets seen in the last N hours."""
        rows = self._conn.execute(
            """SELECT p.*, w.score, w.win_rate, w.label
               FROM tracked_positions p
               JOIN wallets w ON p.address = w.address
               WHERE p.last_seen >= datetime('now', ?)
                 AND w.label != 'bot'
               ORDER BY w.score DESC""",
            (f"-{hours} hours",),
        ).fetchall()
        return [dict(row) for row in rows]

    def purge_stale(self, days: int = 90):
        """Remove old data to keep the database manageable."""
        self._conn.execute(
            "DELETE FROM tracked_trades WHERE timestamp < datetime('now', ?)",
            (f"-{days} days",),
        )
        self._conn.execute(
            "DELETE FROM tracked_positions WHERE last_seen < datetime('now', ?)",
            (f"-{days} days",),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
