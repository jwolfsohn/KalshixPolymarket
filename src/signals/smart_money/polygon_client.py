"""Fetch wallet activity from Polymarket CLOB API and Polygonscan."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WalletTrade:
    """A single trade made by a wallet on Polymarket."""
    address: str
    market_id: str
    market_title: str
    outcome: str  # "Yes" or "No"
    side: str  # "buy" or "sell"
    price: float
    size: float  # in USDC
    timestamp: datetime
    category: str = ""
    resolved: bool = False
    resolution: str | None = None  # "Yes", "No", or None if unresolved


@dataclass
class WalletPosition:
    """A current open position held by a wallet."""
    address: str
    market_id: str
    market_title: str
    outcome: str
    size: float  # contracts
    avg_entry_price: float
    current_price: float
    category: str = ""
    market_end_date: str = ""


class PolygonClient:
    """Fetch Polymarket wallet data from public APIs.

    Uses a combination of:
    - Polymarket CLOB API for trade history and positions
    - Polymarket Gamma API for market metadata
    - Polygonscan API for on-chain verification
    """

    CLOB_API = "https://clob.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"
    POLYGONSCAN_API = "https://api.polygonscan.com/api"

    # Polymarket CTF Exchange contract on Polygon
    CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

    def __init__(
        self,
        polygonscan_api_key: str = "",
        httpx_client: httpx.Client | None = None,
    ):
        self.polygonscan_key = polygonscan_api_key
        self._client = httpx_client or httpx.Client(timeout=15.0)
        self._owns_client = httpx_client is None

    def close(self):
        if self._owns_client:
            self._client.close()

    def get_leaderboard(self, limit: int = 100) -> list[dict]:
        """Fetch the Polymarket profit leaderboard.

        Returns top wallets by profit. Useful for initial wallet discovery.
        """
        try:
            resp = self._client.get(
                f"{self.GAMMA_API}/leaderboard",
                params={"limit": limit, "sort": "profit"},
            )
            resp.raise_for_status()
            return resp.json() if isinstance(resp.json(), list) else resp.json().get("data", [])
        except Exception as e:
            logger.warning(f"Leaderboard fetch failed: {e}")
            return []

    def get_wallet_history(self, address: str, limit: int = 200) -> list[WalletTrade]:
        """Fetch trade history for a specific wallet address."""
        try:
            resp = self._client.get(
                f"{self.CLOB_API}/activity",
                params={"user": address, "limit": limit},
            )
            resp.raise_for_status()
            raw = resp.json() if isinstance(resp.json(), list) else resp.json().get("data", [])

            trades = []
            for item in raw:
                try:
                    ts = item.get("timestamp", item.get("created_at", ""))
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    elif ts:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = datetime.now(timezone.utc)

                    trades.append(WalletTrade(
                        address=address,
                        market_id=item.get("market", item.get("condition_id", "")),
                        market_title=item.get("title", item.get("question", "")),
                        outcome=item.get("outcome", item.get("asset_id", "")),
                        side=item.get("side", "buy"),
                        price=float(item.get("price", 0)),
                        size=float(item.get("size", item.get("amount", 0))),
                        timestamp=dt,
                        category=item.get("category", ""),
                        resolved=item.get("resolved", False),
                        resolution=item.get("resolution"),
                    ))
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping unparseable trade: {e}")
                    continue

            return trades
        except Exception as e:
            logger.warning(f"Wallet history fetch failed for {address}: {e}")
            return []

    def get_wallet_positions(self, address: str) -> list[WalletPosition]:
        """Fetch current open positions for a wallet."""
        try:
            resp = self._client.get(
                f"{self.CLOB_API}/positions",
                params={"user": address},
            )
            resp.raise_for_status()
            raw = resp.json() if isinstance(resp.json(), list) else resp.json().get("data", [])

            positions = []
            for item in raw:
                try:
                    positions.append(WalletPosition(
                        address=address,
                        market_id=item.get("market", item.get("condition_id", "")),
                        market_title=item.get("title", item.get("question", "")),
                        outcome=item.get("outcome", ""),
                        size=float(item.get("size", 0)),
                        avg_entry_price=float(item.get("avgPrice", item.get("avg_price", 0))),
                        current_price=float(item.get("currentPrice", item.get("price", 0))),
                        category=item.get("category", ""),
                        market_end_date=item.get("endDate", item.get("end_date", "")),
                    ))
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping unparseable position: {e}")
                    continue

            return positions
        except Exception as e:
            logger.warning(f"Positions fetch failed for {address}: {e}")
            return []

    def get_recent_large_trades(
        self, min_size_usd: float = 500, limit: int = 200
    ) -> list[WalletTrade]:
        """Fetch recent large trades across all wallets.

        Useful for discovering new wallets to track.
        """
        try:
            resp = self._client.get(
                f"{self.CLOB_API}/trades",
                params={"limit": limit},
            )
            resp.raise_for_status()
            raw = resp.json() if isinstance(resp.json(), list) else resp.json().get("data", [])

            trades = []
            for item in raw:
                try:
                    size = float(item.get("size", item.get("amount", 0)))
                    price = float(item.get("price", 0))
                    notional = size * price if price > 0 else size
                    if notional < min_size_usd:
                        continue

                    ts = item.get("timestamp", item.get("created_at", ""))
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    elif ts:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = datetime.now(timezone.utc)

                    address = item.get("maker", item.get("user", item.get("taker", "")))
                    trades.append(WalletTrade(
                        address=address,
                        market_id=item.get("market", item.get("condition_id", "")),
                        market_title=item.get("title", item.get("question", "")),
                        outcome=item.get("outcome", item.get("asset_id", "")),
                        side=item.get("side", "buy"),
                        price=price,
                        size=size,
                        timestamp=dt,
                        category=item.get("category", ""),
                    ))
                except (ValueError, KeyError):
                    continue

            return trades
        except Exception as e:
            logger.warning(f"Recent large trades fetch failed: {e}")
            return []

    def verify_wallet_on_chain(self, address: str) -> dict | None:
        """Verify a wallet's on-chain activity via Polygonscan.

        Returns basic transaction count and first/last activity timestamps.
        Useful for verifying that a wallet is real and not a throwaway.
        """
        if not self.polygonscan_key:
            return None

        try:
            resp = self._client.get(
                self.POLYGONSCAN_API,
                params={
                    "module": "account",
                    "action": "txlist",
                    "address": address,
                    "startblock": 0,
                    "endblock": 99999999,
                    "page": 1,
                    "offset": 5,
                    "sort": "asc",
                    "apikey": self.polygonscan_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            txs = data.get("result", [])

            if not txs or isinstance(txs, str):
                return None

            return {
                "address": address,
                "tx_count": len(txs),
                "first_tx": txs[0].get("timeStamp"),
                "verified": True,
            }
        except Exception as e:
            logger.warning(f"On-chain verification failed for {address}: {e}")
            return None
