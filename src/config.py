"""Configuration loading from config.yaml + .env for API credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv


@dataclass
class ScanConfig:
    interval_seconds: int = 30
    strategies: list[str] = field(
        default_factory=lambda: ["cross_platform", "bundle", "multi_outcome"]
    )
    queries: list[str] = field(
        default_factory=lambda: [
            "Fed rate", "inflation", "Bitcoin", "Trump", "election",
            "GDP", "unemployment", "tariff", "recession", "S&P 500",
        ]
    )


@dataclass
class MatchingConfig:
    similarity_threshold: float = 0.65
    cache_ttl_hours: int = 24
    structural_prefilter: bool = True


@dataclass
class ArbitrageConfig:
    min_edge_pct: float = 0.02
    min_profit_after_fees_cents: float = 0.5
    max_days_to_settlement: int = 90


@dataclass
class ExecutionConfig:
    default_size_contracts: int = 10
    max_size_contracts: int = 100
    use_limit_orders: bool = True
    timeout_seconds: int = 5


@dataclass
class RiskConfig:
    max_position_per_market: int = 50
    max_global_exposure_dollars: float = 500.0
    max_daily_loss_dollars: float = 50.0
    kill_switch_loss_dollars: float = 100.0


@dataclass
class RotationConfig:
    enabled: bool = True
    exit_spread_threshold_cents: float = 1.0


@dataclass
class DashboardConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass
class SmartMoneyConfig:
    enabled: bool = False
    min_wallet_score: float = 0.7
    min_position_usd: float = 500.0
    max_market_price: float = 0.70
    min_market_price: float = 0.05
    lookback_hours: int = 24
    wallet_refresh_hours: int = 6
    max_tracked_wallets: int = 50
    excluded_categories: list[str] = field(
        default_factory=lambda: ["culture", "mentions"]
    )
    db_path: str = "wallet_tracker.db"


@dataclass
class WeatherMLConfig:
    enabled: bool = False
    min_edge: float = 0.10
    min_confidence: float = 0.5
    max_days_out: int = 10
    min_forecast_sources: int = 2
    default_temp_sigma: float = 3.0
    default_precip_sigma: float = 0.5
    calibration_db_path: str = "weather_calibration.db"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: Literal["human", "json"] = "human"


@dataclass
class BotConfig:
    mode: Literal["scan_only", "dry_run", "live"] = "scan_only"
    scan: ScanConfig = field(default_factory=ScanConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    smart_money: SmartMoneyConfig = field(default_factory=SmartMoneyConfig)
    weather_ml: WeatherMLConfig = field(default_factory=WeatherMLConfig)

    # API credentials (loaded from .env, never in YAML)
    kalshi_api_key: str = ""
    kalshi_private_key_path: str = ""
    polymarket_private_key: str = ""
    polymarket_proxy_address: str = ""
    polygonscan_api_key: str = ""
    visual_crossing_api_key: str = ""


def _build_sub(cls, raw: dict | None):
    if raw is None:
        return cls()
    return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def load_config(
    config_path: str | Path = "config.yaml",
    env_path: str | Path | None = ".env",
) -> BotConfig:
    """Load configuration from YAML file and .env for credentials."""
    if env_path and Path(env_path).exists():
        load_dotenv(env_path)

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    cfg = BotConfig(
        mode=raw.get("mode", "scan_only"),
        scan=_build_sub(ScanConfig, raw.get("scan")),
        matching=_build_sub(MatchingConfig, raw.get("matching")),
        arbitrage=_build_sub(ArbitrageConfig, raw.get("arbitrage")),
        execution=_build_sub(ExecutionConfig, raw.get("execution")),
        risk=_build_sub(RiskConfig, raw.get("risk")),
        rotation=_build_sub(RotationConfig, raw.get("rotation")),
        dashboard=_build_sub(DashboardConfig, raw.get("dashboard")),
        logging=_build_sub(LoggingConfig, raw.get("logging")),
        smart_money=_build_sub(SmartMoneyConfig, raw.get("smart_money")),
        weather_ml=_build_sub(WeatherMLConfig, raw.get("weather_ml")),
        kalshi_api_key=os.getenv("KALSHI_API_KEY", ""),
        kalshi_private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
        polymarket_private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
        polymarket_proxy_address=os.getenv("POLYMARKET_PROXY_ADDRESS", ""),
        polygonscan_api_key=os.getenv("POLYGONSCAN_API_KEY", ""),
        visual_crossing_api_key=os.getenv("VISUAL_CROSSING_API_KEY", ""),
    )
    return cfg
