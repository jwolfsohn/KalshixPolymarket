"""Unified scanner orchestrating arbitrage and signal strategies."""

from __future__ import annotations

import logging

from src.arb.bundle import detect_bundle
from src.arb.cross_platform import detect_cross_platform
from src.arb.fees import KalshiFeeCalculator, PolymarketFeeCalculator
from src.arb.models import Market, MatchedMarket, Opportunity
from src.arb.multi_outcome import detect_multi_outcome
from src.config import BotConfig

logger = logging.getLogger(__name__)


class ArbitrageScanner:
    """Orchestrates all strategies (arbitrage + signals) and returns ranked opportunities."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.kalshi_fees = KalshiFeeCalculator()
        self.poly_fees = PolymarketFeeCalculator()

        # Lazily initialized signal strategy components
        self._weather_components = None
        self._smart_money_components = None

        if "weather_ml" in config.scan.strategies and config.weather_ml.enabled:
            self._init_weather_ml()

        if "smart_money" in config.scan.strategies and config.smart_money.enabled:
            self._init_smart_money()

    def _init_weather_ml(self):
        """Initialize weather ML strategy components."""
        from src.signals.weather.calibration import CalibrationStore
        from src.signals.weather.forecast_client import WeatherForecastClient
        from src.signals.weather.model import WeatherEnsembleModel

        wc = self.config.weather_ml
        calibration = CalibrationStore(db_path=wc.calibration_db_path)

        # Try to use calibrated weights if available
        optimal_weights = calibration.get_optimal_weights()
        model = WeatherEnsembleModel(
            source_weights=optimal_weights,
            default_temp_sigma=wc.default_temp_sigma,
            default_precip_sigma=wc.default_precip_sigma,
        )

        client = WeatherForecastClient(
            visual_crossing_api_key=self.config.visual_crossing_api_key,
        )

        self._weather_components = {
            "forecast_client": client,
            "model": model,
            "calibration": calibration,
            "min_edge": wc.min_edge,
            "min_confidence": wc.min_confidence,
            "max_days_out": wc.max_days_out,
            "min_sources": wc.min_forecast_sources,
        }

    def _init_smart_money(self):
        """Initialize smart money strategy components."""
        from src.signals.smart_money.polygon_client import PolygonClient
        from src.signals.smart_money.wallet_db import WalletDB

        sc = self.config.smart_money
        wallet_db = WalletDB(db_path=sc.db_path)
        poly_client = PolygonClient(
            polygonscan_api_key=self.config.polygonscan_api_key,
        )

        self._smart_money_components = {
            "wallet_db": wallet_db,
            "poly_client": poly_client,
            "min_wallet_score": sc.min_wallet_score,
            "min_position_usd": sc.min_position_usd,
            "max_market_price": sc.max_market_price,
            "min_market_price": sc.min_market_price,
            "lookback_hours": sc.lookback_hours,
            "excluded_categories": sc.excluded_categories,
            "wallet_refresh_hours": sc.wallet_refresh_hours,
            "max_tracked_wallets": sc.max_tracked_wallets,
        }

    def scan(
        self,
        matched_markets: list[MatchedMarket],
        kalshi_markets: list[Market],
        poly_markets: list[Market],
    ) -> list[Opportunity]:
        """Run all enabled strategies, deduplicate, and rank by annualized return."""
        opportunities: list[Opportunity] = []
        strategies = self.config.scan.strategies
        arb = self.config.arbitrage

        # --- Riskless Arbitrage Strategies ---

        if "cross_platform" in strategies:
            cross = detect_cross_platform(
                matched_markets,
                self.kalshi_fees,
                self.poly_fees,
                min_edge_pct=arb.min_edge_pct,
                min_profit_after_fees_cents=arb.min_profit_after_fees_cents,
                max_days_to_settlement=arb.max_days_to_settlement,
            )
            logger.info(f"Cross-platform: found {len(cross)} opportunities")
            opportunities.extend(cross)

        if "bundle" in strategies:
            all_markets = kalshi_markets + poly_markets
            bundle = detect_bundle(
                all_markets,
                self.kalshi_fees,
                self.poly_fees,
                min_profit_after_fees_cents=arb.min_profit_after_fees_cents,
            )
            logger.info(f"Bundle: found {len(bundle)} opportunities")
            opportunities.extend(bundle)

        if "multi_outcome" in strategies:
            all_markets = kalshi_markets + poly_markets
            multi = detect_multi_outcome(
                all_markets,
                self.kalshi_fees,
                self.poly_fees,
                min_profit_after_fees_cents=arb.min_profit_after_fees_cents,
            )
            logger.info(f"Multi-outcome: found {len(multi)} opportunities")
            opportunities.extend(multi)

        # --- Signal-Based Directional Strategies ---

        if "weather_ml" in strategies and self._weather_components:
            from src.signals.weather.detector import detect_weather_ml

            weather = detect_weather_ml(
                kalshi_markets=kalshi_markets,
                kalshi_fees=self.kalshi_fees,
                **self._weather_components,
            )
            logger.info(f"Weather ML: found {len(weather)} opportunities")
            opportunities.extend(weather)

        if "smart_money" in strategies and self._smart_money_components:
            from src.signals.smart_money.detector import detect_smart_money

            smart = detect_smart_money(
                poly_fees=self.poly_fees,
                **self._smart_money_components,
            )
            logger.info(f"Smart money: found {len(smart)} opportunities")
            opportunities.extend(smart)

        # Sort by annualized return (highest first); signal strategies without
        # settlement dates get sorted by edge * confidence instead
        opportunities.sort(
            key=lambda o: o.annualized_return if o.annualized_return > 0
            else (o.edge or 0) * (o.signal_confidence or 0) * 100,
            reverse=True,
        )

        logger.info(f"Total opportunities: {len(opportunities)}")
        return opportunities
