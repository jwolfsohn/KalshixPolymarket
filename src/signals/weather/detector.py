"""Weather ML strategy detector — find mispriced Kalshi weather markets."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from src.arb.fees import KalshiFeeCalculator
from src.arb.models import FeeEstimate, Market, Opportunity, OrderLeg
from src.signals.weather.calibration import CalibrationStore
from src.signals.weather.forecast_client import ForecastPoint, WeatherForecastClient
from src.signals.weather.model import WeatherEnsembleModel
from src.signals.weather.parser import WeatherMarketParams, parse_weather_market

logger = logging.getLogger(__name__)


def _fetch_forecasts_for_markets(
    params_list: list[WeatherMarketParams],
    client: WeatherForecastClient,
) -> dict[str, list[ForecastPoint]]:
    """Fetch forecasts for all parsed weather markets."""
    result: dict[str, list[ForecastPoint]] = {}
    for p in params_list:
        forecasts = client.fetch_all(p.latitude, p.longitude, p.target_date, p.metric, p.location)
        if forecasts:
            result[p.market_id] = forecasts
    return result


def detect_weather_ml(
    kalshi_markets: list[Market],
    kalshi_fees: KalshiFeeCalculator,
    forecast_client: WeatherForecastClient,
    model: WeatherEnsembleModel,
    calibration: CalibrationStore | None = None,
    min_edge: float = 0.10,
    min_confidence: float = 0.5,
    max_days_out: int = 10,
    min_sources: int = 2,
) -> list[Opportunity]:
    """Detect mispriced Kalshi weather markets using forecast ensemble.

    1. Filter Kalshi markets to weather category
    2. Parse each market title into structured parameters
    3. Fetch forecasts from multiple sources
    4. Run ensemble model to estimate probability
    5. Compare model probability to market price
    6. Return opportunities where edge exceeds threshold
    """
    today = date.today()
    opportunities: list[Opportunity] = []

    # Step 1-2: Parse weather markets
    parsed: list[WeatherMarketParams] = []
    for mkt in kalshi_markets:
        params = parse_weather_market(mkt.market_id, mkt.title, mkt.category)
        if params is None:
            continue
        days_out = (params.target_date - today).days
        if days_out < 0 or days_out > max_days_out:
            continue
        parsed.append(params)

    if not parsed:
        return []

    logger.info(f"Weather ML: parsed {len(parsed)} weather markets from {len(kalshi_markets)} Kalshi markets")

    # Step 3: Fetch forecasts
    forecasts_map = _fetch_forecasts_for_markets(parsed, forecast_client)

    # Step 4-6: Run model and find opportunities
    for params in parsed:
        forecasts = forecasts_map.get(params.market_id, [])
        if len(forecasts) < min_sources:
            continue

        estimate = model.predict(forecasts, params)

        if estimate.confidence < min_confidence:
            continue

        # Record forecasts for calibration
        if calibration:
            for detail in estimate.source_details:
                calibration.record_forecast(
                    market_id=params.market_id,
                    source=detail["source"],
                    metric=params.metric,
                    threshold=params.threshold,
                    predicted_prob=detail["probability"],
                    forecast_value=detail["forecast_value"],
                    target_date=params.target_date,
                )

        # Find the corresponding Market object to get prices
        mkt = next((m for m in kalshi_markets if m.market_id == params.market_id), None)
        if mkt is None or not mkt.outcomes:
            continue

        # Determine which side to trade
        yes_outcome = next((o for o in mkt.outcomes if o.side == "yes"), None)
        no_outcome = next((o for o in mkt.outcomes if o.side == "no"), None)
        if yes_outcome is None:
            continue

        model_prob = estimate.probability
        market_yes_price = yes_outcome.best_ask
        market_no_price = no_outcome.best_ask if no_outcome else (1.0 - yes_outcome.best_bid)

        # Check YES side: model says higher probability than market
        yes_edge = model_prob - market_yes_price
        # Check NO side: model says lower probability, so NO is underpriced
        no_edge = (1.0 - model_prob) - market_no_price

        # Pick the better side
        if yes_edge >= no_edge and yes_edge >= min_edge:
            side = "buy"
            price = market_yes_price
            outcome = yes_outcome
            edge = yes_edge
            display_model_prob = model_prob
        elif no_edge >= min_edge:
            side = "buy"
            price = market_no_price
            outcome = no_outcome or yes_outcome
            edge = no_edge
            display_model_prob = 1.0 - model_prob
        else:
            continue

        # Calculate fees and profit
        fee = kalshi_fees.fee_per_contract(price, is_taker=True)
        expected_profit = edge - fee
        if expected_profit <= 0:
            continue

        depth = outcome.ask_depth if outcome else 100
        max_contracts = max(1, depth)
        days_to_settlement = (params.target_date - today).days
        ann_return = (expected_profit / price) * (365 / max(days_to_settlement, 1))

        # Build forecast summary for source description
        source_parts = []
        for d in estimate.source_details:
            source_parts.append(f"{d['source']}={d['forecast_value']:.1f}")
        forecast_summary = ", ".join(source_parts)

        opp = Opportunity(
            strategy="weather_ml",
            legs=[
                OrderLeg(
                    platform="kalshi",
                    market_id=params.market_id,
                    outcome_id=outcome.outcome_id,
                    side=side,
                    price=price,
                    size=max_contracts,
                )
            ],
            fees=[
                FeeEstimate(
                    platform="kalshi",
                    fee_per_contract=fee,
                    total_fee=fee * max_contracts,
                    fee_type="taker",
                )
            ],
            gross_profit_per_contract=edge,
            total_fees_per_contract=fee,
            net_profit_per_contract=expected_profit,
            max_contracts=max_contracts,
            total_net_profit=expected_profit * max_contracts,
            matched_market=None,
            detected_at=datetime.now(timezone.utc),
            settlement_date=datetime(
                params.target_date.year, params.target_date.month, params.target_date.day,
                tzinfo=timezone.utc,
            ),
            days_to_settlement=days_to_settlement,
            annualized_return=ann_return,
            description=(
                f"{params.location} {params.metric} {params.threshold_direction} "
                f"{params.threshold} on {params.target_date}"
            ),
            model_probability=display_model_prob,
            market_price=price,
            edge=edge,
            signal_confidence=estimate.confidence,
            signal_source=f"weather_ensemble({estimate.sources_used} sources: {forecast_summary})",
        )
        opportunities.append(opp)

    return opportunities
