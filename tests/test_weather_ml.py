"""Tests for the Weather ML signal strategy."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arb.fees import KalshiFeeCalculator
from src.arb.models import Market, Outcome
from src.signals.weather.forecast_client import ForecastPoint
from src.signals.weather.model import (
    WeatherEnsembleModel,
    normal_cdf,
    prob_below_threshold,
    prob_exceeds_threshold,
)
from src.signals.weather.parser import WeatherMarketParams, parse_weather_market


# ========== Parser Tests ==========


class TestWeatherParser:
    def test_parse_high_temp_market(self):
        result = parse_weather_market(
            "mkt-1",
            "Will the high temperature in San Francisco exceed 70°F on April 20, 2026?",
        )
        assert result is not None
        assert result.location == "San Francisco"
        assert result.metric == "high_temp"
        assert result.threshold == 70.0
        assert result.threshold_direction == "above"
        assert result.target_date == date(2026, 4, 20)

    def test_parse_low_temp_market(self):
        result = parse_weather_market(
            "mkt-2",
            "Will the low temperature in Chicago drop below 20°F on January 15, 2026?",
        )
        assert result is not None
        assert result.location == "Chicago"
        assert result.metric == "low_temp"
        assert result.threshold == 20.0
        assert result.threshold_direction == "below"

    def test_parse_high_temp_be_above(self):
        result = parse_weather_market(
            "mkt-3",
            "Will the high temperature in Denver be above 90°F on July 4, 2026?",
        )
        assert result is not None
        assert result.location == "Denver"
        assert result.threshold == 90.0

    def test_parse_precipitation_market(self):
        result = parse_weather_market(
            "mkt-4",
            "Will it rain more than 0.5 inches in Miami on June 10, 2026?",
        )
        assert result is not None
        assert result.location == "Miami"
        assert result.metric == "precipitation"
        assert result.threshold == 0.5

    def test_parse_snowfall_market(self):
        result = parse_weather_market(
            "mkt-5",
            "Will snowfall in Boston exceed 6 inches on February 1, 2026?",
        )
        assert result is not None
        assert result.location == "Boston"
        assert result.metric == "snowfall"
        assert result.threshold == 6.0

    def test_parse_wind_speed_market(self):
        result = parse_weather_market(
            "mkt-6",
            "Will wind speed in New York exceed 30 mph on March 15, 2026?",
        )
        assert result is not None
        assert result.location == "New York"
        assert result.metric == "wind_speed"
        assert result.threshold == 30.0

    def test_parse_returns_none_for_non_weather(self):
        result = parse_weather_market(
            "mkt-7", "Will Bitcoin exceed $100,000 by December 31, 2026?"
        )
        assert result is None

    def test_parse_returns_none_without_date(self):
        result = parse_weather_market(
            "mkt-8", "Will the high temperature in Dallas exceed 100°F?"
        )
        assert result is None

    def test_parse_returns_none_for_unknown_location(self):
        result = parse_weather_market(
            "mkt-9",
            "Will the high temperature in Timbuktu exceed 110°F on May 1, 2026?",
        )
        assert result is None

    def test_parse_coordinates(self):
        result = parse_weather_market(
            "mkt-10",
            "Will the high temperature in New York exceed 90°F on August 1, 2026?",
        )
        assert result is not None
        assert abs(result.latitude - 40.7128) < 0.01
        assert abs(result.longitude - (-74.006)) < 0.01

    def test_parse_filters_non_weather_category(self):
        result = parse_weather_market(
            "mkt-11",
            "Will the high temperature in Miami exceed 85°F on June 1, 2026?",
            category="politics",
        )
        assert result is None


# ========== Probability Math Tests ==========


class TestProbabilityMath:
    def test_normal_cdf_at_zero(self):
        assert abs(normal_cdf(0) - 0.5) < 1e-10

    def test_normal_cdf_positive(self):
        # P(Z < 2) ≈ 0.9772
        assert abs(normal_cdf(2.0) - 0.9772) < 0.001

    def test_normal_cdf_negative(self):
        # P(Z < -2) ≈ 0.0228
        assert abs(normal_cdf(-2.0) - 0.0228) < 0.001

    def test_prob_exceeds_forecast_well_above(self):
        # Forecast 75°F, threshold 70°F, sigma 3 → high probability
        p = prob_exceeds_threshold(75.0, 70.0, 3.0)
        assert p > 0.9

    def test_prob_exceeds_forecast_well_below(self):
        # Forecast 65°F, threshold 70°F, sigma 3 → low probability
        p = prob_exceeds_threshold(65.0, 70.0, 3.0)
        assert p < 0.1

    def test_prob_exceeds_at_threshold(self):
        # Forecast exactly at threshold → ~50%
        p = prob_exceeds_threshold(70.0, 70.0, 3.0)
        assert abs(p - 0.5) < 0.01

    def test_prob_exceeds_zero_sigma(self):
        assert prob_exceeds_threshold(71.0, 70.0, 0) == 1.0
        assert prob_exceeds_threshold(69.0, 70.0, 0) == 0.0

    def test_prob_below_threshold(self):
        # Forecast 65°F, threshold 70°F, sigma 3 → high probability of being below
        p = prob_below_threshold(65.0, 70.0, 3.0)
        assert p > 0.9

    def test_prob_below_complement(self):
        # P(below) + P(above) = 1
        above = prob_exceeds_threshold(72.0, 70.0, 3.0)
        below = prob_below_threshold(72.0, 70.0, 3.0)
        assert abs(above + below - 1.0) < 1e-10

    def test_larger_sigma_more_uncertain(self):
        # With larger sigma, the probability should be closer to 0.5
        p_tight = prob_exceeds_threshold(75.0, 70.0, 1.0)
        p_wide = prob_exceeds_threshold(75.0, 70.0, 10.0)
        assert p_tight > p_wide  # tighter sigma → more confident


# ========== Ensemble Model Tests ==========


class TestEnsembleModel:
    def _make_forecast(self, source: str, value: float, horizon: int = 48) -> ForecastPoint:
        return ForecastPoint(
            source=source,
            location="San Francisco",
            target_date=date(2026, 4, 20),
            metric="high_temp",
            value=value,
            forecast_horizon_hours=horizon,
        )

    def _make_params(self, threshold: float = 70.0) -> WeatherMarketParams:
        return WeatherMarketParams(
            location="San Francisco",
            latitude=37.7749,
            longitude=-122.4194,
            metric="high_temp",
            threshold=threshold,
            threshold_direction="above",
            target_date=date(2026, 4, 20),
            market_id="test",
            market_title="test",
        )

    def test_single_source_high_forecast(self):
        model = WeatherEnsembleModel()
        forecasts = [self._make_forecast("open_meteo", 78.0)]
        result = model.predict(forecasts, self._make_params(70.0))
        assert result.probability > 0.9
        assert result.sources_used == 1

    def test_single_source_low_forecast(self):
        model = WeatherEnsembleModel()
        forecasts = [self._make_forecast("open_meteo", 62.0)]
        result = model.predict(forecasts, self._make_params(70.0))
        assert result.probability < 0.1

    def test_multi_source_agreement(self):
        model = WeatherEnsembleModel()
        forecasts = [
            self._make_forecast("open_meteo", 76.0),
            self._make_forecast("nws", 75.0),
            self._make_forecast("visual_crossing", 77.0),
        ]
        result = model.predict(forecasts, self._make_params(70.0))
        assert result.probability > 0.9
        assert result.sources_used == 3
        assert result.confidence > 0.5  # high agreement

    def test_multi_source_disagreement(self):
        model = WeatherEnsembleModel()
        forecasts = [
            self._make_forecast("open_meteo", 75.0),
            self._make_forecast("nws", 62.0),
        ]
        result = model.predict(forecasts, self._make_params(70.0))
        # Should be somewhere in the middle
        assert 0.2 < result.probability < 0.8

    def test_empty_forecasts(self):
        model = WeatherEnsembleModel()
        result = model.predict([], self._make_params())
        assert result.probability == 0.5
        assert result.confidence == 0.0
        assert result.sources_used == 0

    def test_confidence_decreases_with_horizon(self):
        model = WeatherEnsembleModel()
        short = [self._make_forecast("open_meteo", 75.0, horizon=24)]
        long = [self._make_forecast("open_meteo", 75.0, horizon=240)]

        result_short = model.predict(short, self._make_params(70.0))
        result_long = model.predict(long, self._make_params(70.0))
        assert result_short.confidence > result_long.confidence

    def test_source_details_populated(self):
        model = WeatherEnsembleModel()
        forecasts = [self._make_forecast("open_meteo", 74.0)]
        result = model.predict(forecasts, self._make_params(70.0))
        assert len(result.source_details) == 1
        detail = result.source_details[0]
        assert detail["source"] == "open_meteo"
        assert detail["forecast_value"] == 74.0
        assert "sigma" in detail
        assert "probability" in detail
        assert "weight" in detail

    def test_below_threshold_direction(self):
        model = WeatherEnsembleModel()
        forecasts = [self._make_forecast("open_meteo", 15.0)]
        params = WeatherMarketParams(
            location="Chicago",
            latitude=41.8781,
            longitude=-87.6298,
            metric="low_temp",
            threshold=20.0,
            threshold_direction="below",
            target_date=date(2026, 1, 15),
            market_id="test",
            market_title="test",
        )
        result = model.predict(forecasts, params)
        assert result.probability > 0.9  # 15 is below 20, high prob of going below


# ========== Detector Tests ==========


class TestWeatherDetector:
    """Test the detect_weather_ml function with mocked data."""

    def _make_market(
        self, market_id: str, title: str, yes_ask: float = 0.30, no_ask: float = 0.75
    ) -> Market:
        return Market(
            platform="kalshi",
            market_id=market_id,
            title=title,
            category="weather",
            settlement_date=datetime(2026, 4, 20, tzinfo=timezone.utc),
            outcome_count=2,
            outcomes=[
                Outcome(
                    platform="kalshi",
                    market_id=market_id,
                    outcome_id=f"{market_id}-yes",
                    label="Yes",
                    side="yes",
                    best_bid=yes_ask - 0.02,
                    best_ask=yes_ask,
                    bid_depth=100,
                    ask_depth=100,
                    category="weather",
                ),
                Outcome(
                    platform="kalshi",
                    market_id=market_id,
                    outcome_id=f"{market_id}-no",
                    label="No",
                    side="no",
                    best_bid=no_ask - 0.02,
                    best_ask=no_ask,
                    bid_depth=100,
                    ask_depth=100,
                    category="weather",
                ),
            ],
        )

    def test_detect_generates_opportunity_with_edge(self):
        from src.signals.weather.detector import detect_weather_ml

        # Market says 30% chance of exceeding 70°F, but forecast says ~90%
        market = self._make_market(
            "w-1",
            "Will the high temperature in San Francisco exceed 70°F on April 20, 2026?",
            yes_ask=0.30,
        )

        forecasts = {
            "w-1": [
                ForecastPoint("open_meteo", "San Francisco", date(2026, 4, 20), "high_temp", 78.0, forecast_horizon_hours=48),
                ForecastPoint("nws", "San Francisco", date(2026, 4, 20), "high_temp", 77.0, forecast_horizon_hours=48),
            ]
        }

        client = _MockForecastClient(forecasts)
        model = WeatherEnsembleModel()
        fees = KalshiFeeCalculator()

        opps = detect_weather_ml(
            kalshi_markets=[market],
            kalshi_fees=fees,
            forecast_client=client,
            model=model,
            min_edge=0.10,
            min_confidence=0.0,
            max_days_out=30,
            min_sources=2,
        )

        assert len(opps) == 1
        opp = opps[0]
        assert opp.strategy == "weather_ml"
        assert opp.edge > 0.10
        assert opp.model_probability > 0.8
        assert opp.market_price == 0.30
        assert opp.signal_source.startswith("weather_ensemble")
        assert len(opp.legs) == 1
        assert opp.legs[0].platform == "kalshi"

    def test_detect_filters_insufficient_edge(self):
        from src.signals.weather.detector import detect_weather_ml

        # Market says 70%, forecast also says ~70% → no edge
        market = self._make_market(
            "w-2",
            "Will the high temperature in San Francisco exceed 70°F on April 20, 2026?",
            yes_ask=0.70,
        )

        forecasts = {
            "w-2": [
                ForecastPoint("open_meteo", "San Francisco", date(2026, 4, 20), "high_temp", 72.0, forecast_horizon_hours=48),
                ForecastPoint("nws", "San Francisco", date(2026, 4, 20), "high_temp", 71.0, forecast_horizon_hours=48),
            ]
        }

        client = _MockForecastClient(forecasts)
        model = WeatherEnsembleModel()
        fees = KalshiFeeCalculator()

        opps = detect_weather_ml(
            kalshi_markets=[market],
            kalshi_fees=fees,
            forecast_client=client,
            model=model,
            min_edge=0.10,
            min_confidence=0.0,
            max_days_out=30,
            min_sources=2,
        )

        assert len(opps) == 0

    def test_detect_filters_insufficient_sources(self):
        from src.signals.weather.detector import detect_weather_ml

        market = self._make_market(
            "w-3",
            "Will the high temperature in San Francisco exceed 70°F on April 20, 2026?",
            yes_ask=0.30,
        )

        # Only 1 source, but we require 2
        forecasts = {
            "w-3": [
                ForecastPoint("open_meteo", "San Francisco", date(2026, 4, 20), "high_temp", 78.0, forecast_horizon_hours=48),
            ]
        }

        client = _MockForecastClient(forecasts)
        model = WeatherEnsembleModel()
        fees = KalshiFeeCalculator()

        opps = detect_weather_ml(
            kalshi_markets=[market],
            kalshi_fees=fees,
            forecast_client=client,
            model=model,
            min_edge=0.10,
            min_confidence=0.0,
            max_days_out=30,
            min_sources=2,
        )

        assert len(opps) == 0


class _MockForecastClient:
    """Mock forecast client that returns pre-configured data."""

    def __init__(self, data: dict[str, list[ForecastPoint]]):
        self._data = data

    def fetch_all(self, lat, lon, target_date, metric, location=""):
        # Find matching forecasts by looking through all entries
        for market_id, forecasts in self._data.items():
            if forecasts and forecasts[0].target_date == target_date and forecasts[0].metric == metric:
                return forecasts
        return []
