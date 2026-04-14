"""Gaussian ensemble model for weather probability estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.signals.weather.forecast_client import ForecastPoint
from src.signals.weather.parser import WeatherMarketParams


def normal_cdf(x: float) -> float:
    """Standard normal CDF using the complementary error function."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def prob_exceeds_threshold(
    forecast_value: float, threshold: float, sigma: float
) -> float:
    """P(actual > threshold) given a Gaussian forecast distribution.

    Models the actual weather value as N(forecast_value, sigma^2).
    """
    if sigma <= 0:
        return 1.0 if forecast_value > threshold else 0.0
    z = (threshold - forecast_value) / sigma
    return 1.0 - normal_cdf(z)


def prob_below_threshold(
    forecast_value: float, threshold: float, sigma: float
) -> float:
    """P(actual < threshold) given a Gaussian forecast distribution."""
    if sigma <= 0:
        return 1.0 if forecast_value < threshold else 0.0
    z = (threshold - forecast_value) / sigma
    return normal_cdf(z)


# Default uncertainty (sigma) by metric and forecast horizon
# These increase with forecast horizon because forecasts degrade over time
_DEFAULT_SIGMAS: dict[str, tuple[float, float]] = {
    # (base_sigma, per_day_increase)
    "high_temp": (2.0, 0.5),
    "low_temp": (2.0, 0.5),
    "precipitation": (0.3, 0.1),
    "snowfall": (1.0, 0.3),
    "wind_speed": (3.0, 0.8),
}


def _estimate_sigma(
    metric: str,
    horizon_hours: int,
    config_temp_sigma: float = 3.0,
    config_precip_sigma: float = 0.5,
) -> float:
    """Estimate forecast uncertainty based on metric type and horizon."""
    days_out = max(0, horizon_hours / 24)

    if metric in _DEFAULT_SIGMAS:
        base, per_day = _DEFAULT_SIGMAS[metric]
    elif "temp" in metric:
        base, per_day = config_temp_sigma, 0.5
    elif metric in ("precipitation", "snowfall"):
        base, per_day = config_precip_sigma, 0.1
    else:
        base, per_day = 3.0, 0.5

    return base + per_day * days_out


@dataclass
class ProbabilityEstimate:
    """Result of the ensemble probability estimation."""
    probability: float  # 0-1, P(threshold condition is met)
    confidence: float   # 0-1, how confident we are in this estimate
    sources_used: int
    source_details: list[dict]


class WeatherEnsembleModel:
    """Estimate probability of a weather threshold being exceeded.

    Uses a weighted ensemble of multiple forecast sources. Each source's
    point forecast is modeled as a Gaussian distribution, and the
    probability of exceeding the threshold is computed via the normal CDF.
    """

    DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
        "open_meteo": 0.45,
        "nws": 0.35,
        "visual_crossing": 0.20,
    }

    def __init__(
        self,
        source_weights: dict[str, float] | None = None,
        default_temp_sigma: float = 3.0,
        default_precip_sigma: float = 0.5,
    ):
        self.source_weights = source_weights or dict(self.DEFAULT_SOURCE_WEIGHTS)
        self.default_temp_sigma = default_temp_sigma
        self.default_precip_sigma = default_precip_sigma

    def predict(
        self, forecasts: list[ForecastPoint], params: WeatherMarketParams
    ) -> ProbabilityEstimate:
        """Compute ensemble probability that the threshold condition is met.

        For "above" markets: P(actual > threshold)
        For "below" markets: P(actual < threshold)
        """
        if not forecasts:
            return ProbabilityEstimate(
                probability=0.5, confidence=0.0, sources_used=0, source_details=[]
            )

        details = []
        weighted_prob = 0.0
        total_weight = 0.0

        for fp in forecasts:
            sigma = _estimate_sigma(
                fp.metric,
                fp.forecast_horizon_hours,
                self.default_temp_sigma,
                self.default_precip_sigma,
            )

            if params.threshold_direction == "below":
                p = prob_below_threshold(fp.value, params.threshold, sigma)
            else:
                p = prob_exceeds_threshold(fp.value, params.threshold, sigma)

            weight = self.source_weights.get(fp.source, 0.1)
            weighted_prob += p * weight
            total_weight += weight

            details.append({
                "source": fp.source,
                "forecast_value": fp.value,
                "sigma": sigma,
                "probability": p,
                "weight": weight,
                "horizon_hours": fp.forecast_horizon_hours,
            })

        ensemble_prob = weighted_prob / total_weight if total_weight > 0 else 0.5

        # Confidence is higher when:
        # 1. More sources agree
        # 2. Forecast horizon is shorter
        # 3. Sources are close to each other
        probs = [d["probability"] for d in details]
        spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
        avg_horizon = sum(d["horizon_hours"] for d in details) / len(details)
        horizon_factor = max(0.2, 1.0 - avg_horizon / (10 * 24))  # decays over 10 days
        agreement_factor = max(0.2, 1.0 - spread)
        source_factor = min(1.0, len(forecasts) / 3.0)

        confidence = horizon_factor * agreement_factor * source_factor

        return ProbabilityEstimate(
            probability=ensemble_prob,
            confidence=confidence,
            sources_used=len(forecasts),
            source_details=details,
        )
