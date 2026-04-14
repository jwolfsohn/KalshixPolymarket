"""Multi-source weather forecast fetching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# Mapping from our metric names to Open-Meteo daily variable names
_OPEN_METEO_VARS: dict[str, str] = {
    "high_temp": "temperature_2m_max",
    "low_temp": "temperature_2m_min",
    "precipitation": "precipitation_sum",
    "snowfall": "snowfall_sum",
    "wind_speed": "wind_speed_10m_max",
}

# Mapping from our metric names to NWS forecast keys
_NWS_METRIC_MAP: dict[str, str] = {
    "high_temp": "maxTemperature",
    "low_temp": "minTemperature",
}


@dataclass
class ForecastPoint:
    """A single forecast data point from one source."""
    source: str
    location: str
    target_date: date
    metric: str
    value: float  # predicted value in native units (°F, inches, mph)
    value_min: float | None = None
    value_max: float | None = None
    fetched_at: datetime | None = None
    forecast_horizon_hours: int = 0


class WeatherForecastClient:
    """Fetch weather forecasts from multiple free APIs."""

    OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
    NWS_POINTS_BASE = "https://api.weather.gov/points"
    VISUAL_CROSSING_BASE = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices"
        "/rest/services/timeline"
    )

    def __init__(
        self,
        visual_crossing_api_key: str = "",
        httpx_client: httpx.Client | None = None,
    ):
        self.vc_key = visual_crossing_api_key
        self._client = httpx_client or httpx.Client(timeout=15.0)
        self._owns_client = httpx_client is None

    def close(self):
        if self._owns_client:
            self._client.close()

    def fetch_open_meteo(
        self, lat: float, lon: float, target_date: date, metric: str, location: str = ""
    ) -> ForecastPoint | None:
        """Open-Meteo: free, no API key, global, up to 16 days out."""
        var_name = _OPEN_METEO_VARS.get(metric)
        if not var_name:
            return None

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": var_name,
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "wind_speed_unit": "mph",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "timezone": "auto",
        }

        try:
            resp = self._client.get(self.OPEN_METEO_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            values = daily.get(var_name, [])
            if not values or values[0] is None:
                return None

            now = datetime.now(timezone.utc)
            target_dt = datetime(
                target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc
            )
            horizon = max(0, int((target_dt - now).total_seconds() / 3600))

            return ForecastPoint(
                source="open_meteo",
                location=location,
                target_date=target_date,
                metric=metric,
                value=float(values[0]),
                fetched_at=now,
                forecast_horizon_hours=horizon,
            )
        except Exception as e:
            logger.warning(f"Open-Meteo fetch failed: {e}")
            return None

    def fetch_nws(
        self, lat: float, lon: float, target_date: date, metric: str, location: str = ""
    ) -> ForecastPoint | None:
        """NWS/Weather.gov: free, US-only, 7-day forecast."""
        nws_key = _NWS_METRIC_MAP.get(metric)
        if not nws_key:
            return None

        try:
            # Step 1: Get the forecast grid URL for this location
            points_resp = self._client.get(
                f"{self.NWS_POINTS_BASE}/{lat:.4f},{lon:.4f}",
                headers={"User-Agent": "KalshiPolymarketBot/1.0"},
            )
            points_resp.raise_for_status()
            grid_url = points_resp.json()["properties"]["forecastGridData"]

            # Step 2: Get the gridpoint forecast data
            grid_resp = self._client.get(
                grid_url,
                headers={"User-Agent": "KalshiPolymarketBot/1.0"},
            )
            grid_resp.raise_for_status()
            props = grid_resp.json()["properties"]

            series = props.get(nws_key, {}).get("values", [])
            target_str = target_date.isoformat()

            for entry in series:
                # NWS uses ISO 8601 duration ranges like "2026-04-20T06:00:00+00:00/PT12H"
                valid_time = entry.get("validTime", "")
                if target_str in valid_time:
                    value_c = entry["value"]
                    if value_c is None:
                        continue
                    # NWS returns Celsius, convert to Fahrenheit
                    value_f = value_c * 9 / 5 + 32

                    now = datetime.now(timezone.utc)
                    target_dt = datetime(
                        target_date.year, target_date.month, target_date.day,
                        tzinfo=timezone.utc,
                    )
                    horizon = max(0, int((target_dt - now).total_seconds() / 3600))

                    return ForecastPoint(
                        source="nws",
                        location=location,
                        target_date=target_date,
                        metric=metric,
                        value=value_f,
                        fetched_at=now,
                        forecast_horizon_hours=horizon,
                    )

            return None
        except Exception as e:
            logger.warning(f"NWS fetch failed: {e}")
            return None

    def fetch_visual_crossing(
        self, lat: float, lon: float, target_date: date, metric: str, location: str = ""
    ) -> ForecastPoint | None:
        """Visual Crossing: free tier (1000 calls/day), global coverage."""
        if not self.vc_key:
            return None

        vc_metric_map = {
            "high_temp": "tempmax",
            "low_temp": "tempmin",
            "precipitation": "precip",
            "snowfall": "snow",
            "wind_speed": "windspeed",
        }
        vc_key = vc_metric_map.get(metric)
        if not vc_key:
            return None

        try:
            url = f"{self.VISUAL_CROSSING_BASE}/{lat},{lon}/{target_date.isoformat()}"
            resp = self._client.get(
                url,
                params={
                    "unitGroup": "us",
                    "key": self.vc_key,
                    "include": "days",
                    "contentType": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            days = data.get("days", [])
            if not days:
                return None

            value = days[0].get(vc_key)
            if value is None:
                return None

            now = datetime.now(timezone.utc)
            target_dt = datetime(
                target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc
            )
            horizon = max(0, int((target_dt - now).total_seconds() / 3600))

            return ForecastPoint(
                source="visual_crossing",
                location=location,
                target_date=target_date,
                metric=metric,
                value=float(value),
                fetched_at=now,
                forecast_horizon_hours=horizon,
            )
        except Exception as e:
            logger.warning(f"Visual Crossing fetch failed: {e}")
            return None

    def fetch_all(
        self, lat: float, lon: float, target_date: date, metric: str, location: str = ""
    ) -> list[ForecastPoint]:
        """Fetch from all available sources, return all successful results."""
        results: list[ForecastPoint] = []

        for fetcher in [self.fetch_open_meteo, self.fetch_nws, self.fetch_visual_crossing]:
            point = fetcher(lat, lon, target_date, metric, location)
            if point is not None:
                results.append(point)

        return results
