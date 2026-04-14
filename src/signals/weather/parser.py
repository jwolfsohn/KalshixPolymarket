"""Parse Kalshi weather market titles into structured parameters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class WeatherMarketParams:
    """Structured parameters extracted from a Kalshi weather market title."""
    location: str
    latitude: float
    longitude: float
    metric: str  # "high_temp", "low_temp", "precipitation", "wind_speed", "snowfall"
    threshold: float
    threshold_direction: str  # "above" or "below"
    target_date: date
    market_id: str
    market_title: str


# Major US cities that Kalshi typically offers weather markets for
LOCATION_COORDS: dict[str, tuple[float, float]] = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "San Francisco": (37.7749, -122.4194),
    "Miami": (25.7617, -80.1918),
    "Denver": (39.7392, -104.9903),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Seattle": (47.6062, -122.3321),
    "Atlanta": (33.7490, -84.3880),
    "Boston": (42.3601, -71.0589),
    "Dallas": (32.7767, -96.7970),
    "Washington": (38.9072, -77.0369),
    "Philadelphia": (39.9526, -75.1652),
    "Minneapolis": (44.9778, -93.2650),
    "Detroit": (42.3314, -83.0458),
    "Las Vegas": (36.1699, -115.1398),
    "Portland": (45.5152, -122.6784),
    "Nashville": (36.1627, -86.7816),
    "Austin": (30.2672, -97.7431),
    "Charlotte": (35.2271, -80.8431),
    "San Diego": (32.7157, -117.1611),
    "Indianapolis": (39.7684, -86.1581),
    "Columbus": (39.9612, -82.9988),
    "Kansas City": (39.0997, -94.5786),
    "St. Louis": (38.6270, -90.1994),
    "Salt Lake City": (40.7608, -111.8910),
    "New Orleans": (29.9511, -90.0715),
    "Tampa": (27.9506, -82.4572),
    "Orlando": (28.5383, -81.3792),
}

# Patterns for different weather market types
# High temperature: "Will the high temperature in {city} exceed {X}°F on {date}?"
_HIGH_TEMP_PATTERN = re.compile(
    r"(?:will\s+)?(?:the\s+)?high\s+temp(?:erature)?\s+in\s+(.+?)\s+"
    r"(?:exceed|be\s+(?:above|over|at\s+least|greater\s+than))\s+"
    r"(\d+(?:\.\d+)?)\s*°?\s*F",
    re.IGNORECASE,
)

# Low temperature: "Will the low temperature in {city} drop below {X}°F on {date}?"
_LOW_TEMP_PATTERN = re.compile(
    r"(?:will\s+)?(?:the\s+)?low\s+temp(?:erature)?\s+in\s+(.+?)\s+"
    r"(?:drop\s+below|be\s+(?:below|under|less\s+than))\s+"
    r"(\d+(?:\.\d+)?)\s*°?\s*F",
    re.IGNORECASE,
)

# Generic temperature above: "Will it be above {X}°F in {city}..."
_TEMP_ABOVE_PATTERN = re.compile(
    r"(?:will\s+)?(?:it\s+)?(?:be\s+)?(?:above|over|exceed)\s+"
    r"(\d+(?:\.\d+)?)\s*°?\s*F\s+in\s+(.+?)(?:\s+on|\?|$)",
    re.IGNORECASE,
)

# Precipitation: "Will it rain more than {X} inches in {city} on {date}?"
_PRECIP_PATTERN = re.compile(
    r"(?:will\s+)?(?:it\s+)?(?:rain|precipitation)\s+"
    r"(?:more\s+than|exceed|be\s+(?:above|over))\s+"
    r"(\d+(?:\.\d+)?)\s*(?:inches|in\.?)\s+in\s+(.+?)(?:\s+on|\?|$)",
    re.IGNORECASE,
)

# Snowfall: "Will snowfall in {city} exceed {X} inches on {date}?"
_SNOW_PATTERN = re.compile(
    r"(?:will\s+)?(?:the\s+)?snow(?:fall)?\s+in\s+(.+?)\s+"
    r"(?:exceed|be\s+(?:above|over|more\s+than))\s+"
    r"(\d+(?:\.\d+)?)\s*(?:inches|in\.?)",
    re.IGNORECASE,
)

# Wind speed: "Will wind speed in {city} exceed {X} mph on {date}?"
_WIND_PATTERN = re.compile(
    r"(?:will\s+)?(?:the\s+)?wind\s+(?:speed\s+)?in\s+(.+?)\s+"
    r"(?:exceed|be\s+(?:above|over))\s+"
    r"(\d+(?:\.\d+)?)\s*mph",
    re.IGNORECASE,
)

# Date patterns found in market titles
_DATE_PATTERNS = [
    re.compile(r"on\s+(\w+\s+\d{1,2},?\s+\d{4})", re.IGNORECASE),
    re.compile(r"on\s+(\d{1,2}/\d{1,2}/\d{2,4})", re.IGNORECASE),
    re.compile(r"on\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?)", re.IGNORECASE),
    re.compile(r"for\s+(\w+\s+\d{1,2},?\s+\d{4})", re.IGNORECASE),
    re.compile(r"for\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?)", re.IGNORECASE),
]

_DATE_FORMATS = [
    "%B %d, %Y",    # "April 20, 2026"
    "%B %d %Y",     # "April 20 2026"
    "%b %d, %Y",    # "Apr 20, 2026"
    "%b %d %Y",     # "Apr 20 2026"
    "%m/%d/%Y",     # "4/20/2026"
    "%m/%d/%y",     # "4/20/26"
    "%B %d",        # "April 20" (assume current year)
    "%b %d",        # "Apr 20"
]


def _resolve_location(raw: str) -> tuple[str, float, float] | None:
    """Match a raw location string to known coordinates."""
    cleaned = raw.strip().rstrip(",")
    for name, (lat, lon) in LOCATION_COORDS.items():
        if name.lower() in cleaned.lower():
            return name, lat, lon
    return None


def _parse_date(title: str) -> date | None:
    """Extract a date from a market title."""
    for pattern in _DATE_PATTERNS:
        m = pattern.search(title)
        if not m:
            continue
        raw_date = m.group(1).strip().rstrip(",")
        # Strip ordinal suffixes
        raw_date = re.sub(r"(\d+)(?:st|nd|rd|th)", r"\1", raw_date)
        for fmt in _DATE_FORMATS:
            try:
                dt = datetime.strptime(raw_date, fmt)
                if dt.year == 1900:  # no year in format
                    dt = dt.replace(year=datetime.now().year)
                return dt.date()
            except ValueError:
                continue
    return None


def parse_weather_market(
    market_id: str, title: str, category: str = ""
) -> WeatherMarketParams | None:
    """Extract structured weather parameters from a Kalshi market title.

    Returns None if the market is not a parseable weather market.
    """
    if category and category.lower() not in ("weather", "climate", ""):
        return None

    target_date = _parse_date(title)
    if target_date is None:
        return None

    # Try each pattern in order
    for pattern, metric, direction, loc_idx, val_idx in [
        (_HIGH_TEMP_PATTERN, "high_temp", "above", 1, 2),
        (_LOW_TEMP_PATTERN, "low_temp", "below", 1, 2),
        (_TEMP_ABOVE_PATTERN, "high_temp", "above", 2, 1),
        (_PRECIP_PATTERN, "precipitation", "above", 2, 1),
        (_SNOW_PATTERN, "snowfall", "above", 1, 2),
        (_WIND_PATTERN, "wind_speed", "above", 1, 2),
    ]:
        m = pattern.search(title)
        if not m:
            continue
        raw_location = m.group(loc_idx)
        threshold = float(m.group(val_idx))

        loc = _resolve_location(raw_location)
        if loc is None:
            continue

        name, lat, lon = loc
        return WeatherMarketParams(
            location=name,
            latitude=lat,
            longitude=lon,
            metric=metric,
            threshold=threshold,
            threshold_direction=direction,
            target_date=target_date,
            market_id=market_id,
            market_title=title,
        )

    return None
