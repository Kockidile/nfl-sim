"""Venue location + kickoff-window weather (Open-Meteo, free, no API key)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import config
from .espn import Game, get_json

FORECAST = "https://api.open-meteo.com/v1/forecast"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"


@dataclass
class Weather:
    indoor: bool
    temp_f: float | None = None
    wind_mph: float | None = None
    gust_mph: float | None = None
    precip_prob: float | None = None
    precip_in: float | None = None
    source: str = ""

    def describe(self) -> str:
        if self.indoor:
            return "Indoors / roof — weather neutral"
        if self.temp_f is None:
            return f"Outdoors — no forecast yet ({self.source})"
        return (f"{self.temp_f:.0f}°F, wind {self.wind_mph:.0f} mph (gusts {self.gust_mph:.0f}), "
                f"{self.precip_prob:.0f}% precip, {self.precip_in:.2f} in/hr  [{self.source}]")

    # multipliers used by the model ------------------------------------
    def multipliers(self) -> dict[str, float]:
        m = {"pass_yds": 1.0, "rush_yds": 1.0, "rec_yds": 1.0, "pts": 1.0}
        if self.indoor or self.temp_f is None:
            return m
        over = max(0.0, (self.wind_mph or 0) - config.WIND_START_MPH)
        m["pass_yds"] *= max(config.WIND_FLOOR, 1 - config.WIND_PASS_PER_MPH * over)
        m["pts"] *= max(config.WIND_FLOOR, 1 - config.WIND_PTS_PER_MPH * over)
        if self.temp_f < config.COLD_F:
            m["pass_yds"] *= config.COLD_PASS_MULT
            m["pts"] *= 0.98
        if (self.precip_prob or 0) >= config.RAIN_PROB and (self.precip_in or 0) >= config.RAIN_INCH:
            m["pass_yds"] *= config.RAIN_PASS_MULT
            m["rush_yds"] *= config.RAIN_RUSH_MULT
            m["pts"] *= config.RAIN_PTS_MULT
        m["rec_yds"] = m["pass_yds"]
        return m


def venue_location(game: Game) -> tuple[float | None, float | None, str]:
    """(lat, lon, roof). Neutral-site / international games are geocoded from ESPN's city."""
    home = config.STADIUMS.get(game.home.abbr)
    if home and not game.neutral:
        return home
    if game.city:
        res = get_json(GEOCODE, {"name": game.city, "count": 5}, ttl=None).get("results") or []
        pick = next((r for r in res if game.state and game.state in (r.get("admin1", ""), r.get("country_code", ""))), None)
        pick = pick or (res[0] if res else None)
        if pick:
            roof = "dome" if game.indoor else "open"
            return pick["latitude"], pick["longitude"], roof
    return home if home else (None, None, "open")


def game_weather(game: Game) -> Weather:
    lat, lon, roof = venue_location(game)
    indoor = bool(game.indoor) if game.indoor is not None else roof != "open"
    # Retractable roofs: ESPN usually reports indoor=True when closed; otherwise
    # assume the roof closes for bad weather, so treat as neutral.
    if roof in ("dome", "retractable") or indoor:
        return Weather(indoor=True, source="roof")
    if lat is None:
        return Weather(indoor=False, source="unknown venue")

    kick = datetime.fromisoformat(game.date.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if kick - now > timedelta(days=15):
        return Weather(indoor=False, source="beyond 16-day forecast window")
    params = {
        "latitude": lat, "longitude": lon, "timezone": "GMT",
        "hourly": "temperature_2m,precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
        "past_days": 7, "forecast_days": 16,
    }
    d = get_json(FORECAST, params, ttl=3 * 3600).get("hourly") or {}
    times = d.get("time") or []
    window = [i for i, t in enumerate(times)
              if 0 <= (datetime.fromisoformat(t).replace(tzinfo=timezone.utc) - kick).total_seconds() < 3.5 * 3600]
    if not window:
        return Weather(indoor=False, source="no matching forecast hours")

    def avg(key):
        vals = [d[key][i] for i in window if d.get(key) and d[key][i] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return Weather(indoor=False, temp_f=avg("temperature_2m"), wind_mph=avg("wind_speed_10m"),
                   gust_mph=avg("wind_gusts_10m"), precip_prob=avg("precipitation_probability"),
                   precip_in=avg("precipitation"), source="Open-Meteo kickoff window")
