"""Static reference data and model tuning knobs.

Everything a user might want to tweak lives here, so the model can be
re-tuned (e.g. after running `backtest`) without touching the logic.
"""

# Home stadium per ESPN team abbreviation: (lat, lon, roof)
# roof: "open", "dome" (fixed roof), "retractable" (treated as closed in bad weather)
STADIUMS = {
    "ARI": (33.5277, -112.2626, "retractable"),
    "ATL": (33.7554, -84.4008, "retractable"),
    "BAL": (39.2780, -76.6227, "open"),
    "BUF": (42.7738, -78.7870, "open"),
    "CAR": (35.2258, -80.8528, "open"),
    "CHI": (41.8623, -87.6167, "open"),
    "CIN": (39.0955, -84.5161, "open"),
    "CLE": (41.5061, -81.6995, "open"),
    "DAL": (32.7473, -97.0945, "retractable"),
    "DEN": (39.7439, -105.0201, "open"),
    "DET": (42.3400, -83.0456, "dome"),
    "GB": (44.5013, -88.0622, "open"),
    "HOU": (29.6847, -95.4107, "retractable"),
    "IND": (39.7601, -86.1639, "retractable"),
    "JAX": (30.3239, -81.6373, "open"),
    "KC": (39.0489, -94.4839, "open"),
    "LV": (36.0909, -115.1833, "dome"),
    "LAC": (33.9535, -118.3392, "dome"),
    "LAR": (33.9535, -118.3392, "dome"),
    "MIA": (25.9580, -80.2389, "open"),
    "MIN": (44.9737, -93.2577, "dome"),
    "NE": (42.0909, -71.2643, "open"),
    "NO": (29.9511, -90.0812, "dome"),
    "NYG": (40.8135, -74.0745, "open"),
    "NYJ": (40.8135, -74.0745, "open"),
    "PHI": (39.9008, -75.1675, "open"),
    "PIT": (40.4468, -80.0158, "open"),
    "SF": (37.4030, -121.9700, "open"),
    "SEA": (47.5952, -122.3316, "open"),
    "TB": (27.9759, -82.5033, "open"),
    "TEN": (36.1665, -86.7713, "open"),
    "WSH": (38.9076, -76.8645, "open"),
}

# ---- Injury handling -------------------------------------------------------
# Probability a player suits up, by ESPN status text (matched case-insensitively,
# first match wins). Questionable players play in ~75% of simulations, etc.
PLAY_PROB = [
    ("injured reserve", 0.0),
    ("physically unable", 0.0),
    ("non-football", 0.0),
    ("suspen", 0.0),
    ("out", 0.0),
    ("doubtful", 0.15),
    ("questionable", 0.75),
    ("day-to-day", 0.90),
    ("probable", 0.95),
]

RETAIN_VOLUME = 0.85       # share of a missing player's yards absorbed by teammates
BACKUP_QB_EFF = 0.80       # backup QB passing yards vs starter's
QB_OUT_REC_MULT = 0.88     # receivers' yards when the starting QB sits
QB_OUT_PTS_MULT = 0.86     # team points when the starting QB sits
SKILL_LOSS_PTS = 0.25      # points lost per 100% of skill-position yards missing
DEF_OUT_PTS = 0.35         # opponent points added per defender ruled Out/Doubtful (this week only)
DEF_OUT_CAP = 2.0
DEF_POSITIONS = {"CB", "S", "FS", "SS", "LB", "ILB", "OLB", "MLB", "DE", "DT", "NT", "EDGE", "DL", "DB"}
STALE_WEEKS = 2            # a player absent from his team's last N games is treated as already priced in

# ---- Team strength ---------------------------------------------------------
SHRINK_GAMES = 3           # early-season regression: weight = gp / (gp + SHRINK_GAMES)
HOME_FIELD_PTS = 0.8       # added to home, subtracted from away (≈1.6 pt spread)
RECENCY_DECAY = 0.85       # player average weights: most recent game 1.0, previous 0.85, ...

# ---- Weather ---------------------------------------------------------------
WIND_START_MPH = 10        # passing starts to suffer above this sustained wind
WIND_PASS_PER_MPH = 0.012  # passing yards lost per mph above threshold
WIND_PTS_PER_MPH = 0.006
WIND_FLOOR = 0.75
COLD_F = 25                # below this, small passing/points penalty
COLD_PASS_MULT = 0.96
RAIN_PROB = 60             # % chance, and...
RAIN_INCH = 0.04           # ...inches/hour to count as a wet game
RAIN_PASS_MULT = 0.94
RAIN_RUSH_MULT = 1.03
RAIN_PTS_MULT = 0.96

# ---- Simulation noise ------------------------------------------------------
N_SIMS = 20000
TEAM_FORM_SD = 0.25        # lognormal sd of a team's game-level offensive "form"
PACE_SD = 0.06             # shared game pace (correlates both offenses)
SCORE_NOISE_SD = 7.0
PLAYER_FORM_BETA = 0.6     # how strongly player yards follow team form (drives SGP correlation)
DEFAULT_CV = {"pass_yds": 0.28, "rush_yds": 0.50, "rec_yds": 0.60}
CV_BOUNDS = (0.18, 1.1)

STATS = ("pass_yds", "rush_yds", "rec_yds")
