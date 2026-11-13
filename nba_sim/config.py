"""NBA model tuning knobs (same idea as nfl_sim/config.py)."""

STATS = ("pts", "reb", "ast", "fg3")

# Probability a player suits up, by ESPN status text (first match wins)
PLAY_PROB = [
    ("out for season", 0.0),
    ("out", 0.0),
    ("suspen", 0.0),
    ("doubtful", 0.20),
    ("questionable", 0.55),
    ("game time", 0.55),
    ("day-to-day", 0.65),
    ("probable", 0.93),
]

STALE_GAMES = 3            # missed the team's last N games -> injured ones are "priced in"
MIN_MINUTES = 10           # rotation threshold (recent minutes per game)
MAX_PLAYERS = 11

SHRINK_GAMES = 12          # team-rating regression toward league average (early-season ratings are noisy)
HOME_COURT_PTS = 1.4       # added to home, subtracted from away (≈2.8-pt spread)
B2B_PTS = 1.5              # second night of a back-to-back
MATCHUP_WEIGHT = 0.5       # how much of an opponent's stat-allowed factor to apply (0..1)

RATE_DECAY = 0.90          # per-minute production: recent games weigh more
MINUTES_DECAY = 0.75       # minutes: recent role matters most

RETAIN = 0.85              # share of a missing player's production absorbed by teammates
STAR_LOSS_PTS = 0.30       # team points lost per 100% of scoring missing (bench covers the rest)

# Simulation noise
N_SIMS = 20000
TEAM_FORM_SD = 0.05        # team-level shooting night (lognormal)
PACE_SD = 0.04             # shared game pace
SCORE_NOISE_SD = 7.0
PLAYER_FORM_BETA = 0.8     # player points follow team form (SGP correlation)
BLOWOUT_START = 12         # margin where starters start losing minutes
BLOWOUT_PER_PT = 0.012
BLOWOUT_FLOOR = 0.78
STARTER_MIN = 28           # players at or above this many minutes are hurt by blowouts
DEFAULT_CV = {"pts": 0.32, "reb": 0.40, "ast": 0.45, "fg3": 0.65}
CV_BOUNDS = (0.18, 1.2)
MARGIN_SD = 12.5           # for quick win-probability summaries
