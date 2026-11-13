"""NBA season data: player per-minute rates and team pace / ratings."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config, espn

PLAYER_COLS = ["game_id", "day", "team_id", "opp_id", "player_id", "name", "pos", "starter",
               "min", "pts", "reb", "ast", "fg3"]
TEAM_COLS = ["game_id", "day", "team_id", "abbr", "opp_id", "home", "pts", "pa", "poss",
             "reb", "ast", "fg3", "opp_reb", "opp_ast", "opp_fg3"]


@dataclass
class SeasonData:
    players: pd.DataFrame
    teams: pd.DataFrame

    def before(self, day: date | None) -> "SeasonData":
        if day is None:
            return self
        d = day.isoformat()
        return SeasonData(self.players[self.players.day < d], self.teams[self.teams.day < d])


def load_season(through: date, verbose=True) -> SeasonData:
    start = espn.season_start()
    prow, trow, day = [], [], start
    while day <= through:
        games = [g for g in espn.games_on(day) if g.completed and g.season_type == 2]
        for g in games:
            p, t = espn.parse_boxscore(espn.get_summary(g.id, completed=True), g)
            prow += p
            trow += t
        if verbose and games:
            print(f"  {day}: {len(games)} games", flush=True)
        day += timedelta(days=1)
    return SeasonData(pd.DataFrame(prow, columns=PLAYER_COLS), pd.DataFrame(trow, columns=TEAM_COLS))


def team_table(data: SeasonData) -> pd.DataFrame:
    tg = data.teams.copy()
    if tg.empty:
        return pd.DataFrame()
    tg["ortg"] = 100 * tg.pts / tg.poss
    tg["drtg"] = 100 * tg.pa / tg.poss
    t = tg.groupby("team_id").agg(
        abbr=("abbr", "last"), gp=("pts", "count"), ppg=("pts", "mean"), pa_pg=("pa", "mean"),
        pace=("poss", "mean"), ortg=("ortg", "mean"), drtg=("drtg", "mean"),
        reb_allowed=("opp_reb", "mean"), ast_allowed=("opp_ast", "mean"), fg3_allowed=("opp_fg3", "mean"),
        last_day=("day", "max"),
    )
    lg = {"pace": tg.poss.mean(), "ortg": tg.ortg.mean(), "reb": tg.reb.mean(), "ast": tg.ast.mean(),
          "fg3": tg.fg3.mean(), "pts": tg.pts.mean()}
    # opponent adjustment: offense vs the defenses faced, defense vs the offenses faced
    g = tg.join(t[["ortg", "drtg"]].add_prefix("o_"), on="opp_id")
    t["adj_ortg"] = (g.ortg / g.o_drtg).groupby(g.team_id).mean() * lg["ortg"]
    t["adj_drtg"] = (g.drtg / g.o_ortg).groupby(g.team_id).mean() * lg["ortg"]
    t["net"] = t.adj_ortg - t.adj_drtg
    l10 = tg.sort_values("day").groupby("team_id").tail(10).groupby("team_id")
    t["l10_net"] = (l10.ortg.mean() - l10.drtg.mean())
    t["rk_off"] = t.adj_ortg.rank(ascending=False, method="min").astype(int)
    t["rk_def"] = t.adj_drtg.rank(ascending=True, method="min").astype(int)
    t["rk_net"] = t.net.rank(ascending=False, method="min").astype(int)
    t["rk_pace"] = t.pace.rank(ascending=False, method="min").astype(int)
    t.attrs["league"] = lg
    t.attrs["days"] = tg.groupby("team_id").day.apply(lambda s: sorted(set(s))).to_dict()
    return t


def player_table(data: SeasonData) -> pd.DataFrame:
    pg = data.players.sort_values("day")
    if pg.empty:
        return pd.DataFrame()
    team_days = data.teams.groupby("team_id").day.apply(lambda s: sorted(set(s))).to_dict()
    rows = []
    for pid, g in pg.groupby("player_id"):
        n = len(g)
        tid = g.team_id.iloc[-1]
        age = np.arange(n)[::-1]
        wr = config.RATE_DECAY ** age
        wm = config.MINUTES_DECAY ** age
        mins = g["min"].to_numpy(float)
        r = {"player_id": pid, "name": g.name.iloc[-1], "pos": g.pos.iloc[-1], "team_id": tid, "gp": n,
             "last_day": g.day.iloc[-1], "min_avg": mins.mean(), "min_w": float((mins * wm).sum() / wm.sum()),
             "starter_share": g.starter.mean()}
        recent = team_days.get(tid, [])[-config.STALE_GAMES:]
        r["missed_recent"] = bool(recent) and not set(recent) & set(g.day)
        for s in config.STATS:
            v = g[s].to_numpy(float)
            r[f"{s}_avg"] = v.mean()
            r[f"{s}_l5"] = v[-5:].mean()
            r[f"{s}_rate"] = float((v * wr).sum() / max((mins * wr).sum(), 1e-6))   # per minute
            r[f"{s}_cv"] = float(v.std() / v.mean()) if n >= 4 and v.mean() > 0.5 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("player_id")
