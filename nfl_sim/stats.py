"""Season data: player weekly logs and team offense/defense rankings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, espn

PLAYER_COLS = ["game_id", "week", "team_id", "opp_id", "player_id", "name", "pos",
               "pass_yds", "pass_att", "rush_yds", "rush_att", "rec_yds", "rec", "tgt"]
TEAM_COLS = ["game_id", "week", "team_id", "abbr", "opp_id", "home", "pts", "pa",
             "pass_yds", "rush_yds", "opp_pass_yds", "opp_rush_yds"]


@dataclass
class SeasonData:
    players: pd.DataFrame   # one row per player per game
    teams: pd.DataFrame     # one row per team per game

    def before(self, week: int | None) -> "SeasonData":
        if week is None:
            return self
        return SeasonData(self.players[self.players.week < week], self.teams[self.teams.week < week])


def load_season(season: int, through_week: int, seasontype: int = 2, verbose: bool = True) -> SeasonData:
    prow, trow = [], []
    for wk in range(1, through_week + 1):
        games = [g for g in espn.get_week_games(season, wk, seasontype) if g.completed]
        if verbose:
            print(f"  week {wk}: {len(games)} completed games", flush=True)
        for g in games:
            p, t = espn.parse_boxscore(espn.get_summary(g.id, completed=True), g)
            prow += p
            trow += t
    return SeasonData(pd.DataFrame(prow, columns=PLAYER_COLS), pd.DataFrame(trow, columns=TEAM_COLS))


# --------------------------------------------------------------------------- teams
def team_table(data: SeasonData) -> pd.DataFrame:
    """Per-team offense & defense, raw and opponent-adjusted, with 1-32 ranks.

    Opponent adjustment: each game's output is divided by what that opponent
    normally allows (or gains), then scaled back to league average. So 300 pass
    yards against the league's best pass defense counts for more than 300
    against the worst.
    """
    tg = data.teams
    if tg.empty:
        return pd.DataFrame()
    t = tg.groupby("team_id").agg(
        abbr=("abbr", "last"), gp=("week", "count"),
        ppg=("pts", "mean"), pa_pg=("pa", "mean"),
        pass_ypg=("pass_yds", "mean"), rush_ypg=("rush_yds", "mean"),
        pass_allowed_pg=("opp_pass_yds", "mean"), rush_allowed_pg=("opp_rush_yds", "mean"),
    )
    lg = {"pts": tg.pts.mean(), "pass": tg.pass_yds.mean(), "rush": tg.rush_yds.mean()}

    opp = t[["ppg", "pa_pg", "pass_ypg", "rush_ypg", "pass_allowed_pg", "rush_allowed_pg"]].add_prefix("o_")
    g = tg.join(opp, on="opp_id")
    eps = 1e-6
    g["off_pts_r"] = g.pts / (g.o_pa_pg + eps)
    g["off_pass_r"] = g.pass_yds / (g.o_pass_allowed_pg + eps)
    g["off_rush_r"] = g.rush_yds / (g.o_rush_allowed_pg + eps)
    g["def_pts_r"] = g.pa / (g.o_ppg + eps)
    g["def_pass_r"] = g.opp_pass_yds / (g.o_pass_ypg + eps)
    g["def_rush_r"] = g.opp_rush_yds / (g.o_rush_ypg + eps)
    adj = g.groupby("team_id")[[c for c in g.columns if c.endswith("_r")]].mean()
    t["adj_off_pts"] = adj.off_pts_r * lg["pts"]
    t["adj_off_pass"] = adj.off_pass_r * lg["pass"]
    t["adj_off_rush"] = adj.off_rush_r * lg["rush"]
    t["adj_def_pts"] = adj.def_pts_r * lg["pts"]
    t["adj_def_pass"] = adj.def_pass_r * lg["pass"]
    t["adj_def_rush"] = adj.def_rush_r * lg["rush"]

    last4 = tg.sort_values("week").groupby("team_id").tail(4).groupby("team_id")
    t["l4_ppg"] = last4.pts.mean()
    t["l4_pa_pg"] = last4.pa.mean()

    # Ranks: offense 1 = most, defense 1 = fewest allowed
    for col in ["adj_off_pts", "adj_off_pass", "adj_off_rush"]:
        t[col.replace("adj_", "rk_")] = t[col].rank(ascending=False, method="min").astype(int)
    for col in ["adj_def_pts", "adj_def_pass", "adj_def_rush"]:
        t[col.replace("adj_", "rk_")] = t[col].rank(ascending=True, method="min").astype(int)
    t.attrs["league"] = lg
    return t


# --------------------------------------------------------------------------- players
def player_table(data: SeasonData) -> pd.DataFrame:
    """Per-player weekly averages (straight, last-3, recency-weighted) and volatility."""
    pg = data.players
    if pg.empty:
        return pd.DataFrame()
    pg = pg.sort_values("week")
    team_last_week = data.teams.groupby("team_id").week.max()
    team_games = data.teams.groupby("team_id").week.apply(lambda s: sorted(s))

    rows = []
    for pid, g in pg.groupby("player_id"):
        tid = g.team_id.iloc[-1]
        n = len(g)
        w = config.RECENCY_DECAY ** np.arange(n)[::-1]
        r = {"player_id": pid, "name": g.name.iloc[-1], "pos": g.pos.iloc[-1], "team_id": tid,
             "gp": n, "last_week": int(g.week.iloc[-1]),
             "pass_att_pg": g.pass_att.mean(), "rush_att_pg": g.rush_att.mean(), "tgt_pg": g.tgt.mean()}
        tw = team_games.get(tid, [])
        recent = tw[-config.STALE_WEEKS:] if len(tw) else []
        r["missed_recent"] = bool(recent) and not set(recent) & set(g.week)
        for s in config.STATS:
            v = g[s].to_numpy(dtype=float)
            r[f"{s}_avg"] = v.mean()
            r[f"{s}_l3"] = v[-3:].mean()
            r[f"{s}_wavg"] = float((v * w).sum() / w.sum())
            r[f"{s}_cv"] = float(v.std() / v.mean()) if n >= 3 and v.mean() > 1 else np.nan
        rows.append(r)
    out = pd.DataFrame(rows).set_index("player_id")
    if (out.pos == "").any():  # infer position when ESPN omits it
        inferred = np.where(out.pass_att_pg >= 8, "QB", np.where(out.rush_att_pg > out.tgt_pg, "RB", "WR/TE"))
        out["pos"] = np.where(out.pos == "", inferred, out.pos)
    out["team_last_week"] = out.team_id.map(team_last_week)
    return out
