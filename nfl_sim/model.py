"""Matchup construction + Monte Carlo game simulation.

Pipeline per game
-----------------
1. Team strength: opponent-adjusted offense vs opponent's opponent-adjusted
   defense (passing, rushing, points), regressed to league average early season.
2. Matchup factor: expected team yards / team's season average, applied to each
   player's recency-weighted average.
3. Weather multipliers (wind, cold, rain) and home field.
4. Injuries: every simulation samples who plays (Questionable ≈ 75%, etc.).
   Missing players' volume is redistributed to teammates; a missing starting
   QB hands passing to the backup at reduced efficiency and dents team scoring.
5. Shared "team form" and "game pace" draws make players on the same team
   move together, so same-game parlay probabilities include correlation.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .espn import Game, Injury
from .weather import Weather

S = config.STATS  # ("pass_yds", "rush_yds", "rec_yds")


@dataclass
class PlayerIn:
    pid: str
    name: str
    pos: str
    p_play: float
    status: str
    mean: dict
    cv: dict
    season_avg: dict


@dataclass
class SideIn:
    team_id: str
    abbr: str
    pts_mu: float
    exp_pass: float
    exp_rush: float
    players: list[PlayerIn]
    notes: list[str] = field(default_factory=list)


@dataclass
class Matchup:
    game: Game
    home: SideIn
    away: SideIn
    weather: Weather | None


@dataclass
class SimResult:
    matchup: Matchup
    home_pts: np.ndarray
    away_pts: np.ndarray
    yards: dict          # pid -> {stat: array}
    played: dict         # pid -> bool array
    players: dict        # pid -> (PlayerIn, team abbr)


# --------------------------------------------------------------------------- build
def _shrink(val, gp, lg):
    w = gp / (gp + config.SHRINK_GAMES)
    return w * val + (1 - w) * lg


def play_probability(status: str) -> float:
    s = (status or "").lower()
    if not s or s == "active":
        return 1.0
    for key, p in config.PLAY_PROB:
        if key in s:
            return p
    return 1.0


def _injury_lookup(injuries: list[Injury]) -> tuple[dict, dict]:
    by_id = {i.player_id: i for i in injuries if i.player_id}
    by_name = {i.name.lower(): i for i in injuries}
    return by_id, by_name


def build_side(team_id, opp_id, abbr, teams, players, injuries, roster, opp_injuries,
               wx_mult, hfa, available=None) -> SideIn:
    lg = teams.attrs["league"]
    T, O = teams.loc[team_id], teams.loc[opp_id]
    notes = []

    off_pass = _shrink(T.adj_off_pass, T.gp, lg["pass"])
    def_pass = _shrink(O.adj_def_pass, O.gp, lg["pass"])
    off_rush = _shrink(T.adj_off_rush, T.gp, lg["rush"])
    def_rush = _shrink(O.adj_def_rush, O.gp, lg["rush"])
    off_pts = _shrink(T.adj_off_pts, T.gp, lg["pts"])
    def_pts = _shrink(O.adj_def_pts, O.gp, lg["pts"])

    exp_pass = off_pass * def_pass / lg["pass"] * wx_mult["pass_yds"]
    exp_rush = off_rush * def_rush / lg["rush"] * wx_mult["rush_yds"]
    pts_mu = off_pts * def_pts / lg["pts"] * wx_mult["pts"] + hfa

    # opponent defenders ruled out THIS week (IR is already in their numbers)
    def_out = [i for i in (opp_injuries or [])
               if i.pos.upper() in config.DEF_POSITIONS and any(k in i.status.lower() for k in ("out", "doubtful"))
               and "reserve" not in i.status.lower()]
    if def_out:
        bump = min(config.DEF_OUT_CAP, config.DEF_OUT_PTS * len(def_out))
        pts_mu += bump
        notes.append(f"+{bump:.1f} pts: opp defenders out ({', '.join(i.name for i in def_out[:5])})")

    f_pass = exp_pass / T.pass_ypg if T.pass_ypg > 0 else 1.0
    f_rush = exp_rush / T.rush_ypg if T.rush_ypg > 0 else 1.0
    factor = {"pass_yds": f_pass, "rush_yds": f_rush, "rec_yds": f_pass}

    # candidate players: on the current roster (catches trades) or last played here
    if roster:
        cand = players[players.index.isin(list(roster))]
    else:
        cand = players[players.team_id == team_id]
    if available is None:
        stale = cand[cand.missed_recent]
        if len(stale):
            notes.append("already priced in (missed recent games): " + ", ".join(stale.name.head(6)))
        cand = cand[~cand.missed_recent]
    cand = cand.copy()
    if roster:  # roster positions are more reliable than box-score positions
        cand["pos"] = [roster.get(pid, {}).get("pos") or pos for pid, pos in zip(cand.index, cand.pos)]
    is_qb = (cand.pos == "QB") & (cand.pass_att_pg >= 5)
    skill = (cand.rush_yds_avg + cand.rec_yds_avg) >= 5
    cand = cand[is_qb | skill].copy()
    cand["tot"] = cand.pass_yds_avg + cand.rush_yds_avg + cand.rec_yds_avg
    cand = cand.sort_values("tot", ascending=False).head(16)

    by_id, by_name = _injury_lookup(injuries or [])
    plist = []
    for pid, r in cand.iterrows():
        if available is not None:
            p, status = (1.0 if pid in available else 0.0), ("played" if pid in available else "did not play")
        else:
            inj = by_id.get(pid) or by_name.get(str(r["name"]).lower())
            status = inj.status if inj else ((roster or {}).get(pid, {}).get("injury_status", "") or "")
            p = play_probability(status)
        mean, cv = {}, {}
        for s in S:
            mean[s] = max(0.0, r[f"{s}_wavg"] * factor[s])
            obs = r[f"{s}_cv"]
            d = config.DEFAULT_CV[s]
            cv[s] = float(np.clip(d if np.isnan(obs) else 0.5 * obs + 0.5 * d, *config.CV_BOUNDS))
        plist.append(PlayerIn(pid, r["name"], r.pos, p, status, mean, cv,
                              {s: r[f"{s}_avg"] for s in S}))
    # make sure a backup QB exists so a missing starter hands off realistically
    if available is None and sum(p.pos == "QB" for p in plist) < 2:
        have = {p.pid for p in plist}
        backups = [(pid, r) for pid, r in (roster or {}).items() if r.get("pos") == "QB" and pid not in have]
        pid, r = backups[0] if backups else (f"{team_id}-QB2", {"name": f"{abbr} backup QB", "injury_status": ""})
        zero = {s: 0.0 for s in S}
        plist.append(PlayerIn(pid, r["name"], "QB", play_probability(r.get("injury_status", "")),
                              r.get("injury_status", ""), dict(zero), dict(config.DEFAULT_CV), dict(zero)))
    return SideIn(team_id, abbr, float(pts_mu), float(exp_pass), float(exp_rush), plist, notes)


def build_matchup(game: Game, teams: pd.DataFrame, players: pd.DataFrame,
                  injuries: dict | None = None, rosters: dict | None = None,
                  weather: Weather | None = None, available: dict | None = None) -> Matchup:
    injuries, rosters, available = injuries or {}, rosters or {}, available or {}
    wx = weather.multipliers() if weather else {"pass_yds": 1, "rush_yds": 1, "rec_yds": 1, "pts": 1}
    hfa = 0.0 if game.neutral else config.HOME_FIELD_PTS
    h, a = game.home, game.away
    home = build_side(h.id, a.id, h.abbr, teams, players, injuries.get(h.id), rosters.get(h.id),
                      injuries.get(a.id), wx, +hfa, available.get(h.id))
    away = build_side(a.id, h.id, a.abbr, teams, players, injuries.get(a.id), rosters.get(a.id),
                      injuries.get(h.id), wx, -hfa, available.get(a.id))
    return Matchup(game, home, away, weather)


# --------------------------------------------------------------------------- simulate
def _sim_side(side: SideIn, rng, n, pace):
    P = len(side.players)
    base = np.array([[p.mean[s] for s in S] for p in side.players]).reshape(P, 3)
    cvs = np.array([[p.cv[s] for s in S] for p in side.players]).reshape(P, 3)
    pp = np.array([p.p_play for p in side.players])
    avail = rng.random((n, P)) < pp
    yards = np.zeros((n, P, 3))

    qbs = sorted([i for i, p in enumerate(side.players) if p.pos == "QB"], key=lambda i: -base[i, 0])
    qb_out = np.zeros(n, bool)
    if qbs:
        s = qbs[0]
        qb_out = ~avail[:, s]
        yards[:, s, 0] = base[s, 0]
        if len(qbs) > 1:
            b = qbs[1]
            backup_in = qb_out & avail[:, b]
            yards[:, b, 0] = np.where(backup_in, base[s, 0] * config.BACKUP_QB_EFF, 0.0)
            avail[:, b] = backup_in
            for extra in qbs[2:]:
                avail[:, extra] = False
    elif P:
        side.notes.append("no QB with stats found — passing projections unavailable")

    for c in (1, 2):  # rushing, receiving: redistribute missing volume
        m = base[:, c]
        miss = (m * ~avail).sum(1)
        have = (m * avail).sum(1)
        boost = 1 + config.RETAIN_VOLUME * miss / np.maximum(have, 1.0)
        yards[:, :, c] = m * avail * boost[:, None]
    yards[:, :, 2] *= np.where(qb_out, config.QB_OUT_REC_MULT, 1.0)[:, None]
    if qbs:
        yards[:, qbs[0], :] *= avail[:, qbs[0], None]

    skill = base[:, 1] + base[:, 2]
    lost = (skill * ~avail).sum(1) / max(skill.sum(), 1.0)
    pts_mult = (1 - config.SKILL_LOSS_PTS * lost) * np.where(qb_out, config.QB_OUT_PTS_MULT, 1.0)

    sd = config.TEAM_FORM_SD
    form = rng.lognormal(-sd * sd / 2, sd, n)
    pts = side.pts_mu * pts_mult * form * pace + rng.normal(0, config.SCORE_NOISE_SD, n)
    pts = np.clip(np.round(pts), 0, None)

    k = 1 / cvs ** 2
    noise = rng.gamma(k[None].repeat(n, 0), (cvs ** 2)[None].repeat(n, 0))
    yards *= ((form ** config.PLAYER_FORM_BETA) * pace)[:, None, None] * noise
    return pts, yards, avail


def simulate(m: Matchup, n: int = config.N_SIMS, seed: int | None = None) -> SimResult:
    rng = np.random.default_rng(seed)
    pace = rng.lognormal(-config.PACE_SD ** 2 / 2, config.PACE_SD, n)
    hp, hy, ha = _sim_side(m.home, rng, n, pace)
    ap, ay, aa = _sim_side(m.away, rng, n, pace)
    tie = hp == ap
    home_wins_ot = rng.random(n) < m.home.pts_mu / max(m.home.pts_mu + m.away.pts_mu, 1e-6)
    hp = hp + np.where(tie & home_wins_ot, 3, 0)
    ap = ap + np.where(tie & ~home_wins_ot, 3, 0)

    yards, played, players = {}, {}, {}
    for side, y, av in ((m.home, hy, ha), (m.away, ay, aa)):
        for j, p in enumerate(side.players):
            yards[p.pid] = {s: y[:, j, i] for i, s in enumerate(S)}
            played[p.pid] = av[:, j]
            players[p.pid] = (p, side.abbr)
    return SimResult(m, hp, ap, yards, played, players)


# --------------------------------------------------------------------------- outputs
def american(p: float) -> str:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return f"{-100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def find_player(res: SimResult, name: str) -> str | None:
    names = {p.name.lower(): pid for pid, (p, _) in res.players.items()}
    hit = difflib.get_close_matches(name.lower(), list(names), n=1, cutoff=0.6)
    return names[hit[0]] if hit else None


def prop(res: SimResult, pid: str, stat: str, line: float) -> dict:
    played = res.played[pid]
    y = res.yards[pid][stat][played]
    p_over = float((y > line).mean()) if len(y) else float("nan")
    return {"p_play": float(played.mean()), "p_over": p_over, "p_under": 1 - p_over,
            "fair_over": american(p_over), "fair_under": american(1 - p_over)}


def leg_mask(res: SimResult, leg: str) -> np.ndarray:
    """Leg syntax:
        ML:BUF                       moneyline
        SPREAD:BUF:-3.5              team covers
        TOTAL:over:47.5              game total
        TEAMTOTAL:BUF:over:24.5
        PROP:Josh Allen:pass_yds:over:245.5   (player must play, else leg fails)
    """
    parts = [p.strip() for p in leg.split(":")]
    kind = parts[0].upper()
    g = res.matchup.game
    margin_for = lambda abbr: (res.home_pts - res.away_pts) if abbr.upper() == g.home.abbr else (res.away_pts - res.home_pts)
    pts_for = lambda abbr: res.home_pts if abbr.upper() == g.home.abbr else res.away_pts
    side = lambda s, v, line: v > line if s.lower() == "over" else v < line
    if kind == "ML":
        return margin_for(parts[1]) > 0
    if kind == "SPREAD":
        return margin_for(parts[1]) + float(parts[2]) > 0
    if kind == "TOTAL":
        return side(parts[1], res.home_pts + res.away_pts, float(parts[2]))
    if kind == "TEAMTOTAL":
        return side(parts[2], pts_for(parts[1]), float(parts[3]))
    if kind == "PROP":
        pid = find_player(res, parts[1])
        if pid is None:
            raise ValueError(f"player not found in this game: {parts[1]}")
        return res.played[pid] & side(parts[3], res.yards[pid][parts[2]], float(parts[4]))
    raise ValueError(f"unknown leg type: {leg}")


def parlay(res: SimResult, legs: list[str]) -> dict:
    masks = [leg_mask(res, l) for l in legs]
    singles = [float(m.mean()) for m in masks]
    joint = float(np.logical_and.reduce(masks).mean())
    naive = float(np.prod(singles))
    return {"legs": list(zip(legs, singles)), "joint": joint, "independent": naive,
            "fair_odds": american(joint), "correlation_lift": joint / naive if naive else float("nan")}


def summary(res: SimResult) -> dict:
    g = res.matchup.game
    margin = res.home_pts - res.away_pts
    total = res.home_pts + res.away_pts
    hw = float((margin > 0).mean())
    return {"game": g.label, "home": g.home.abbr, "away": g.away.abbr,
            "home_win": hw, "away_win": 1 - hw,
            "home_ml": american(hw), "away_ml": american(1 - hw),
            "home_pts": float(res.home_pts.mean()), "away_pts": float(res.away_pts.mean()),
            "spread_home": float(-margin.mean()), "total": float(total.mean()),
            "total_mean": float(total.mean())}


def player_lines(res: SimResult, min_yards: float = 15) -> pd.DataFrame:
    rows = []
    for pid, (p, team) in res.players.items():
        played = res.played[pid]
        for s in S:
            y = res.yards[pid][s][played]
            if not len(y) or np.median(y) < min_yards:
                continue
            rows.append({"team": team, "player": p.name, "pos": p.pos, "stat": s,
                         "status": p.status or "active", "p_play": played.mean(),
                         "season_avg": p.season_avg[s], "proj_mean": y.mean(),
                         "median (fair line)": np.median(y), "p10": np.percentile(y, 10),
                         "p90": np.percentile(y, 90)})
    df = pd.DataFrame(rows)
    return df.sort_values(["team", "stat", "proj_mean"], ascending=[True, True, False]) if len(df) else df
