"""NBA matchup construction + Monte Carlo (the phone page runs a JavaScript port of simulate())."""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config
from .espn import Game
from nfl_sim.espn import Injury

S = config.STATS
COMBOS = {"pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast"), "ra": ("reb", "ast")}


@dataclass
class PlayerIn:
    pid: str
    name: str
    pos: str
    p_play: float
    status: str
    minutes: float
    mean: dict
    cv: dict
    season_avg: dict
    note: str = ""


@dataclass
class SideIn:
    team_id: str
    abbr: str
    pts_mu: float
    pace: float
    b2b: bool
    players: list[PlayerIn]
    notes: list[str] = field(default_factory=list)


@dataclass
class Matchup:
    game: Game
    home: SideIn
    away: SideIn
    pace: float


def play_probability(status: str) -> float:
    s = (status or "").lower()
    if not s or s == "active":
        return 1.0
    for key, p in config.PLAY_PROB:
        if key in s:
            return p
    return 1.0


def _shrink(v, gp, lg):
    w = gp / (gp + config.SHRINK_GAMES)
    return w * v + (1 - w) * lg


SCHEDULED: dict[str, set] = {}   # team_id -> ET days with a scheduled (not yet final) game


def _b2b(team_id, teams, day: date) -> bool:
    prev = (day - timedelta(days=1)).isoformat()
    return prev in teams.attrs.get("days", {}).get(team_id, []) or prev in SCHEDULED.get(team_id, set())


def build_side(team_id, opp_id, abbr, teams, players, injuries, roster, game_pace, hfa, day,
               available=None) -> SideIn:
    lg = teams.attrs["league"]
    T, O = teams.loc[team_id], teams.loc[opp_id]
    notes = []
    ortg = _shrink(T.adj_ortg, T.gp, lg["ortg"])
    drtg = _shrink(O.adj_drtg, O.gp, lg["ortg"])
    b2b = _b2b(team_id, teams, day)
    pts_mu = game_pace / 100 * ortg * drtg / lg["ortg"] + hfa - (config.B2B_PTS if b2b else 0)
    if b2b:
        notes.append(f"second night of a back-to-back (−{config.B2B_PTS} pts, watch for rest)")

    team_pace = _shrink(T.pace, T.gp, lg["pace"])
    w = config.MATCHUP_WEIGHT
    factor = {
        "pts": (pts_mu / max(T.ppg, 1)) ** 1.0,
        "reb": 1 + w * (_shrink(O.reb_allowed, O.gp, lg["reb"]) / lg["reb"] - 1),
        "ast": 1 + w * (_shrink(O.ast_allowed, O.gp, lg["ast"]) / lg["ast"] - 1),
        "fg3": 1 + w * (_shrink(O.fg3_allowed, O.gp, lg["fg3"]) / lg["fg3"] - 1),
    }
    pace_f = game_pace / team_pace
    for k in ("reb", "ast", "fg3"):
        factor[k] *= pace_f

    cand = players[players.index.isin(list(roster))] if roster else players[players.team_id == team_id]
    cand = cand.copy()
    by_id = {i.player_id: i for i in (injuries or []) if i.player_id}
    by_name = {i.name.lower(): i for i in (injuries or [])}
    plist = []
    for pid, r in cand.sort_values("min_w", ascending=False).iterrows():
        if available is not None:
            if pid not in available:
                continue
            p, status = 1.0, "played"
        else:
            inj = by_id.get(pid) or by_name.get(str(r["name"]).lower())
            status = inj.status if inj else ((roster or {}).get(pid, {}).get("injury_status", "") or "")
            p = play_probability(status)
            if r.missed_recent and p == 0:
                # team ratings still include the games he played, so take off that share of his scoring
                share = r.pts_avg / max(T.ppg, 1) * min(r.gp / max(T.gp, 1), 1)
                pts_mu *= 1 - config.STAR_LOSS_PTS * share
                notes.append(f"{r['name']} out (−{pts_mu * config.STAR_LOSS_PTS * share / (1 - config.STAR_LOSS_PTS * share):.1f} pts for games the ratings still count him in)")
                continue
        minutes = r.min_w if not r.missed_recent else r.min_avg
        if minutes < config.MIN_MINUTES and r.min_avg < config.MIN_MINUTES + 4:
            continue
        mean, cv = {}, {}
        for s in S:
            mean[s] = max(0.0, r[f"{s}_rate"] * minutes * factor[s])
            obs, d = r[f"{s}_cv"], config.DEFAULT_CV[s]
            cv[s] = float(np.clip(d if np.isnan(obs) else 0.5 * obs + 0.5 * d, *config.CV_BOUNDS))
        note = "returning from absence" if (r.missed_recent and available is None) else ""
        if (roster or {}).get(pid, {}).get("pos"):
            pos = roster[pid]["pos"]
        else:
            pos = r.pos
        plist.append(PlayerIn(pid, r["name"], pos, p, status, float(minutes), mean, cv,
                              {s: round(float(r[f"{s}_avg"]), 1) for s in S}, note))
        if len(plist) >= config.MAX_PLAYERS:
            break
    back = [p.name for p in plist if p.note]
    if back:
        notes.append("returning from absence: " + ", ".join(back))
    return SideIn(team_id, abbr, float(pts_mu), float(game_pace), b2b, plist, notes)


def build_matchup(game: Game, teams, players, injuries=None, rosters=None, available=None) -> Matchup:
    injuries, rosters, available = injuries or {}, rosters or {}, available or {}
    lg = teams.attrs["league"]
    H, A = teams.loc[game.home.id], teams.loc[game.away.id]
    ph, pa = _shrink(H.pace, H.gp, lg["pace"]), _shrink(A.pace, A.gp, lg["pace"])
    game_pace = ph * pa / lg["pace"]
    hfa = 0.0 if game.neutral else config.HOME_COURT_PTS
    home = build_side(game.home.id, game.away.id, game.home.abbr, teams, players, injuries.get(game.home.id),
                      rosters.get(game.home.id), game_pace, +hfa, game.day, available.get(game.home.id))
    away = build_side(game.away.id, game.home.id, game.away.abbr, teams, players, injuries.get(game.away.id),
                      rosters.get(game.away.id), game_pace, -hfa, game.day, available.get(game.away.id))
    return Matchup(game, home, away, float(game_pace))


def quick_summary(m: Matchup) -> dict:
    """Closed-form summary for game cards (the phone re-simulates on demand)."""
    margin = m.home.pts_mu - m.away.pts_mu
    hw = 0.5 * (1 + math.erf(margin / (config.MARGIN_SD * math.sqrt(2))))
    return {"home_win": hw, "away_win": 1 - hw, "spread_home": -margin,
            "total": m.home.pts_mu + m.away.pts_mu, "home_pts": m.home.pts_mu, "away_pts": m.away.pts_mu}


# --------------------------------------------------------------------------- simulation
def _side(side: SideIn, rng, n, pace):
    P = len(side.players)
    base = np.array([[p.mean[s] for s in S] for p in side.players]).reshape(P, 4)
    cvs = np.array([[p.cv[s] for s in S] for p in side.players]).reshape(P, 4)
    pp = np.array([p.p_play for p in side.players])
    starters = np.array([p.minutes >= config.STARTER_MIN for p in side.players])
    avail = rng.random((n, P)) < pp
    y = np.zeros((n, P, 4))
    for c in range(4):
        m = base[:, c]
        miss = (m * ~avail).sum(1)
        have = (m * avail).sum(1)
        y[:, :, c] = m * avail * (1 + config.RETAIN * miss / np.maximum(have, 1e-6))[:, None]
    lost = (base[:, 0] * ~avail).sum(1) / max(base[:, 0].sum(), 1)
    sd = config.TEAM_FORM_SD
    form = rng.lognormal(-sd * sd / 2, sd, n)
    pts = side.pts_mu * (1 - config.STAR_LOSS_PTS * lost) * form * pace + rng.normal(0, config.SCORE_NOISE_SD, n)
    k = 1 / cvs ** 2
    noise = rng.gamma(k[None].repeat(n, 0), (cvs ** 2)[None].repeat(n, 0))
    beta = np.array([config.PLAYER_FORM_BETA, 0.3, 0.6, config.PLAYER_FORM_BETA])
    y *= noise * (form[:, None, None] ** beta[None, None, :]) * pace[:, None, None]
    return np.round(pts), y, avail, starters


def simulate(m: Matchup, n=config.N_SIMS, seed=None) -> dict:
    rng = np.random.default_rng(seed)
    pace = rng.lognormal(-config.PACE_SD ** 2 / 2, config.PACE_SD, n)
    hp, hy, ha, hs = _side(m.home, rng, n, pace)
    ap, ay, aa, as_ = _side(m.away, rng, n, pace)
    for _ in range(4):  # overtime(s)
        tie = hp == ap
        if not tie.any():
            break
        hp = hp + np.where(tie, np.round(rng.normal(11, 3, n)), 0)
        ap = ap + np.where(tie, np.round(rng.normal(11, 3, n)), 0)
    hp = hp + ((hp == ap) & (rng.random(n) < .5))
    margin = np.abs(hp - ap)
    blow = np.clip(1 - config.BLOWOUT_PER_PT * np.maximum(margin - config.BLOWOUT_START, 0), config.BLOWOUT_FLOOR, 1)
    out = {"hp": hp, "ap": ap, "players": {}}
    for side, y, av, st in ((m.home, hy, ha, hs), (m.away, ay, aa, as_)):
        y[:, st, :] *= blow[:, None, None]
        for j, p in enumerate(side.players):
            out["players"][p.pid] = {"p": p, "team": side.abbr, "played": av[:, j],
                                     **{s: y[:, j, i] for i, s in enumerate(S)}}
    return out


def stat_array(rec: dict, stat: str) -> np.ndarray:
    return sum(rec[s] for s in COMBOS[stat]) if stat in COMBOS else rec[stat]


def find_player(res, name):
    names = {r["p"].name.lower(): pid for pid, r in res["players"].items()}
    hit = difflib.get_close_matches(name.lower(), list(names), n=1, cutoff=0.6)
    return names[hit[0]] if hit else None
