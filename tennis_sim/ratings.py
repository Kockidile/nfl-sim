"""Surface-aware Elo ratings and player profiles (form, serve/return, fatigue)."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SURFACES = ("Hard", "Clay", "Grass")
START = 1500.0
SURF_WEIGHT = 0.5          # blend of surface Elo vs overall Elo when the surface is known
LEVEL_K = {"G": 1.1, "M": 1.05, "F": 1.05, "A": 1.0, "T": 1.0, "D": 0.9, "C": 0.9, "Q": 0.85, "S": 0.8, "15": 0.8, "25": 0.8}


TOUR_LEVELS = {"G", "M", "A", "F", "D", "P", "PM", "I", "O", "T", "T1", "T2", "T3", "T4", "T5", "KXATPMATCH", "KXWTAMATCH",
               "KXUNITEDCUPMATCH", "KXDAVISCUPMATCH", "KXLAVERCUPMATCH"}


def start_rating(level) -> float:
    """New players start near the level they first appear at, so ITF win streaks don't inflate them."""
    lv = str(level)
    if lv in TOUR_LEVELS:
        return 1550.0
    if lv == "C" or "CHALLENGER" in lv:
        return 1400.0
    return 1250.0   # futures / ITF / unknown low level


def k_factor(n: int) -> float:
    return 250.0 / (n + 5) ** 0.4          # FiveThirtyEight-style: newcomers move faster


def expect(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


ELO_SCALE = 1.15   # backtest (2025-26, 70k matches): raw Elo is slightly under-confident


def predict(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** (ELO_SCALE * (rb - ra) / 400))


@dataclass
class Player:
    key: str
    name: str
    tour: str
    elo: float = START
    n: int = 0
    s_elo: dict = field(default_factory=lambda: {s: START for s in SURFACES})
    s_n: dict = field(default_factory=lambda: {s: 0 for s in SURFACES})
    results: deque = field(default_factory=lambda: deque(maxlen=60))   # (date, won, opp, surface, tourney, score, source)
    rank: float | None = None
    serve: dict = field(default_factory=dict)

    def rating(self, surface: str | None) -> float:
        if surface in self.s_elo and self.s_n[surface] >= 3:
            return (1 - SURF_WEIGHT) * self.elo + SURF_WEIGHT * self.s_elo[surface]
        return self.elo


class Ratings:
    def __init__(self):
        self.p: dict[tuple[str, str], Player] = {}
        self.log = []   # pre-match probabilities for backtests

    def get(self, key, name, tour, level="") -> Player:
        k = (tour, key)
        if k not in self.p:
            r0 = start_rating(level)
            self.p[k] = Player(key, name, tour, elo=r0, s_elo={s: r0 for s in SURFACES})
        return self.p[k]

    def find(self, key, tour=None) -> Player | None:
        if tour in ("M", "W") and (tour, key) in self.p:
            return self.p[(tour, key)]
        for t in ("M", "W"):
            if (t, key) in self.p:
                return self.p[(t, key)]
        return None

    def update(self, m) -> None:
        tour = m.tour if m.tour in ("M", "W") else ("W" if self.find(m.w_key, "W") and not self.find(m.w_key, "M") else "M")
        w, l = self.get(m.w_key, m.w_name, tour, m.level), self.get(m.l_key, m.l_name, tour, m.level)
        surf = m.surface if m.surface in SURFACES else None
        p = expect(w.rating(surf), l.rating(surf))
        self.log.append((m.date, tour, m.source, m.level, w.key, l.key, p, w.n, l.n))
        if m.retired:   # still record the result for form/fatigue, but don't move ratings much
            mult = 0.25
        else:
            mult = LEVEL_K.get(str(m.level)[:2], LEVEL_K.get(str(m.level)[:1], 1.0))
        e = expect(w.elo, l.elo)
        w.elo += k_factor(w.n) * mult * (1 - e)
        l.elo -= k_factor(l.n) * mult * (1 - e)
        w.n += 1; l.n += 1
        if surf:
            es = expect(w.s_elo[surf], l.s_elo[surf])
            w.s_elo[surf] += k_factor(w.s_n[surf]) * mult * (1 - es)
            l.s_elo[surf] -= k_factor(l.s_n[surf]) * mult * (1 - es)
            w.s_n[surf] += 1; l.s_n[surf] += 1
        for pl, won, opp, rank in ((w, True, l, m.w_rank), (l, False, w, m.l_rank)):
            pl.results.append((m.date, won, opp.name, m.surface, m.tourney, m.score, m.source))
            if pd.notna(rank):
                pl.rank = float(rank)


def build(matches: pd.DataFrame) -> Ratings:
    r = Ratings()
    for m in matches.itertuples(index=False):
        r.update(m)
    serve_profiles(r, matches)
    return r


def serve_profiles(r: Ratings, matches: pd.DataFrame, weeks: int = 52) -> None:
    """Serve & return point-win %, hold % and break % over the last year of archive stats."""
    a = matches[(matches.source == "archive") & matches.w_svpt.notna() & (matches.w_svpt > 0)]
    if a.empty:
        return
    a = a[a.date >= a.date.max() - pd.Timedelta(weeks=weeks)]
    agg = defaultdict(lambda: np.zeros(8))
    for m in a.itertuples(index=False):
        for me, op in (("w", "l"), ("l", "w")):
            key = (m.tour, getattr(m, f"{me}_key"))
            sv_won = getattr(m, f"{me}_1stWon") + getattr(m, f"{me}_2ndWon")
            ret_won = getattr(m, f"{op}_svpt") - getattr(m, f"{op}_1stWon") - getattr(m, f"{op}_2ndWon")
            agg[key] += np.nan_to_num(np.array([
                getattr(m, f"{me}_svpt"), sv_won, getattr(m, f"{op}_svpt"), ret_won,
                getattr(m, f"{me}_SvGms"), getattr(m, f"{me}_bpFaced") - getattr(m, f"{me}_bpSaved"),
                getattr(m, f"{op}_SvGms"), getattr(m, f"{op}_bpFaced") - getattr(m, f"{op}_bpSaved")], dtype=float))
    tour_avg = {}
    for tour in ("M", "W"):
        tot = sum((v for k, v in agg.items() if k[0] == tour), np.zeros(8))
        tour_avg[tour] = tot[1] / max(tot[0], 1)
    r.tour_spw = tour_avg
    for (tour, key), v in agg.items():
        pl = r.p.get((tour, key))
        if not pl or v[0] < 150:
            continue
        pl.serve = {"spw": v[1] / v[0], "rpw": v[3] / max(v[2], 1), "points": int(v[0]),
                    "hold": 1 - v[5] / max(v[4], 1), "brk": v[7] / max(v[6], 1)}


def profile(pl: Player, asof, surface: str | None) -> dict:
    res = list(pl.results)
    last10 = res[-10:]
    yr = [x for x in res if (asof - x[0]).days <= 365]
    on_surf = [x for x in yr if x[3] == surface] if surface else []
    last = res[-1][0] if res else None
    return {
        "name": pl.name, "elo": round(pl.elo), "surf_elo": round(pl.s_elo.get(surface, pl.elo)) if surface else None,
        "rating": round(pl.rating(surface)), "matches": pl.n, "rank": pl.rank,
        "form": "".join("W" if x[1] else "L" for x in last10),
        "last10": f"{sum(x[1] for x in last10)}-{sum(not x[1] for x in last10)}",
        "surface_record": f"{sum(x[1] for x in on_surf)}-{sum(not x[1] for x in on_surf)}" if surface else None,
        "days_rest": (asof - last).days if last is not None else None,
        "played_7d": sum((asof - x[0]).days <= 7 for x in res),
        "last_match": last.strftime("%Y-%m-%d") if last is not None else None,
        "recent": [{"date": x[0].strftime("%b %d"), "won": bool(x[1]), "opp": x[2], "score": x[5], "event": x[4]} for x in res[-5:][::-1]],
        "serve": {k: round(v, 3) if isinstance(v, float) else v for k, v in pl.serve.items()},
    }
