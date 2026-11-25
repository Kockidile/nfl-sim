"""Time-weighted Poisson team ratings with league-level shrinkage and a Dixon-Coles low-score correction.

Expected goals: home = BASE * HOME * att[home] * dfn[away],  away = BASE * att[away] * dfn[home]
(att > 1 scores more than average, dfn > 1 concedes more than average).
Team ratings are pulled toward their league's rating with PRIOR_MATCHES pseudo-matches, so a
promoted side or a team with few games starts near its league average, and European cup games
tie the leagues together.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .data import CUPS

HALF_LIFE_DAYS = 150
# League strength in log-goals relative to the Premier League (roughly UEFA-coefficient order). Only matters when
# teams from different leagues meet (European cups); "other" = clubs from leagues we don't load (cup games only).
LEAGUE_OFFSET = {"eng.1": 0.0, "esp.1": -0.12, "ita.1": -0.2, "ger.1": -0.2, "fra.1": -0.35, "por.1": -0.5,
                 "ned.1": -0.55, "bel.1": -0.7, "tur.1": -0.7, "eng.2": -0.7, "sco.1": -0.9, "bra.1": -0.55,
                 "usa.1": -0.85, "mex.1": -0.85, "other": -0.95}
PRIOR_MATCHES = 6.0
RHO = -0.07          # Dixon-Coles: slightly more 0-0 / 1-1 than independent Poisson
MAX_GOALS = 10


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


@dataclass
class Ratings:
    base: float
    home: float
    att: dict
    dfn: dict
    league_of: dict
    lg_att: dict
    lg_dfn: dict
    n_eff: dict
    games: dict = field(default_factory=dict)
    sharp: float = 1.0

    def off(self, t, league=None) -> float:
        return LEAGUE_OFFSET.get(self.league_of.get(t, league if league not in CUPS else "other"), LEAGUE_OFFSET["other"])

    def lam(self, h: str, a: str, neutral=False, league=None) -> tuple[float, float]:
        def get(d, lg, t):
            return d.get(t, lg.get(self.league_of.get(t, league), 1.0))
        H = 1.0 if neutral else self.home
        g = math.exp((self.off(h, league) - self.off(a, league)) / 2)
        lh = self.base * H * get(self.att, self.lg_att, h) * get(self.dfn, self.lg_dfn, a) * g
        la = self.base * get(self.att, self.lg_att, a) * get(self.dfn, self.lg_dfn, h) / g
        return lh, la


def fit(matches: list[dict], asof: datetime | None = None, iters=60) -> Ratings:
    asof = asof or datetime.now(timezone.utc)
    done = [m for m in matches if m["hg"] is not None and _ts(m["date"]) < asof]
    if not done:
        raise ValueError("no finished matches")
    # each team's league = the domestic league it plays most in
    cnt = defaultdict(lambda: defaultdict(int))
    for m in done:
        if m["league"] not in CUPS:
            cnt[m["home_id"]][m["league"]] += 1; cnt[m["away_id"]][m["league"]] += 1
    league_of = {t: max(c, key=c.get) for t, c in cnt.items()}
    for m in done:   # cup-only teams (e.g. a club from a league we don't load) get their own pool
        for t in (m["home_id"], m["away_id"]):
            league_of.setdefault(t, "other")
    H = np.array([m["home_id"] for m in done]); A = np.array([m["away_id"] for m in done])
    hg = np.array([m["hg"] for m in done], float); ag = np.array([m["ag"] for m in done], float)
    age = np.array([(asof - _ts(m["date"])).total_seconds() / 86400 for m in done])
    w = 0.5 ** (age / HALF_LIFE_DAYS)
    neu = np.array([m["neutral"] for m in done])
    gap = np.exp(np.array([(LEAGUE_OFFSET.get(league_of[h], -0.95) - LEAGUE_OFFSET.get(league_of[a], -0.95)) / 2 for h, a in zip(H, A)]))
    teams = sorted(set(H) | set(A))
    att = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}
    lgs = sorted(set(league_of.values()))
    lg_att = {l: 1.0 for l in lgs}; lg_dfn = {l: 1.0 for l in lgs}
    base = float((w * (hg + ag)).sum() / (2 * w.sum())); home = 1.25
    idx = {t: i for i, t in enumerate(teams)}
    hi = np.array([idx[t] for t in H]); ai = np.array([idx[t] for t in A])
    tl = np.array([league_of[t] for t in teams])
    n_eff = np.bincount(hi, w, len(teams)) + np.bincount(ai, w, len(teams))
    for _ in range(iters):
        at = np.array([att[t] for t in teams]); df = np.array([dfn[t] for t in teams])
        hf = np.where(neu, 1.0, home)
        eh = base * hf * at[hi] * df[ai] * gap; ea = base * at[ai] * df[hi] / gap
        # home advantage and base rate
        home = float((w * hg)[~neu].sum() / max((w * eh / hf)[~neu].sum(), 1e-9))
        base *= float((w * (hg + ag)).sum() / max((w * (eh + ea)).sum(), 1e-9))
        hf = np.where(neu, 1.0, home)
        # attack: goals scored vs expected at att = 1
        gf = np.bincount(hi, w * hg, len(teams)) + np.bincount(ai, w * ag, len(teams))
        ef = np.bincount(hi, w * base * hf * df[ai] * gap, len(teams)) + np.bincount(ai, w * base * df[hi] / gap, len(teams))
        for l in lgs:   # league level, unshrunk
            mk = tl == l
            lg_att[l] = float(gf[mk].sum() / max(ef[mk].sum(), 1e-9))
        k = PRIOR_MATCHES * base
        la_vec = np.array([lg_att[l] for l in tl])
        at = (gf + k * la_vec) / (ef + k)
        # defense: goals conceded vs expected at dfn = 1
        ga = np.bincount(hi, w * ag, len(teams)) + np.bincount(ai, w * hg, len(teams))
        eg = np.bincount(hi, w * base * at[ai] / gap, len(teams)) + np.bincount(ai, w * base * hf * at[hi] * gap, len(teams))
        for l in lgs:
            mk = tl == l
            lg_dfn[l] = float(ga[mk].sum() / max(eg[mk].sum(), 1e-9))
        ld_vec = np.array([lg_dfn[l] for l in tl])
        df = (ga + k * ld_vec) / (eg + k)
        # Domestic results only fix att*dfn within a league, not how strong the league is; pin each league's
        # average attack = average defense so LEAGUE_OFFSET alone sets league strength (cup games then refine teams)
        for l in lgs:
            mk = tl == l
            wl = n_eff[mk] + 1e-9
            c = math.sqrt(math.exp((wl * np.log(df[mk])).sum() / wl.sum()) / math.exp((wl * np.log(at[mk])).sum() / wl.sum()))
            at[mk] *= c; df[mk] /= c
            lg_att[l] *= c; lg_dfn[l] /= c
        # keep the scale pinned: mean attack = 1 (weighted by games)
        s = float((at * n_eff).sum() / n_eff.sum())
        at /= s; base *= s
        att = dict(zip(teams, at)); dfn = dict(zip(teams, df))
        lg_att = {l: v / s for l, v in lg_att.items()}
    games = defaultdict(int)
    for m in done:
        games[m["home_id"]] += 1; games[m["away_id"]] += 1
    return Ratings(base, home, att, dfn, league_of, lg_att, lg_dfn, dict(zip(teams, n_eff)), dict(games))


def score_matrix(lh: float, la: float) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    ph = np.exp(-lh) * lh ** g / np.array([math.factorial(int(i)) for i in g])
    pa = np.exp(-la) * la ** g / np.array([math.factorial(int(i)) for i in g])
    M = np.outer(ph, pa)
    M[0, 0] *= 1 - lh * la * RHO; M[0, 1] *= 1 + lh * RHO
    M[1, 0] *= 1 + la * RHO;      M[1, 1] *= 1 - RHO
    return M / M.sum()


def outcomes(lh: float, la: float, sharp: float = 1.0) -> dict:
    if abs(sharp - 1) > 1e-9:   # stretch the xG ratio around the geometric mean, keep the total
        tot, r = lh + la, (lh / la) ** sharp
        la = tot / (1 + r); lh = tot - la
    M = score_matrix(lh, la)
    n = MAX_GOALS + 1
    i, j = np.indices((n, n))
    tot = i + j
    return {"home": float(M[i > j].sum()), "draw": float(np.trace(M)), "away": float(M[i < j].sum()),
            "over": {x + 0.5: float(M[tot > x].sum()) for x in range(0, 6)},
            "btts": float(M[(i > 0) & (j > 0)].sum()),
            "xg_home": lh, "xg_away": la,
            "top_scores": sorted(((float(M[a, b]), f"{a}-{b}") for a in range(5) for b in range(5)), reverse=True)[:3]}


def backtest(matches: list[dict], days=28) -> dict:
    """Refit as of `days` ago and score every finished match since then (3-way result).
    Also picks the sharpness that minimises log loss (stored as 'sharp')."""
    now = datetime.now(timezone.utc)
    cut = now.timestamp() - days * 86400
    test = [m for m in matches if m["hg"] is not None and cut <= _ts(m["date"]).timestamp() < now.timestamp()]
    if len(test) < 30:
        return {}
    R = fit(matches, asof=datetime.fromtimestamp(cut, timezone.utc))
    lams = [R.lam(m["home_id"], m["away_id"], m["neutral"], m["league"]) for m in test]
    ys = [0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2 for m in test]

    def score(s):
        ll = brier = hit = 0.0
        for (lh, la), y in zip(lams, ys):
            o = outcomes(lh, la, s)
            p = np.array([o["home"], o["draw"], o["away"]])
            ll -= math.log(max(p[y], 1e-9)); brier += float(((p - np.eye(3)[y]) ** 2).sum()); hit += float(p.argmax() == y)
        n = len(ys)
        return ll / n, brier / n, hit / n
    base_ll, base_b, base_h = score(1.0)
    grid = [round(0.8 + 0.05 * i, 2) for i in range(13)]   # 0.80 .. 1.40
    best = min(grid, key=lambda s: score(s)[0])
    ll, b, h = score(best)
    return {"matches": len(test), "days": days, "accuracy": round(h, 3), "log_loss": round(ll, 3), "brier": round(b, 3),
            "sharp": best, "log_loss_uncalibrated": round(base_ll, 3)}
