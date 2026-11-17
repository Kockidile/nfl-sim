"""Point-based tennis match model.

Each player's chance of winning a point on serve comes from their serve/return
stats against the opponent. Those two numbers are then shifted together until the
model's match-win probability equals the Elo prediction (Elo is the better
predictor of who wins; the serve numbers shape *how*: sets, scores, total games).
A game-by-game Monte Carlo then gives set scores and total-games lines.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_SPW = {"M": 0.635, "W": 0.565}


def p_game(p: float) -> float:
    q = 1 - p
    return p**4 * (1 + 4*q + 10*q*q) + 20 * p**3 * q**3 * p*p / (1 - 2*p*q)


def p_tiebreak(pa: float, pb: float) -> float:
    """A serves first point; serve alternates every two points. pa/pb = each player's serve point-win prob."""
    @lru_cache(None)
    def f(a, b):
        if a >= 7 and a - b >= 2: return 1.0
        if b >= 7 and b - a >= 2: return 0.0
        if a == b and a >= 6:
            w = pa * (1 - pb); l = (1 - pa) * pb
            return w / (w + l)
        n = a + b
        a_serving = (n % 4) in (0, 3)
        p = pa if a_serving else 1 - pb
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)
    return f(0, 0)


def p_set(pa: float, pb: float, a_first: bool) -> float:
    ha, hb, tb = p_game(pa), p_game(pb), p_tiebreak(pa, pb)
    @lru_cache(None)
    def f(i, j):
        if i >= 6 and i - j >= 2: return 1.0
        if j >= 6 and j - i >= 2: return 0.0
        if i == 6 and j == 6: return tb if a_first else 1 - p_tiebreak(pb, pa)
        a_srv = ((i + j) % 2 == 0) == a_first
        p = ha if a_srv else 1 - hb
        return p * f(i + 1, j) + (1 - p) * f(i, j + 1)
    return f(0, 0)


def p_match(pa: float, pb: float, best_of: int = 3) -> float:
    s = 0.5 * (p_set(pa, pb, True) + p_set(pa, pb, False))
    need = best_of // 2 + 1
    @lru_cache(None)
    def f(a, b):
        if a == need: return 1.0
        if b == need: return 0.0
        return s * f(a + 1, b) + (1 - s) * f(a, b + 1)
    return f(0, 0)


def base_serve_probs(sa: dict, sb: dict, tour: str) -> tuple[float, float]:
    avg = DEFAULT_SPW.get(tour, 0.6)
    spw_a, rpw_a = sa.get("spw", avg), sa.get("rpw", 1 - avg)
    spw_b, rpw_b = sb.get("spw", avg), sb.get("rpw", 1 - avg)
    # opponent-adjusted (Barnett–Clarke): my serve vs your return, relative to tour average
    pa = avg + (spw_a - avg) - (rpw_b - (1 - avg))
    pb = avg + (spw_b - avg) - (rpw_a - (1 - avg))
    return float(np.clip(pa, .35, .9)), float(np.clip(pb, .35, .9))


def calibrate(pa: float, pb: float, target: float, best_of: int) -> tuple[float, float]:
    lo, hi = -0.25, 0.25
    for _ in range(40):
        d = (lo + hi) / 2
        if p_match(np.clip(pa + d, .3, .95), np.clip(pb - d, .3, .95), best_of) < target:
            lo = d
        else:
            hi = d
    d = (lo + hi) / 2
    return float(np.clip(pa + d, .3, .95)), float(np.clip(pb - d, .3, .95))


def simulate(pa: float, pb: float, best_of=3, n=20000, seed=1) -> dict:
    """Game-level Monte Carlo (tiebreaks via the exact tiebreak probability)."""
    rng = np.random.default_rng(seed)
    ha, hb = p_game(pa), p_game(pb)
    tb_a_first, tb_b_first = p_tiebreak(pa, pb), 1 - p_tiebreak(pb, pa)
    need = best_of // 2 + 1
    sets_a = np.zeros(n, int); sets_b = np.zeros(n, int); games = np.zeros(n, int)
    a_serves_first = rng.random(n) < 0.5
    set1_a = np.zeros(n, bool)
    scores = []
    for s in range(best_of):
        live = (sets_a < need) & (sets_b < need)
        if not live.any():
            break
        ga = np.zeros(n, int); gb = np.zeros(n, int)
        server_a = a_serves_first.copy()
        done = ~live
        for _ in range(13):
            act = ~done
            tie = act & (ga == 6) & (gb == 6)
            if tie.any():
                ptb = np.where(server_a, tb_a_first, tb_b_first)
                wa = rng.random(n) < ptb
                ga = np.where(tie & wa, 7, ga); gb = np.where(tie & ~wa, 7, gb)
                done |= tie
                act &= ~tie
            if not act.any():
                break
            win_a = rng.random(n) < np.where(server_a, ha, 1 - hb)
            ga = np.where(act & win_a, ga + 1, ga); gb = np.where(act & ~win_a, gb + 1, gb)
            server_a = np.where(act, ~server_a, server_a)
            fin = act & (((ga >= 6) & (ga - gb >= 2)) | ((gb >= 6) & (gb - ga >= 2)))
            done |= fin
        a_set = live & (ga > gb)
        sets_a += a_set; sets_b += live & (gb > ga)
        games += np.where(live, ga + gb, 0)
        if s == 0:
            set1_a = a_set
        a_serves_first = np.where(live, server_a, a_serves_first)  # next set: whoever is due to serve
        scores.append(None)
    a_win = sets_a > sets_b
    score_dist = {}
    for x, y in zip(sets_a, sets_b):
        score_dist[(int(x), int(y))] = score_dist.get((int(x), int(y)), 0) + 1
    return {"a_win": float(a_win.mean()), "set1_a": float(set1_a.mean()),
            "set_scores": {f"{a}-{b}": c / n for (a, b), c in sorted(score_dist.items(), key=lambda kv: -kv[1])},
            "games": games, "games_median": float(np.median(games)), "games_mean": float(games.mean()),
            "straight_sets": float(((sets_a == need) & (sets_b == 0) | (sets_b == need) & (sets_a == 0)).mean()),
            "hold_a": p_game(pa), "hold_b": p_game(pb)}


def american(p: float) -> str:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return f"{-100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"
