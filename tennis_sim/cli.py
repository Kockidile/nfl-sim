"""python -m tennis_sim export | backtest | slate"""
from __future__ import annotations

import argparse
import difflib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, model, ratings

SERIES_TOUR = data.MATCH_SERIES
LEVEL_NAME = {"KXATPMATCH": "ATP", "KXWTAMATCH": "WTA", "KXATPCHALLENGERMATCH": "ATP Challenger", "KXCHALLENGERMATCH": "Challenger",
              "KXWTACHALLENGERMATCH": "WTA 125", "KXITFMATCH": "ITF Men", "KXITFWMATCH": "ITF Women", "KXUNITEDCUPMATCH": "United Cup",
              "KXDAVISCUPMATCH": "Davis Cup", "KXLAVERCUPMATCH": "Laver Cup", "KXTENNISEXHIBITION": "Exhibition", "KXEXHIBITIONMEN": "Exhibition"}


def _lookup(R: ratings.Ratings, name: str, tour: str):
    key = data.norm(name)
    p = R.find(key, tour)
    if p:
        return p
    pool = [k for (t, k) in R.p if tour in ("?", t)]
    parts = key.split()
    if parts:  # same surname + first initial
        cands = [k for k in pool if k.split() and k.split()[-1] == parts[-1] and k[0] == key[0]]
        if len(cands) == 1:
            return R.find(cands[0], tour)
    hit = difflib.get_close_matches(key, pool, n=1, cutoff=0.88)
    return R.find(hit[0], tour) if hit else None


def _context(verbose=True):
    matches, smap, end = data.load_all(verbose)
    R = ratings.build(matches)
    return matches, smap, end, R


def analyze(R, smap, ev, series, now):
    p = data.parse_match_event(ev)
    if not p:
        return None
    tour = SERIES_TOUR.get(series, "?")
    a_name, b_name = p["players"]
    A, B = _lookup(R, a_name, tour), _lookup(R, b_name, tour)
    surface = data.surface_from_name(p["tourney"] or p["title"], smap) or "Hard"
    slam = any(s in (p["tourney"] or "").lower() for s in ("australian open", "roland", "french open", "wimbledon", "us open"))
    best_of = 5 if tour == "M" and slam and "qualif" not in (p["tourney"] or "").lower() else 3
    pa_mkt, pb_mkt = data.price(p["markets"][a_name]), data.price(p["markets"][b_name])
    mkt = None
    if pa_mkt["mid"] and pb_mkt["mid"]:
        mkt = pa_mkt["mid"] / (pa_mkt["mid"] + pb_mkt["mid"])
    item = {"event": p["event"], "series": series, "level": LEVEL_NAME.get(series, series), "title": p["title"],
            "tourney": p["tourney"], "surface": surface, "best_of": best_of,
            "start": p["start"].isoformat() + "Z" if p["start"] else None,
            "players": [a_name, b_name], "market": {"p_a": mkt, "a": pa_mkt, "b": pb_mkt}}
    if not A or not B:
        item["missing"] = [n for n, x in ((a_name, A), (b_name, B)) if not x]
        return item
    if tour == "?":
        tour = A.tour
    asof = pd.Timestamp(now.replace(tzinfo=None))
    pa_prof, pb_prof = ratings.profile(A, asof, surface), ratings.profile(B, asof, surface)
    elo_p = ratings.predict(A.rating(surface), B.rating(surface))
    sa, sb = model.base_serve_probs(A.serve, B.serve, tour)
    ca, cb = model.calibrate(sa, sb, elo_p, best_of)
    sim = model.simulate(ca, cb, best_of, n=12000)
    g = sim.pop("games")
    med = sim["games_median"]
    lines = {f"{x + .5:g}": float((g > x + .5).mean()) for x in range(int(med) - 4, int(med) + 4)}
    n_min = min(A.n, B.n)
    stale = max(pa_prof["days_rest"] or 999, pb_prof["days_rest"] or 999)
    conf = "high" if n_min >= 40 and stale <= 45 else "medium" if n_min >= 15 and stale <= 120 else "low"
    notes = []
    for nm, pr in ((a_name, pa_prof), (b_name, pb_prof)):
        if pr["played_7d"] >= 4:
            notes.append(f"{nm} has played {pr['played_7d']} matches in the last 7 days (fatigue risk)")
        if pr["days_rest"] is not None and pr["days_rest"] > 60:
            notes.append(f"{nm}: no result in our data for {pr['days_rest']} days (injury layoff or data gap)")
        if not pr["serve"]:
            notes.append(f"{nm}: no serve stats in the last year; set/score shape uses tour averages")
    h2h = [x for x in A.results if data.norm(x[2]) == B.key]
    item.update({
        "tour": tour, "a": pa_prof, "b": pb_prof, "confidence": conf, "notes": notes,
        "h2h": f"{sum(x[1] for x in h2h)}-{sum(not x[1] for x in h2h)}" if h2h else "0-0",
        "model": {"p_a": sim["a_win"], "elo_p": elo_p, "set1_a": sim["set1_a"],
                  "set_scores": dict(list(sim["set_scores"].items())[:6]), "straight_sets": sim["straight_sets"],
                  "games_median": med, "games_mean": round(sim["games_mean"], 1), "games_over": lines,
                  "hold_a": round(sim["hold_a"], 3), "hold_b": round(sim["hold_b"], 3),
                  "spw_a": round(ca, 3), "spw_b": round(cb, 3)},
    })
    started = p["start"] is not None and p["start"] <= now.replace(tzinfo=None)
    if started:
        item["live"] = True
        notes.insert(0, "Match has started: Kalshi's price reflects play so far, so there's no pre-match edge")
    elif mkt is not None:
        item["edge_a"] = sim["a_win"] - mkt
        vol = pa_mkt["volume"] + pb_mkt["volume"]
        if abs(item["edge_a"]) >= 0.20 and vol >= 10000:
            notes.insert(0, "Big gap vs a heavily traded market: traders may know something the model doesn't (injury, withdrawal, form). Treat with caution.")
    return item


def cmd_export(args):
    now = datetime.now(timezone.utc)
    matches, smap, end, R = _context(not args.quiet)
    out = {"sport": "tennis", "generated": now.isoformat(timespec="seconds"), "archive_end": str(end),
           "n_matches": int(len(matches)), "sources": matches.source.value_counts().to_dict(), "matches": []}
    for series in SERIES_TOUR:
        for ev in data.kalshi_events(series, "open", max_pages=5, ttl=900):
            try:
                it = analyze(R, smap, ev, series, now)
            except Exception as e:  # keep going if one match fails
                it = {"event": ev.get("event_ticker"), "title": ev.get("title"), "error": str(e)}
            if it:
                out["matches"].append(it)
    out["matches"].sort(key=lambda m: m.get("start") or "9999")
    out["backtest"] = backtest_summary(R)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, default=lambda o: None if o is None else float(o) if isinstance(o, (np.floating, np.integer)) else str(o), separators=(",", ":")))
    ok = sum(1 for m in out["matches"] if "model" in m)
    print(f"Wrote {args.out}: {len(out['matches'])} Kalshi matches ({ok} modelled)")


def backtest_summary(R, since="2025-01-01", min_n=10) -> dict:
    log = pd.DataFrame(R.log, columns=["date", "tour", "source", "level", "w", "l", "p", "wn", "ln"])
    log = log[(log.date >= since) & (log.wn >= min_n) & (log.ln >= min_n)]
    if log.empty:
        return {}
    lo = np.log(log.p / (1 - log.p)) * ratings.ELO_SCALE
    p = (1 / (1 + np.exp(-lo))).clip(.01, .99)
    res = {"matches": int(len(log)), "accuracy": float((p > .5).mean()), "brier": float(((1 - p) ** 2).mean()),
           "log_loss": float(-np.log(p).mean())}
    # calibration: favourite's predicted vs actual win rate
    fav = np.maximum(p, 1 - p); fav_won = p > .5
    bins = pd.cut(fav, [.5, .6, .7, .8, .9, 1.0], include_lowest=True)
    cal = pd.DataFrame({"bin": bins, "pred": fav, "won": fav_won}).groupby("bin", observed=True).agg(n=("won", "size"), pred=("pred", "mean"), actual=("won", "mean"))
    res["calibration"] = [{"bin": str(i), "n": int(r.n), "pred": round(r.pred, 3), "actual": round(r.actual, 3)} for i, r in cal.iterrows()]
    res["by_level"] = {}
    lv = log.level.astype(str).str[:1].map({"G": "Grand Slam", "M": "Masters", "A": "Tour", "F": "Tour Finals", "C": "Challenger", "S": "ITF", "1": "ITF", "2": "ITF", "D": "Davis Cup"}).fillna("Other")
    for name, g in pd.DataFrame({"lv": lv, "p": p}).groupby("lv"):
        res["by_level"][name] = {"n": int(len(g)), "acc": round(float((g.p > .5).mean()), 3)}
    return res


def cmd_backtest(args):
    matches, smap, end, R = _context(not args.quiet)
    print(json.dumps(backtest_summary(R, args.since), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tennis_sim")
    ap.add_argument("-q", "--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--out", default="docs/tennis.json"); e.set_defaults(fn=cmd_export)
    b = sub.add_parser("backtest"); b.add_argument("--since", default="2025-01-01"); b.set_defaults(fn=cmd_backtest)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
