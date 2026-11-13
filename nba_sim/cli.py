"""NBA command line.  python -m nba_sim export | slate"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config, espn, model, stats


def _context(verbose):
    today = espn.today_et()
    if verbose:
        print(f"Loading NBA box scores through {today - timedelta(days=1)}...")
    data = stats.load_season(today - timedelta(days=1), verbose)
    return today, data, stats.team_table(data), stats.player_table(data)


def _slate(today, days=2):
    games = []
    for d in range(days):
        games += espn.games_on(today + timedelta(days=d))
    seen, out = set(), []
    for g in games:
        if g.id not in seen and g.season_type in (2, 3, 5):
            seen.add(g.id)
            out.append(g)
    return out


def _side_json(side, injuries):
    return {"abbr": side.abbr, "pts_mu": round(side.pts_mu, 2), "pace": round(side.pace, 1), "b2b": side.b2b,
            "notes": side.notes,
            "players": [{"pid": p.pid, "name": p.name, "pos": p.pos, "p_play": p.p_play, "status": p.status or "",
                         "minutes": round(p.minutes, 1), "note": p.note,
                         "mean": {k: round(float(v), 2) for k, v in p.mean.items()},
                         "cv": {k: round(float(v), 3) for k, v in p.cv.items()},
                         "season_avg": p.season_avg} for p in side.players],
            "injuries": [{"name": i.name, "pos": i.pos, "status": i.status, "detail": i.detail} for i in injuries]}


def cmd_export(args):
    today, data, tt, pt = _context(not args.quiet)
    teams = espn.get_teams()
    injuries = espn.get_injuries(teams)
    out = {"sport": "nba", "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "today": today.isoformat(),
           "config": {k: getattr(config, k) for k in (
               "RETAIN", "STAR_LOSS_PTS", "TEAM_FORM_SD", "PACE_SD", "SCORE_NOISE_SD", "PLAYER_FORM_BETA",
               "BLOWOUT_START", "BLOWOUT_PER_PT", "BLOWOUT_FLOOR", "STARTER_MIN")},
           "games": [], "rankings": []}
    for g in _slate(today):
        item = {"id": g.id, "label": g.label, "date": g.date, "day": g.day.isoformat(), "venue": g.venue,
                "city": g.city, "neutral": g.neutral, "completed": g.completed,
                "home": {"id": g.home.id, "abbr": g.home.abbr, "name": g.home.name, "score": g.home.score},
                "away": {"id": g.away.id, "abbr": g.away.abbr, "name": g.away.name, "score": g.away.score}}
        if not g.completed and not tt.empty and g.home.id in tt.index and g.away.id in tt.index:
            try:
                rosters = {g.home.id: espn.get_roster(g.home.id), g.away.id: espn.get_roster(g.away.id)}
                m = model.build_matchup(g, tt, pt, injuries, rosters)
                res = model.simulate(m, n=10000, seed=1)
                mg = res["hp"] - res["ap"]
                item["summary"] = {"home_win": float((mg > 0).mean()), "away_win": float((mg < 0).mean()),
                                   "spread_home": float(-mg.mean()), "total": float((res["hp"] + res["ap"]).mean()),
                                   "home_pts": float(res["hp"].mean()), "away_pts": float(res["ap"].mean())}
                item["pace"] = round(m.pace, 1)
                item["sides"] = {"home": _side_json(m.home, injuries.get(g.home.id, [])),
                                 "away": _side_json(m.away, injuries.get(g.away.id, []))}
            except Exception as e:  # keep the slate if one game fails
                item["error"] = str(e)
        out["games"].append(item)
    if not tt.empty:
        cols = ["abbr", "gp", "ppg", "pa_pg", "pace", "adj_ortg", "adj_drtg", "net", "l10_net",
                "rk_off", "rk_def", "rk_net", "rk_pace"]
        out["rankings"] = json.loads(tt[cols].round(1).to_json(orient="records"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, default=float, separators=(",", ":")))
    print(f"Wrote {args.out}: {len(out['games'])} games ({today} and next day)")


def cmd_slate(args):
    today, data, tt, pt = _context(not args.quiet)
    injuries = espn.get_injuries(espn.get_teams())
    for g in _slate(today):
        if g.completed or g.home.id not in tt.index or g.away.id not in tt.index:
            continue
        m = model.build_matchup(g, tt, pt, injuries,
                                {g.home.id: espn.get_roster(g.home.id), g.away.id: espn.get_roster(g.away.id)})
        res = model.simulate(m, n=10000, seed=1)
        mg = res["hp"] - res["ap"]
        print(f"\n{g.label}  {g.home.abbr} win {(mg > 0).mean():.0%}  spread {-mg.mean():+.1f}  total {(res['hp'] + res['ap']).mean():.1f}")
        for pid, r in res["players"].items():
            pl = r["played"]
            if pl.mean() == 0:
                continue
            med = {s: np.median(model.stat_array(r, s)[pl]) for s in ("pts", "reb", "ast", "fg3", "pra")}
            print(f"  {r['team']} {r['p'].name:24s} min {r['p'].minutes:4.1f} play {pl.mean():.2f} "
                  + " ".join(f"{k} {v:5.1f}" for k, v in med.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nba_sim")
    ap.add_argument("-q", "--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--out", default="docs/nba.json"); e.set_defaults(fn=cmd_export)
    sub.add_parser("slate").set_defaults(fn=cmd_slate)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
