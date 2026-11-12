"""Command line interface.  Run `python -m nfl_sim --help`."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, espn, model, stats
from .weather import game_weather

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


# --------------------------------------------------------------------------- helpers
def _week(args):
    season, week, stype = espn.current_week()
    season = args.season or season
    week = args.week or week
    return season, week, stype


def _load(season, week, stype, verbose=True) -> stats.SeasonData:
    through = week - 1 if stype == 2 else 18
    if verbose:
        print(f"Loading {season} box scores through week {through} (cached after first run)...")
    return stats.load_season(season, max(through, 0), 2, verbose)


def _team_id(abbr: str, teams: dict) -> str:
    for tid, t in teams.items():
        if t.abbr.upper() == abbr.upper():
            return tid
    sys.exit(f"Unknown team abbreviation: {abbr}")


def _kick(g):
    try:
        return datetime.fromisoformat(g.date.replace("Z", "+00:00")).strftime("%a %b %d %H:%M UTC")
    except ValueError:
        return g.date


# --------------------------------------------------------------------------- commands
def cmd_games(args):
    season, week, stype = _week(args)
    print(f"\n{season} week {week}\n")
    for g in espn.get_week_games(season, week, stype):
        wx = game_weather(g)
        score = f"  FINAL {g.away.score}-{g.home.score}" if g.completed else ""
        print(f"{g.label:<12} {_kick(g):<22} {g.venue} ({g.city}, {g.state}){' [neutral]' if g.neutral else ''}{score}")
        print(f"{'':<12} {wx.describe()}")


def cmd_rankings(args):
    season, week, stype = _week(args)
    t = stats.team_table(_load(season, week, stype, not args.quiet))
    cols = ["abbr", "gp", "ppg", "pa_pg", "l4_ppg", "pass_ypg", "rk_off_pass", "rush_ypg", "rk_off_rush",
            "pass_allowed_pg", "rk_def_pass", "rush_allowed_pg", "rk_def_rush", "rk_off_pts", "rk_def_pts"]
    sort = {"off": "rk_off_pts", "def": "rk_def_pts", "pass": "rk_off_pass", "rush": "rk_off_rush",
            "passdef": "rk_def_pass", "rushdef": "rk_def_rush"}[args.sort]
    print("\nRanks are opponent-adjusted (1 = best). Raw per-game averages shown alongside.\n")
    print(t[cols].sort_values(sort).round(1).to_string(index=False))
    if args.csv:
        t.to_csv(args.csv)
        print(f"\nSaved {args.csv}")


def cmd_injuries(args):
    teams = espn.get_teams()
    tid = _team_id(args.team, teams)
    inj = espn.get_injuries(teams).get(tid, [])
    roster = espn.get_roster(tid)
    print(f"\n{teams[tid].name} injury report ({len(inj)} listed)\n")
    for i in sorted(inj, key=lambda i: model.play_probability(i.status)):
        print(f"  {i.pos:<4} {i.name:<26} {i.status:<18} play≈{model.play_probability(i.status):.0%}  {i.detail}")
    print(f"\nActive roster size: {len(roster)}")


def cmd_players(args):
    season, week, stype = _week(args)
    teams = espn.get_teams()
    tid = _team_id(args.team, teams)
    p = stats.player_table(_load(season, week, stype, not args.quiet))
    roster = espn.get_roster(tid)
    p = p[p.index.isin(list(roster))].copy() if roster else p[p.team_id == tid].copy()
    if roster:
        p["pos"] = [roster.get(pid, {}).get("pos") or pos for pid, pos in zip(p.index, p.pos)]
    p = p[(p.pass_yds_avg + p.rush_yds_avg + p.rec_yds_avg) >= 3]
    cols = ["name", "pos", "gp", "pass_yds_avg", "pass_yds_l3", "rush_yds_avg", "rush_yds_l3",
            "rec_yds_avg", "rec_yds_l3", "tgt_pg", "rush_att_pg"]
    print(f"\n{teams[tid].name} — weekly averages (avg = season, l3 = last 3 games played)\n")
    print(p[cols].sort_values("pass_yds_avg", ascending=False).round(1).to_string(index=False))


def print_report(res: model.SimResult, show_players=True):
    m, s = res.matchup, model.summary(res)
    g = m.game
    print("\n" + "=" * 92)
    print(f"{g.label}   {_kick(g)}   {g.venue} ({g.city}, {g.state}){'  [neutral site]' if g.neutral else ''}")
    print(f"Weather: {m.weather.describe() if m.weather else 'n/a'}")
    print("-" * 92)
    print(f"Projected score: {s['home']} {s['home_pts']:.1f} – {s['away']} {s['away_pts']:.1f}")
    print(f"Win prob: {s['home']} {s['home_win']:.1%} (fair {s['home_ml']})  |  "
          f"{s['away']} {s['away_win']:.1%} (fair {s['away_ml']})")
    print(f"Fair spread: {s['home']} {s['spread_home']:+.1f}   Fair total: {s['total']:.1f}")
    for side in (m.home, m.away):
        print(f"  {side.abbr}: expected pass {side.exp_pass:.0f} / rush {side.exp_rush:.0f} yds, {side.pts_mu:.1f} pts baseline")
        hurt = [p for p in side.players if p.p_play < 1]
        if hurt:
            print("    injuries: " + "; ".join(f"{p.name} ({p.status}, plays {p.p_play:.0%})" for p in hurt))
        for n in side.notes:
            print(f"    note: {n}")
    if show_players:
        df = model.player_lines(res)
        if len(df):
            print("\nPlayer yardage (median = fair prop line; p10/p90 = range)")
            print(df.round({"p_play": 2, "season_avg": 1, "proj_mean": 1, "median (fair line)": 1,
                            "p10": 0, "p90": 0}).to_string(index=False))


def cmd_sim(args):
    season, week, stype = _week(args)
    data = _load(season, week, stype, not args.quiet)
    tt, pt = stats.team_table(data), stats.player_table(data)
    if tt.empty:
        sys.exit("No completed games yet this season — nothing to build ratings from.")
    teams = espn.get_teams()
    injuries = espn.get_injuries(teams)
    games = espn.get_week_games(season, week, stype)
    if args.game:
        want = {a.upper() for a in args.game}
        games = [g for g in games if g.home.abbr in want or g.away.abbr in want]
    if not games:
        sys.exit("No matching games.")

    outdir = Path(args.csv) if args.csv else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for g in games:
        if g.home.id not in tt.index or g.away.id not in tt.index:
            print(f"Skipping {g.label}: no data for one of the teams yet")
            continue
        rosters = {g.home.id: espn.get_roster(g.home.id), g.away.id: espn.get_roster(g.away.id)}
        wx = game_weather(g)
        mu = model.build_matchup(g, tt, pt, injuries, rosters, wx)
        res = model.simulate(mu, n=args.n, seed=args.seed)
        print_report(res, show_players=not args.no_players)
        summaries.append(model.summary(res))

        for spec in args.line or []:
            name, stat, line = spec.rsplit(":", 2)
            pid = model.find_player(res, name)
            if pid:
                r = model.prop(res, pid, stat, float(line))
                print(f"  PROP {res.players[pid][0].name} {stat} {line}: over {r['p_over']:.1%} ({r['fair_over']}) "
                      f"/ under {r['p_under']:.1%} ({r['fair_under']}), plays {r['p_play']:.0%}")
        if args.parlay:
            try:
                pr = model.parlay(res, args.parlay)
                print("\n  SAME-GAME PARLAY")
                for leg, p in pr["legs"]:
                    print(f"    {leg:<45} {p:.1%}")
                print(f"    joint {pr['joint']:.1%} (fair {pr['fair_odds']}) vs {pr['independent']:.1%} if independent "
                      f"→ correlation lift ×{pr['correlation_lift']:.2f}")
            except ValueError as e:
                print(f"  parlay skipped for {g.label}: {e}")
        if outdir:
            model.player_lines(res).to_csv(outdir / f"wk{week}_{g.away.abbr}_at_{g.home.abbr}_players.csv", index=False)

    if len(summaries) > 1:
        print("\n" + "=" * 92 + "\nSLATE SUMMARY")
        df = pd.DataFrame(summaries)[["game", "home_win", "home_ml", "away_ml", "spread_home", "total"]]
        print(df.round(3).to_string(index=False))
    if outdir and summaries:
        pd.DataFrame(summaries).to_csv(outdir / f"wk{week}_games.csv", index=False)
        print(f"\nCSV files written to {outdir}/")


def cmd_backtest(args):
    """Re-simulate past weeks using only data available before each week."""
    season, week, stype = _week(args)
    end = (args.end or week - 1)
    data = _load(season, end + 1, 2, not args.quiet)
    rows, prow = [], []
    for w in range(args.start, end + 1):
        prior = data.before(w)
        tt, pt = stats.team_table(prior), stats.player_table(prior)
        for g in [g for g in espn.get_week_games(season, w, 2) if g.completed]:
            if g.home.id not in tt.index or g.away.id not in tt.index:
                continue
            actual = data.players[data.players.game_id == g.id]
            avail = {tid: set(actual[actual.team_id == tid].player_id) for tid in (g.home.id, g.away.id)}
            res = model.simulate(model.build_matchup(g, tt, pt, available=avail), n=4000, seed=1)
            s = model.summary(res)
            hw = 1.0 if g.home.score > g.away.score else 0.0 if g.home.score < g.away.score else 0.5
            rows.append({"week": w, "game": g.label, "p_home": s["home_win"], "home_won": hw,
                         "proj_margin": -s["spread_home"], "margin": g.home.score - g.away.score,
                         "proj_total": s["total"], "total": g.home.score + g.away.score})
            for pid, (p, _) in res.players.items():
                act = actual[actual.player_id == pid]
                if act.empty:
                    continue
                for st in config.STATS:
                    if p.season_avg[st] >= 15:
                        prow.append({"proj": float(np.median(res.yards[pid][st][res.played[pid]])),
                                     "naive": p.season_avg[st], "actual": float(act[st].iloc[0])})
        print(f"  week {w}: {sum(r['week'] == w for r in rows)} games", flush=True)
    if not rows:
        sys.exit("Nothing to backtest.")
    df = pd.DataFrame(rows)
    p = df.p_home.clip(0.01, 0.99)
    brier = ((p - df.home_won) ** 2).mean()
    acc = ((p > 0.5) == (df.home_won > 0.5)).mean()
    print(f"\nBACKTEST weeks {args.start}-{end}: {len(df)} games")
    print(f"  Winner accuracy: {acc:.1%}    Brier: {brier:.3f}  (0.25 = coin flip; lower is better)")
    print(f"  Spread MAE: {(df.proj_margin - df.margin).abs().mean():.1f} pts   "
          f"Total MAE: {(df.proj_total - df.total).abs().mean():.1f} pts")
    if prow:
        pr = pd.DataFrame(prow)
        print(f"  Player yards MAE: model {(pr.proj - pr.actual).abs().mean():.1f}  vs  "
              f"season-average baseline {(pr.naive - pr.actual).abs().mean():.1f}  ({len(pr)} props)")
    df["bucket"] = pd.cut(df.p_home.where(df.p_home >= .5, 1 - df.p_home), [.5, .6, .7, .8, 1.0], include_lowest=True)
    df["fav_won"] = np.where(df.p_home >= .5, df.home_won, 1 - df.home_won)
    print("\n  Calibration (favorite's predicted win% vs actual):")
    print(df.groupby("bucket", observed=True).agg(games=("game", "count"), predicted=("p_home", lambda s: s.where(s >= .5, 1 - s).mean()),
                                                  actual=("fav_won", "mean")).round(3).to_string())
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved {args.csv}")


def _side_json(side, injuries):
    return {
        "abbr": side.abbr, "pts_mu": side.pts_mu, "exp_pass": side.exp_pass, "exp_rush": side.exp_rush,
        "notes": side.notes,
        "players": [{"pid": p.pid, "name": p.name, "pos": p.pos, "p_play": p.p_play, "status": p.status or "",
                     "mean": {k: round(float(v), 2) for k, v in p.mean.items()},
                     "cv": {k: round(float(v), 3) for k, v in p.cv.items()},
                     "season_avg": {k: round(float(v), 1) for k, v in p.season_avg.items()}} for p in side.players],
        "injuries": [{"name": i.name, "pos": i.pos, "status": i.status, "detail": i.detail} for i in injuries],
    }


def cmd_export(args):
    """Write everything the phone web app needs to one JSON file."""
    import json
    from datetime import timezone
    season, week, stype = _week(args)
    data = _load(season, week, stype, not args.quiet)
    tt, pt = stats.team_table(data), stats.player_table(data)
    teams = espn.get_teams()
    injuries = espn.get_injuries(teams)
    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "season": season, "week": week,
           "config": {k: getattr(config, k) for k in (
               "RETAIN_VOLUME", "BACKUP_QB_EFF", "QB_OUT_REC_MULT", "QB_OUT_PTS_MULT", "SKILL_LOSS_PTS",
               "TEAM_FORM_SD", "PACE_SD", "SCORE_NOISE_SD", "PLAYER_FORM_BETA")},
           "games": [], "rankings": []}
    for g in espn.get_week_games(season, week, stype):
        item = {"id": g.id, "label": g.label, "date": g.date, "venue": g.venue, "city": g.city, "state": g.state,
                "neutral": g.neutral, "completed": g.completed,
                "home": {"id": g.home.id, "abbr": g.home.abbr, "name": g.home.name, "score": g.home.score},
                "away": {"id": g.away.id, "abbr": g.away.abbr, "name": g.away.name, "score": g.away.score}}
        if not g.completed and g.home.id in tt.index and g.away.id in tt.index:
            try:
                rosters = {g.home.id: espn.get_roster(g.home.id), g.away.id: espn.get_roster(g.away.id)}
                wx = game_weather(g)
                mu = model.build_matchup(g, tt, pt, injuries, rosters, wx)
                item["weather"] = {"indoor": wx.indoor, "text": wx.describe(), "temp_f": wx.temp_f,
                                   "wind_mph": wx.wind_mph, "precip_prob": wx.precip_prob}
                item["summary"] = model.summary(model.simulate(mu, n=20000, seed=1))
                item["sides"] = {"home": _side_json(mu.home, injuries.get(g.home.id, [])),
                                 "away": _side_json(mu.away, injuries.get(g.away.id, []))}
            except Exception as e:  # keep the rest of the slate if one game fails
                item["error"] = str(e)
        out["games"].append(item)
    if not tt.empty:
        cols = ["abbr", "gp", "ppg", "pa_pg", "l4_ppg", "pass_ypg", "rush_ypg", "pass_allowed_pg", "rush_allowed_pg",
                "rk_off_pts", "rk_def_pts", "rk_off_pass", "rk_off_rush", "rk_def_pass", "rk_def_rush"]
        out["rankings"] = json.loads(tt[cols].round(1).to_json(orient="records"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, default=float, separators=(",", ":")))
    print(f"Wrote {args.out}: {len(out['games'])} games, week {week}")


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="nfl_sim", description="NFL Monte Carlo game & player-prop simulator")
    ap.add_argument("--season", type=int, help="season year (default: current)")
    ap.add_argument("--week", type=int, help="week to simulate (default: current)")
    ap.add_argument("-q", "--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("games", help="this week's matchups, venues, kickoff weather").set_defaults(fn=cmd_games)

    r = sub.add_parser("rankings", help="team offense/defense rankings (pass, rush, points)")
    r.add_argument("--sort", default="off", choices=["off", "def", "pass", "rush", "passdef", "rushdef"])
    r.add_argument("--csv")
    r.set_defaults(fn=cmd_rankings)

    i = sub.add_parser("injuries", help="injury report for a team")
    i.add_argument("team")
    i.set_defaults(fn=cmd_injuries)

    p = sub.add_parser("players", help="weekly passing/rushing/receiving averages for a team")
    p.add_argument("team")
    p.set_defaults(fn=cmd_players)

    s = sub.add_parser("sim", help="simulate games")
    s.add_argument("--game", nargs="*", help="team abbreviation(s) to filter, e.g. BUF KC")
    s.add_argument("--n", type=int, default=config.N_SIMS)
    s.add_argument("--seed", type=int)
    s.add_argument("--line", action="append", help='prop line, e.g. "Josh Allen:pass_yds:245.5"')
    s.add_argument("--parlay", action="append",
                   help='parlay leg (repeat): ML:BUF | SPREAD:BUF:-3.5 | TOTAL:over:47.5 | '
                        'TEAMTOTAL:BUF:over:24.5 | "PROP:Josh Allen:pass_yds:over:245.5"')
    s.add_argument("--no-players", action="store_true")
    s.add_argument("--csv", help="folder to write game + player CSVs")
    s.set_defaults(fn=cmd_sim)

    b = sub.add_parser("backtest", help="score the model on past weeks")
    b.add_argument("--start", type=int, default=4)
    b.add_argument("--end", type=int)
    b.add_argument("--csv")
    b.set_defaults(fn=cmd_backtest)

    e = sub.add_parser("export", help="write data.json for the phone web app")
    e.add_argument("--out", default="docs/data.json")
    e.set_defaults(fn=cmd_export)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
