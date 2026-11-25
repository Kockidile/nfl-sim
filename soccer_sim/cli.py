"""python -m soccer_sim export --out docs/soccer.json"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from . import data, model

MARKET_WEIGHT = 0.5       # final chance = 50% model + 50% Kalshi's own price (where the market is two-sided)
MARKET_WEIGHT_CUP = 0.7   # European cups: teams from different leagues, so the model is less sure; lean on the market
SEASON_START = {"usa.1": (1, 1), "bra.1": (1, 1), "mex.1": (7, 1)}   # everyone else: Jul 15


def _ts(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def team_context(matches, now):
    """Per team: last-5 form, recent goals for/against, rest days, league table position."""
    done = sorted((m for m in matches if m["hg"] is not None and _ts(m["date"]) < now), key=lambda m: m["date"])
    hist = defaultdict(list)
    for m in done:
        hist[m["home_id"]].append((m, m["hg"], m["ag"], True))
        hist[m["away_id"]].append((m, m["ag"], m["hg"], False))
    ctx = {}
    for t, h in hist.items():
        last = h[-10:]
        form = "".join("W" if f > a else "D" if f == a else "L" for _, f, a, _ in h[-5:])
        ctx[t] = {"form": form, "gf10": round(sum(f for _, f, _, _ in last) / len(last), 2),
                  "ga10": round(sum(a for _, _, a, _ in last) / len(last), 2),
                  "last": h[-1][0]["date"], "games": len(h)}
    # league tables this season
    for slug, _ in data.LEAGUES.values():
        if slug in data.CUPS:
            continue
        mth, day = SEASON_START.get(slug, (7, 15))
        start = datetime(now.year if (now.month, now.day) >= (mth, day) else now.year - 1, mth, day, tzinfo=timezone.utc)
        tab = defaultdict(lambda: [0, 0, 0])   # pts, gd, gp
        for m in done:
            if m["league"] != slug or _ts(m["date"]) < start:
                continue
            for t, f, a in ((m["home_id"], m["hg"], m["ag"]), (m["away_id"], m["ag"], m["hg"])):
                tab[t][0] += 3 if f > a else 1 if f == a else 0; tab[t][1] += f - a; tab[t][2] += 1
        order = sorted(tab, key=lambda t: (-tab[t][0], -tab[t][1]))
        for i, t in enumerate(order):
            if t in ctx:
                ctx[t].update({"pos": i + 1, "of": len(order), "pts": tab[t][0], "gp": tab[t][2]})
    return ctx


def _mid(pr):
    if not pr or pr.get("ask") is None:
        return None
    b, a = pr["bid"] or 0.0, pr["ask"]
    return (a + b) / 2 if b and a > b else None


def _leg(kind, label, p, yes_pr, side="yes", line=None, mw=MARKET_WEIGHT):
    """A buyable Kalshi position with the model's chance. side='no' buys the NO contract."""
    if not yes_pr or yes_pr.get("ask") is None:
        return None
    bid, ask = yes_pr["bid"] or 0.0, yes_pr["ask"]
    if side == "no":
        if not bid:
            return None
        ask, bid = 1 - bid, 1 - ask
    spread = ask - bid
    q = (ask + bid) / 2 if bid > 0 else None
    if ask <= 0 or ask >= 0.99 or spread > 0.12 or q is None:
        return None
    pm = p
    p = (1 - mw) * pm + mw * q
    return {"kind": kind, "label": label, "p": round(p, 4), "p_model": round(pm, 4), "ask": round(ask, 3), "q": round(q, 4),
            "spread": round(spread, 3), "volume": yes_pr.get("volume", 0), "line": line, "side": side}


def cmd_export(args):
    now = datetime.now(timezone.utc)
    matches = data.load_matches(days_back=args.days, verbose=not args.quiet)
    R = model.fit(matches, asof=now)
    try:
        bt = model.backtest(matches)
    except Exception as e:  # noqa: BLE001
        bt = {"error": str(e)}
    sharp = min(max(bt.get("sharp", 1.0), 0.9), 1.3) if isinstance(bt, dict) else 1.0
    ctx = team_context(matches, now)
    fixtures = [m for m in matches if m["state"] == "pre" and now - timedelta(minutes=5) < _ts(m["date"]) < now + timedelta(hours=args.hours)]
    kal = data.load_kalshi(verbose=not args.quiet)
    link = data.match_kalshi(fixtures, kal)
    names = {v[0]: v[1] for v in data.LEAGUES.values()}
    out = {"sport": "soccer", "generated": now.isoformat(timespec="seconds"), "backtest": bt,
           "model": {"base": round(R.base, 3), "home": round(R.home, 3), "half_life_days": model.HALF_LIFE_DAYS,
                     "sharp": sharp, "market_weight": MARKET_WEIGHT, "market_weight_cup": MARKET_WEIGHT_CUP},
           "matches": []}
    for f in sorted(fixtures, key=lambda m: m["date"]):
        h, a = f["home_id"], f["away_id"]
        lh, la = R.lam(h, a, f["neutral"], f["league"])
        o = model.outcomes(lh, la, sharp)
        ne = min(R.n_eff.get(h, 0), R.n_eff.get(a, 0))
        conf = "high" if ne >= 12 else "medium" if ne >= 6 else "low"
        item = {"id": f["id"], "league": names.get(f["league"], f["league"]), "slug": f["league"], "start": f["date"],
                "home": f["home"], "away": f["away"], "home_abbr": f["home_abbr"], "away_abbr": f["away_abbr"],
                "neutral": f["neutral"], "confidence": conf,
                "model": {"home": round(o["home"], 4), "draw": round(o["draw"], 4), "away": round(o["away"], 4),
                          "xg_home": round(lh, 2), "xg_away": round(la, 2), "btts": round(o["btts"], 4),
                          "over": {str(k): round(v, 4) for k, v in o["over"].items()},
                          "top_scores": [s for _, s in o["top_scores"]]},
                "ctx": {}, "legs": [], "kalshi": None}
        for side, t in (("home", h), ("away", a)):
            c = dict(ctx.get(t, {}))
            if c.get("last"):
                c["rest_days"] = round((_ts(f["date"]) - _ts(c["last"])).total_seconds() / 86400, 1)
            c["att"] = round(R.att.get(t, 1.0), 3); c["dfn"] = round(R.dfn.get(t, 1.0), 3)
            item["ctx"][side] = c
        if f["id"] in link:
            e, kh, ka, sc = link[f["id"]]
            item["kalshi"] = {"event": e["suffix"], "match_score": round(sc, 2)}
            mw = MARKET_WEIGHT_CUP if f["league"] in data.CUPS else MARKET_WEIGHT
            legs = [
                _leg("win", f"{f['home']} to win", o["home"], e["win"].get(kh), mw=mw),
                _leg("win", f"{f['away']} to win", o["away"], e["win"].get(ka), mw=mw),
                _leg("dc", f"{f['home']} win or draw", o["home"] + o["draw"], e["win"].get(ka), "no", mw=mw),
                _leg("dc", f"{f['away']} win or draw", o["away"] + o["draw"], e["win"].get(kh), "no", mw=mw),
            ]
            for x, pr in sorted(e["totals"].items()):
                po = o["over"].get(x)
                if po is None:
                    continue
                legs.append(_leg("over", f"Over {x} goals", po, pr, line=x, mw=mw))
                legs.append(_leg("under", f"Under {x} goals", 1 - po, pr, "no", line=x, mw=mw))
            if e.get("btts"):
                legs.append(_leg("btts", "Both teams score", o["btts"], e["btts"], mw=mw))
                legs.append(_leg("btts_no", "Not both teams score", 1 - o["btts"], e["btts"], "no", mw=mw))
            item["legs"] = [L for L in legs if L]
            mk = {k: _mid(e["win"].get(n)) for k, n in (("home", kh), ("away", ka))}
            mk["draw"] = _mid(e.get("tie"))
            item["market"] = mk
        out["matches"].append(item)
    if not args.quiet:
        n = sum(1 for m in out["matches"] if m["kalshi"])
        print(f"soccer: {len(out['matches'])} fixtures in the next {args.hours}h, {n} linked to Kalshi, "
              f"{sum(len(m['legs']) for m in out['matches'])} legs; backtest {bt}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="soccer_sim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", default="docs/soccer.json")
    e.add_argument("--hours", type=float, default=36)
    e.add_argument("--days", type=int, default=270)
    e.add_argument("-q", "--quiet", action="store_true")
    e.set_defaults(fn=cmd_export)
    a = ap.parse_args(argv)
    a.fn(a)
