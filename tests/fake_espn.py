"""Offline fake of the ESPN + Open-Meteo JSON endpoints, shaped like the real responses.

Lets the whole pipeline run without network access: `install()` swaps get_json.
"""
import random
from datetime import datetime, timedelta, timezone

from nfl_sim import config, espn, weather

ABBRS = list(config.STADIUMS)
CURRENT_WEEK = 11
SEASON = 2026
rng = random.Random(7)

TEAMS = {str(i + 1): {"id": str(i + 1), "abbreviation": a, "displayName": f"{a} Team"} for i, a in enumerate(ABBRS)}
STRENGTH = {tid: (rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)) for tid in TEAMS}  # (offense, defense-weakness)
PLAYERS = {}
for tid in TEAMS:
    o = STRENGTH[tid][0]
    PLAYERS[tid] = [
        {"id": f"{tid}01", "name": f"{TEAMS[tid]['abbreviation']} QB1", "pos": "QB", "pass": 235 * o, "rush": 15},
        {"id": f"{tid}02", "name": f"{TEAMS[tid]['abbreviation']} QB2", "pos": "QB", "pass": 0, "rush": 0},
        {"id": f"{tid}03", "name": f"{TEAMS[tid]['abbreviation']} RB1", "pos": "RB", "rush": 70 * o, "rec": 20},
        {"id": f"{tid}04", "name": f"{TEAMS[tid]['abbreviation']} RB2", "pos": "RB", "rush": 30, "rec": 10},
        {"id": f"{tid}05", "name": f"{TEAMS[tid]['abbreviation']} WR1", "pos": "WR", "rec": 80 * o},
        {"id": f"{tid}06", "name": f"{TEAMS[tid]['abbreviation']} WR2", "pos": "WR", "rec": 55 * o},
        {"id": f"{tid}07", "name": f"{TEAMS[tid]['abbreviation']} WR3", "pos": "WR", "rec": 30},
        {"id": f"{tid}08", "name": f"{TEAMS[tid]['abbreviation']} TE1", "pos": "TE", "rec": 40},
    ]

SCHEDULE = {}
for wk in range(1, 19):
    ids = list(TEAMS)
    rng.shuffle(ids)
    SCHEDULE[wk] = [(ids[i], ids[i + 1]) for i in range(0, 32, 2)]  # (home, away)

KICK0 = datetime.now(timezone.utc) + timedelta(days=3)
BOX = {}


def _game_id(wk, i):
    return f"4017{wk:02d}{i:02d}"


def _box(wk, i):
    if (wk, i) in BOX:
        return BOX[(wk, i)]
    h, a = SCHEDULE[wk][i]
    blocks, score = [], {}
    for tid, opp in ((h, a), (a, h)):
        d = STRENGTH[opp][1]
        pas, rus, rec = [], [], []
        qb_yds = 0
        for p in PLAYERS[tid]:
            if p.get("pass"):
                qb_yds = max(0, rng.gauss(p["pass"] * d, 55))
                att = int(qb_yds / 7)
                pas.append({"athlete": {"id": p["id"], "displayName": p["name"], "position": {"abbreviation": "QB"}},
                            "stats": [f"{int(att*0.65)}/{att}", f"{qb_yds:.0f}", "7.0", "1", "0", "2-10", "55", "90"]})
            if p.get("rush"):
                y = max(-5, rng.gauss(p["rush"] * d, p["rush"] * 0.5))
                rus.append({"athlete": {"id": p["id"], "displayName": p["name"]},
                            "stats": [str(max(1, int(y / 4.3))), f"{y:.0f}", "4.3", "0", "20"]})
            if p.get("rec"):
                y = max(0, rng.gauss(p["rec"] * d, p["rec"] * 0.6))
                rec.append({"athlete": {"id": p["id"], "displayName": p["name"]},
                            "stats": [str(int(y / 12) + 1), f"{y:.0f}", "12", "0", "30", str(int(y / 8) + 2)]})
        blocks.append({"team": {"id": tid}, "statistics": [
            {"name": "passing", "labels": ["C/ATT", "YDS", "AVG", "TD", "INT", "SACKS", "QBR", "RTG"],
             "keys": ["completions/passingAttempts", "passingYards", "yardsPerPassAttempt", "passingTouchdowns",
                      "interceptions", "sacks-sackYardsLost", "adjQBR", "QBRating"], "athletes": pas},
            {"name": "rushing", "labels": ["CAR", "YDS", "AVG", "TD", "LONG"], "athletes": rus},  # labels only
            {"name": "receiving", "labels": ["REC", "YDS", "AVG", "TD", "LONG", "TGTS"],
             "keys": ["receptions", "receivingYards", "yardsPerReception", "receivingTouchdowns",
                      "longReception", "receivingTargets"], "athletes": rec}]})
        score[tid] = max(0, int(rng.gauss(22 * STRENGTH[tid][0] * d, 9)))
    BOX[(wk, i)] = ({"boxscore": {"players": blocks}}, score)
    return BOX[(wk, i)]


def _scoreboard(wk):
    events = []
    for i, (h, a) in enumerate(SCHEDULE[wk]):
        done = wk < CURRENT_WEEK
        sc = _box(wk, i)[1] if done else {}
        kick = KICK0 + timedelta(days=7 * (wk - CURRENT_WEEK), hours=i % 3)
        events.append({"id": _game_id(wk, i), "date": kick.strftime("%Y-%m-%dT%H:%MZ"),
                       "competitions": [{
                           "neutralSite": i == 15 and wk == CURRENT_WEEK,
                           "venue": {"fullName": f"{TEAMS[h]['abbreviation']} Stadium",
                                     "address": {"city": "London" if (i == 15 and wk == CURRENT_WEEK) else "Somecity",
                                                 "state": ""}},
                           "status": {"type": {"completed": done}},
                           "competitors": [
                               {"homeAway": "home", "team": TEAMS[h], "score": str(sc.get(h, 0))},
                               {"homeAway": "away", "team": TEAMS[a], "score": str(sc.get(a, 0))}]}]})
    return {"season": {"year": SEASON, "type": 2}, "week": {"number": wk}, "events": events}


def fake_get_json(url, params=None, ttl=None):
    params = params or {}
    if url.endswith("/scoreboard"):
        return _scoreboard(int(params.get("week", CURRENT_WEEK)))
    if url.endswith("/summary"):
        gid = params["event"]
        return _box(int(gid[4:6]), int(gid[6:8]))[0]
    if url.endswith("/teams"):
        return {"sports": [{"leagues": [{"teams": [{"team": t} for t in TEAMS.values()]}]}]}
    if "/roster" in url:
        tid = url.split("/teams/")[1].split("/")[0]
        return {"athletes": [{"position": "offense", "items": [
            {"id": p["id"], "displayName": p["name"], "position": {"abbreviation": p["pos"]}, "injuries": []}
            for p in PLAYERS[tid]]}]}
    if url.endswith("/injuries"):
        return {"injuries": [
            {"id": "1", "displayName": "ARI Team", "injuries": [
                {"athlete": {"displayName": "ARI QB1", "position": {"abbreviation": "QB"},
                             "links": [{"href": "https://www.espn.com/nfl/player/_/id/101/x"}]},
                 "status": "Questionable", "shortComment": "ankle"},
                {"athlete": {"id": "105", "displayName": "ARI WR1", "position": {"abbreviation": "WR"}},
                 "status": "Out", "shortComment": "hamstring"}]},
            {"id": "2", "displayName": "ATL Team", "injuries": [
                {"athlete": {"id": "999", "displayName": "ATL CB1", "position": {"abbreviation": "CB"}},
                 "status": "Out", "shortComment": "knee"}]}]}
    if "geocoding" in url:
        return {"results": [{"latitude": 51.5, "longitude": -0.12, "admin1": "England", "country_code": "GB"}]}
    if "open-meteo" in url:
        start = (datetime.now(timezone.utc) - timedelta(days=7)).replace(minute=0, second=0, microsecond=0)
        times = [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(24 * 23)]
        n = len(times)
        return {"hourly": {"time": times, "temperature_2m": [41.0] * n, "precipitation_probability": [20] * n,
                           "precipitation": [0.0] * n, "wind_speed_10m": [9.0] * n, "wind_gusts_10m": [16.0] * n}}
    raise KeyError(url)


def install():
    espn.get_json = fake_get_json
    weather.get_json = fake_get_json
