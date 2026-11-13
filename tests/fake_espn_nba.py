"""Offline fake of ESPN's NBA endpoints, shaped like the real JSON."""
import random
from datetime import date, datetime, timedelta, timezone

from nba_sim import espn as nba_espn
from nfl_sim import espn as nfl_espn

rng = random.Random(11)
ABBRS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND", "LAC", "LAL", "MEM",
         "MIA", "MIL", "MIN", "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SA", "TOR", "UTAH", "WSH"]
TEAMS = {str(i + 1): {"id": str(i + 1), "abbreviation": a, "displayName": f"{a} Team"} for i, a in enumerate(ABBRS)}
STRENGTH = {t: (rng.uniform(.93, 1.07), rng.uniform(.93, 1.07), rng.uniform(96, 104)) for t in TEAMS}
ROLES = [("G", 34, 26, 4, 7, 3), ("G", 32, 18, 3, 5, 2.5), ("F", 33, 20, 7, 3, 1.5), ("F", 30, 14, 8, 2, 1),
         ("C", 29, 15, 11, 2, .3), ("G", 24, 10, 3, 3, 1.5), ("F", 20, 8, 4, 1, 1), ("C", 16, 6, 5, 1, 0),
         ("G", 13, 5, 2, 2, .8), ("F", 9, 3, 2, 1, .4)]
PLAYERS = {t: [{"id": f"{t}{j:02d}", "name": f"{TEAMS[t]['abbreviation']} P{j}", "pos": r[0], "min": r[1],
                "pts": r[2], "reb": r[3], "ast": r[4], "fg3": r[5]} for j, r in enumerate(ROLES)] for t in TEAMS}
START = date(2026, 10, 20)
TODAY = nba_espn.today_et()
SCHED = {}
for i in range((TODAY - START).days + 2):
    d = START + timedelta(days=i)
    ids = list(TEAMS); rng.shuffle(ids)
    SCHED[d] = [(ids[k], ids[k + 1]) for k in range(0, 14, 2)]
BOX = {}
KEYS = ["minutes", "points", "fieldGoalsMade-fieldGoalsAttempted", "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
        "freeThrowsMade-freeThrowsAttempted", "rebounds", "assists", "turnovers", "steals", "blocks",
        "offensiveRebounds", "defensiveRebounds", "fouls", "plusMinus"]
LABELS = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]


def _gid(d, i):
    return f"40188{d.strftime('%m%d')}{i}"


def _box(d, i):
    if (d, i) in BOX:
        return BOX[(d, i)]
    h, a = SCHED[d][i]
    blocks, score = [], {}
    for tid, opp in ((h, a), (a, h)):
        o, dd, pace = STRENGTH[tid][0], STRENGTH[opp][1], (STRENGTH[tid][2] + STRENGTH[opp][2]) / 2
        aths, tot = [], 0
        for j, p in enumerate(PLAYERS[tid]):
            if tid == "1" and j == 0 and d > TODAY - timedelta(days=5):
                aths.append({"athlete": {"id": p["id"], "displayName": p["name"]}, "didNotPlay": True, "stats": []})
                continue
            f = o * dd * pace / 100 * rng.lognormvariate(0, .25)
            pts = max(0, round(p["pts"] * f)); tot += pts
            fg3 = max(0, round(p["fg3"] * rng.lognormvariate(0, .5)))
            fga = round(pts / 1.1) + 1
            aths.append({"athlete": {"id": p["id"], "displayName": p["name"], "position": {"abbreviation": p["pos"]}},
                         "starter": j < 5, "didNotPlay": False,
                         "stats": [str(round(p["min"] * rng.uniform(.85, 1.1))), str(pts), f"{round(fga*.47)}-{fga}",
                                   f"{fg3}-{fg3*3}", "2-3", str(max(0, round(p["reb"] * rng.lognormvariate(0, .35)))),
                                   str(max(0, round(p["ast"] * rng.lognormvariate(0, .4)))), "2", "1", "0", "1", "3", "2", "+3"]})
        score[tid] = tot
        blk = {"keys": KEYS, "labels": LABELS, "athletes": aths} if rng.random() < .5 else {"labels": LABELS, "athletes": aths}
        blocks.append({"team": {"id": tid, "abbreviation": TEAMS[tid]["abbreviation"]}, "statistics": [blk]})
    BOX[(d, i)] = ({"boxscore": {"players": blocks}}, score)
    return BOX[(d, i)]


def fake_get_json(url, params=None, ttl=None):
    params = params or {}
    if "basketball" not in url:
        raise KeyError(url)
    if url.endswith("/scoreboard"):
        if "dates" not in params:
            return {"leagues": [{"season": {"year": 2027, "startDate": "2026-10-20T07:00Z", "type": 2}}], "events": []}
        d = datetime.strptime(str(params["dates"]), "%Y%m%d").date()
        evs = []
        for i, (h, a) in enumerate(SCHED.get(d, [])):
            done = d < TODAY
            sc = _box(d, i)[1] if done else {}
            tip = datetime(d.year, d.month, d.day, 23, 30, tzinfo=timezone.utc) + timedelta(minutes=30 * i)
            evs.append({"id": _gid(d, i), "date": tip.strftime("%Y-%m-%dT%H:%MZ"), "season": {"type": 2},
                        "competitions": [{"venue": {"fullName": f"{TEAMS[h]['abbreviation']} Arena", "address": {"city": "City"}},
                                          "status": {"type": {"completed": done}},
                                          "competitors": [{"homeAway": "home", "team": TEAMS[h], "score": str(sc.get(h, 0))},
                                                          {"homeAway": "away", "team": TEAMS[a], "score": str(sc.get(a, 0))}]}]})
        return {"events": evs}
    if url.endswith("/summary"):
        gid = params["event"]
        d = date(2026 if int(gid[5:7]) >= 9 else 2027, int(gid[5:7]), int(gid[7:9]))
        return _box(d, int(gid[9:]))[0]
    if url.endswith("/teams"):
        return {"sports": [{"leagues": [{"teams": [{"team": t} for t in TEAMS.values()]}]}]}
    if "/roster" in url:
        tid = url.split("/teams/")[1].split("/")[0]
        return {"athletes": [{"id": p["id"], "displayName": p["name"], "position": {"abbreviation": p["pos"]}, "injuries": []}
                             for p in PLAYERS[tid]]}
    if url.endswith("/injuries"):
        return {"injuries": [
            {"id": "1", "displayName": "ATL Team", "injuries": [
                {"athlete": {"id": "100", "displayName": "ATL P0"}, "status": "Out", "shortComment": "ankle"},
                {"athlete": {"id": "102", "displayName": "ATL P2"}, "status": "Day-To-Day", "shortComment": "knee"}]},
            {"id": "2", "displayName": "BOS Team", "injuries": [
                {"athlete": {"id": "201", "displayName": "BOS P1"}, "status": "Questionable", "shortComment": "back"}]}]}
    raise KeyError(url)


def install():
    nba_espn.get_json = fake_get_json
