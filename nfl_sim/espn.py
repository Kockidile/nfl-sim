"""ESPN data access: schedule/venues, box scores, rosters, injuries.

Uses ESPN's public (undocumented) JSON endpoints, the same ones espn.com loads.
They can change without notice, so every parser here is defensive: a missing
field gives a blank/zero rather than a crash. Responses are cached on disk;
finished games are cached forever, live data for minutes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CACHE_DIR = Path(os.environ.get("NFL_SIM_CACHE", Path.home() / ".nfl_sim_cache"))

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (personal nfl-sim research tool)"})


def get_json(url: str, params: dict | None = None, ttl: float | None = 3600) -> dict:
    """GET JSON with disk cache. ttl=None caches forever; stale cache is used if the network fails."""
    key = hashlib.md5((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists() and (ttl is None or time.time() - path.stat().st_mtime < ttl):
        return json.loads(path.read_text())
    last_err = None
    for attempt in range(3):
        try:
            r = _session.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            time.sleep(0.15)  # be polite
            return data
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    if path.exists():
        return json.loads(path.read_text())
    raise RuntimeError(f"Could not fetch {url}: {last_err}")


# --------------------------------------------------------------------------- data classes
@dataclass
class TeamRef:
    id: str
    abbr: str
    name: str
    score: int | None = None


@dataclass
class Game:
    id: str
    date: str            # ISO-8601 UTC kickoff
    season: int
    week: int
    home: TeamRef
    away: TeamRef
    completed: bool
    venue: str
    city: str
    state: str
    indoor: bool | None
    neutral: bool
    espn_weather: dict | None

    @property
    def label(self) -> str:
        return f"{self.away.abbr} @ {self.home.abbr}"


@dataclass
class Injury:
    player_id: str
    name: str
    pos: str
    status: str
    detail: str


# --------------------------------------------------------------------------- schedule
def current_week() -> tuple[int, int, int]:
    """(season year, week number, season type: 1 pre / 2 regular / 3 post)."""
    d = get_json(f"{SITE}/scoreboard", ttl=900)
    season = d.get("season") or {}
    week = d.get("week") or {}
    return int(season.get("year", 0)), int(week.get("number", 1)), int(season.get("type", 2))


def _ref(c: dict, completed: bool) -> TeamRef:
    t = c.get("team") or {}
    s = c.get("score")
    score = None
    if completed and s not in (None, ""):
        try:
            score = int(float(s if not isinstance(s, dict) else s.get("value", 0)))
        except (TypeError, ValueError):
            score = None
    return TeamRef(str(t.get("id", "")), t.get("abbreviation", "?"), t.get("displayName", "?"), score)


def get_week_games(season: int, week: int, seasontype: int = 2) -> list[Game]:
    d = get_json(f"{SITE}/scoreboard",
                 {"dates": season, "seasontype": seasontype, "week": week}, ttl=600)
    games = []
    for ev in d.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        status = (comp.get("status") or ev.get("status") or {}).get("type") or {}
        completed = bool(status.get("completed"))
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in sides or "away" not in sides:
            continue
        venue = comp.get("venue") or {}
        addr = venue.get("address") or {}
        games.append(Game(
            id=str(ev.get("id")),
            date=ev.get("date") or comp.get("date", ""),
            season=season, week=week,
            home=_ref(sides["home"], completed), away=_ref(sides["away"], completed),
            completed=completed,
            venue=venue.get("fullName", ""), city=addr.get("city", ""), state=addr.get("state", ""),
            indoor=venue.get("indoor"), neutral=bool(comp.get("neutralSite")),
            espn_weather=ev.get("weather") or comp.get("weather"),
        ))
    return games


def get_teams() -> dict[str, TeamRef]:
    d = get_json(f"{SITE}/teams", ttl=86400)
    out = {}
    for lg in (d.get("sports") or [{}])[0].get("leagues", []):
        for item in lg.get("teams", []):
            t = item.get("team") or item
            out[str(t.get("id"))] = TeamRef(str(t.get("id")), t.get("abbreviation", "?"), t.get("displayName", "?"))
    return out


# --------------------------------------------------------------------------- box scores
def get_summary(event_id: str, completed: bool) -> dict:
    return get_json(f"{SITE}/summary", {"event": event_id}, ttl=None if completed else 600)


def _num(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _pick(block: dict, stats: list, keys: list[str], labels: list[str]):
    k = block.get("keys") or []
    lab = block.get("labels") or []
    for key in keys:
        if key in k and k.index(key) < len(stats):
            return stats[k.index(key)]
    for name in labels:
        if name in lab and lab.index(name) < len(stats):
            return stats[lab.index(name)]
    return None


def parse_boxscore(summary: dict, game: Game) -> tuple[list[dict], list[dict]]:
    """Returns (player_rows, team_rows) for one completed game."""
    players: dict[tuple[str, str], dict] = {}
    for team_blk in (summary.get("boxscore") or {}).get("players", []):
        tid = str((team_blk.get("team") or {}).get("id", ""))
        opp = game.away.id if tid == game.home.id else game.home.id
        for blk in team_blk.get("statistics", []):
            cat = (blk.get("name") or "").lower()
            if cat not in ("passing", "rushing", "receiving"):
                continue
            for a in blk.get("athletes", []):
                ath = a.get("athlete") or {}
                pid = str(ath.get("id", ""))
                if not pid:
                    continue
                st = a.get("stats") or []
                row = players.setdefault((tid, pid), {
                    "game_id": game.id, "week": game.week, "team_id": tid, "opp_id": opp,
                    "player_id": pid, "name": ath.get("displayName", ""),
                    "pos": (ath.get("position") or {}).get("abbreviation", ""),
                    "pass_yds": 0.0, "pass_att": 0.0, "rush_yds": 0.0, "rush_att": 0.0,
                    "rec_yds": 0.0, "rec": 0.0, "tgt": 0.0})
                if cat == "passing":
                    row["pass_yds"] = _num(_pick(blk, st, ["passingYards"], ["YDS"]))
                    ca = str(_pick(blk, st, ["completions/passingAttempts"], ["C/ATT"]) or "0/0")
                    row["pass_att"] = _num(ca.split("/")[-1])
                elif cat == "rushing":
                    row["rush_yds"] = _num(_pick(blk, st, ["rushingYards"], ["YDS"]))
                    row["rush_att"] = _num(_pick(blk, st, ["rushingAttempts"], ["CAR"]))
                else:
                    row["rec_yds"] = _num(_pick(blk, st, ["receivingYards"], ["YDS"]))
                    row["rec"] = _num(_pick(blk, st, ["receptions"], ["REC"]))
                    row["tgt"] = _num(_pick(blk, st, ["receivingTargets"], ["TGTS"]))
    prow = list(players.values())

    teams = []
    for me, opp in ((game.home, game.away), (game.away, game.home)):
        mine = [r for r in prow if r["team_id"] == me.id]
        theirs = [r for r in prow if r["team_id"] == opp.id]
        teams.append({
            "game_id": game.id, "week": game.week, "team_id": me.id, "abbr": me.abbr,
            "opp_id": opp.id, "home": me is game.home,
            "pts": me.score or 0, "pa": opp.score or 0,
            "pass_yds": sum(r["pass_yds"] for r in mine), "rush_yds": sum(r["rush_yds"] for r in mine),
            "opp_pass_yds": sum(r["pass_yds"] for r in theirs),
            "opp_rush_yds": sum(r["rush_yds"] for r in theirs),
        })
    return prow, teams


# --------------------------------------------------------------------------- rosters & injuries
def _id_from_links(links) -> str:
    for ln in links or []:
        m = re.search(r"/id/(\d+)", ln.get("href", ""))
        if m:
            return m.group(1)
    return ""


def get_roster(team_id: str) -> dict[str, dict]:
    """player_id -> {name, pos, injury_status} for the current roster."""
    d = get_json(f"{SITE}/teams/{team_id}/roster", ttl=3600)
    groups = d.get("athletes", [])
    items = []
    for g in groups:
        items.extend(g.get("items", []) if isinstance(g, dict) and "items" in g else [g])
    out = {}
    for a in items:
        inj = (a.get("injuries") or [{}])
        out[str(a.get("id"))] = {
            "name": a.get("displayName") or a.get("fullName", ""),
            "pos": (a.get("position") or {}).get("abbreviation", ""),
            "injury_status": (inj[0] or {}).get("status", "") if inj else "",
        }
    return out


def get_injuries(teams: dict[str, TeamRef] | None = None) -> dict[str, list[Injury]]:
    """team_id -> injuries, from ESPN's league-wide injury report."""
    d = get_json(f"{SITE}/injuries", ttl=1800)
    by_name = {t.name: tid for tid, t in (teams or {}).items()}
    out: dict[str, list[Injury]] = {}
    for blk in d.get("injuries", []):
        tid = str(blk.get("id") or by_name.get(blk.get("displayName", ""), ""))
        for inj in blk.get("injuries", []):
            a = inj.get("athlete") or {}
            out.setdefault(tid, []).append(Injury(
                player_id=str(a.get("id") or _id_from_links(a.get("links"))),
                name=a.get("displayName", ""),
                pos=(a.get("position") or {}).get("abbreviation", ""),
                status=inj.get("status", ""),
                detail=inj.get("shortComment") or ((inj.get("details") or {}).get("type") or ""),
            ))
    return out
