"""ESPN NBA data: schedule, box scores, rosters, injuries.

Reuses the cached, defensive fetcher from nfl_sim.espn.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from nfl_sim.espn import Injury, TeamRef, _id_from_links, get_json

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ET = ZoneInfo("America/New_York")


@dataclass
class Game:
    id: str
    date: str          # ISO UTC tip-off
    day: date          # US Eastern calendar date
    home: TeamRef
    away: TeamRef
    completed: bool
    season_type: int
    venue: str
    city: str
    neutral: bool

    @property
    def label(self) -> str:
        return f"{self.away.abbr} @ {self.home.abbr}"


def today_et() -> date:
    return datetime.now(ET).date()


def season_start() -> date:
    d = get_json(f"{SITE}/scoreboard", ttl=6 * 3600)
    lg = (d.get("leagues") or [{}])[0]
    for key in (("season", "startDate"), ("calendarStartDate",)):
        v = lg
        for k in key:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        if isinstance(v, str) and len(v) >= 10:
            return date.fromisoformat(v[:10])
    t = today_et()
    return date(t.year if t.month >= 9 else t.year - 1, 10, 1)


def _ref(c: dict, completed: bool) -> TeamRef:
    t = c.get("team") or {}
    s = c.get("score")
    try:
        score = int(float(s)) if completed and s not in (None, "") else None
    except (TypeError, ValueError):
        score = None
    return TeamRef(str(t.get("id", "")), t.get("abbreviation", "?"), t.get("displayName", "?"), score)


def games_on(day: date) -> list[Game]:
    past = day < today_et() - timedelta(days=1)
    d = get_json(f"{SITE}/scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": 50},
                 ttl=None if past else 600)
    out = []
    for ev in d.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        status = (comp.get("status") or ev.get("status") or {}).get("type") or {}
        completed = bool(status.get("completed"))
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in sides or "away" not in sides:
            continue
        venue = comp.get("venue") or {}
        iso = ev.get("date") or comp.get("date", "")
        try:
            et_day = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(ET).date()
        except ValueError:
            et_day = day
        out.append(Game(
            id=str(ev.get("id")), date=iso, day=et_day,
            home=_ref(sides["home"], completed), away=_ref(sides["away"], completed),
            completed=completed, season_type=int((ev.get("season") or {}).get("type", 2) or 2),
            venue=venue.get("fullName", ""), city=(venue.get("address") or {}).get("city", ""),
            neutral=bool(comp.get("neutralSite")),
        ))
    return out


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
        return float(str(x).split(":")[0].replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _made_att(x) -> tuple[float, float]:
    parts = str(x or "0-0").split("-")
    try:
        return float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0
    except ValueError:
        return 0.0, 0.0


FIELDS = {  # our name: (ESPN keys, ESPN labels)
    "min": (["minutes"], ["MIN"]),
    "pts": (["points"], ["PTS"]),
    "reb": (["rebounds", "totalRebounds"], ["REB"]),
    "ast": (["assists"], ["AST"]),
    "oreb": (["offensiveRebounds"], ["OREB"]),
    "tov": (["turnovers"], ["TO"]),
    "fg": (["fieldGoalsMade-fieldGoalsAttempted"], ["FG"]),
    "fg3": (["threePointFieldGoalsMade-threePointFieldGoalsAttempted"], ["3PT"]),
    "ft": (["freeThrowsMade-freeThrowsAttempted"], ["FT"]),
}


def _idx(blk: dict, name: str) -> int | None:
    keys, labels = FIELDS[name]
    k, lab = blk.get("keys") or [], blk.get("labels") or []
    for key in keys:
        if key in k:
            return k.index(key)
    for l in labels:
        if l in lab:
            return lab.index(l)
    return None


def parse_boxscore(summary: dict, game: Game) -> tuple[list[dict], list[dict]]:
    players, sums = [], {}
    for team_blk in (summary.get("boxscore") or {}).get("players", []):
        tid = str((team_blk.get("team") or {}).get("id", ""))
        opp = game.away.id if tid == game.home.id else game.home.id
        blk = (team_blk.get("statistics") or [{}])[0]
        idx = {f: _idx(blk, f) for f in FIELDS}
        agg = sums.setdefault(tid, {"fga": 0.0, "fta": 0.0, "oreb": 0.0, "tov": 0.0, "reb": 0.0, "ast": 0.0, "fg3": 0.0})
        for a in blk.get("athletes", []):
            st = a.get("stats") or []
            if a.get("didNotPlay") or not st:
                continue
            get = lambda f: st[idx[f]] if idx[f] is not None and idx[f] < len(st) else None
            mins = _num(get("min"))
            if mins <= 0:
                continue
            fgm, fga = _made_att(get("fg"))
            f3m, _ = _made_att(get("fg3"))
            _, fta = _made_att(get("ft"))
            ath = a.get("athlete") or {}
            row = {"game_id": game.id, "day": game.day.isoformat(), "team_id": tid, "opp_id": opp,
                   "player_id": str(ath.get("id", "")), "name": ath.get("displayName", ""),
                   "pos": (ath.get("position") or {}).get("abbreviation", ""), "starter": bool(a.get("starter")),
                   "min": mins, "pts": _num(get("pts")), "reb": _num(get("reb")), "ast": _num(get("ast")), "fg3": f3m}
            players.append(row)
            agg["fga"] += fga; agg["fta"] += fta; agg["oreb"] += _num(get("oreb")); agg["tov"] += _num(get("tov"))
            agg["reb"] += row["reb"]; agg["ast"] += row["ast"]; agg["fg3"] += f3m

    teams = []
    for me, opp in ((game.home, game.away), (game.away, game.home)):
        a, b = sums.get(me.id), sums.get(opp.id)
        if not a or not b:
            continue
        poss = lambda s: s["fga"] - s["oreb"] + s["tov"] + 0.44 * s["fta"]
        teams.append({"game_id": game.id, "day": game.day.isoformat(), "team_id": me.id, "abbr": me.abbr,
                      "opp_id": opp.id, "home": me is game.home, "pts": me.score or 0, "pa": opp.score or 0,
                      "poss": (poss(a) + poss(b)) / 2,
                      "reb": a["reb"], "ast": a["ast"], "fg3": a["fg3"],
                      "opp_reb": b["reb"], "opp_ast": b["ast"], "opp_fg3": b["fg3"]})
    return players, teams


# --------------------------------------------------------------------------- rosters & injuries
def get_roster(team_id: str) -> dict[str, dict]:
    d = get_json(f"{SITE}/teams/{team_id}/roster", ttl=3600)
    items = []
    for g in d.get("athletes", []):
        items.extend(g.get("items", []) if isinstance(g, dict) and "items" in g else [g])
    out = {}
    for a in items:
        inj = a.get("injuries") or []
        out[str(a.get("id"))] = {"name": a.get("displayName") or a.get("fullName", ""),
                                 "pos": (a.get("position") or {}).get("abbreviation", ""),
                                 "injury_status": (inj[0] or {}).get("status", "") if inj else ""}
    return out


def get_injuries(teams: dict[str, TeamRef] | None = None) -> dict[str, list[Injury]]:
    d = get_json(f"{SITE}/injuries", ttl=900)
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
