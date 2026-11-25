"""ESPN soccer results/fixtures and Kalshi soccer markets (match winner, total goals, both teams to score)."""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import requests

CACHE = Path.home() / ".nfl_sim_cache" / "soccer"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi series prefix -> (ESPN league slug, display name)
LEAGUES = {
    "EPL": ("eng.1", "Premier League"), "LALIGA": ("esp.1", "La Liga"), "SERIEA": ("ita.1", "Serie A"),
    "BUNDESLIGA": ("ger.1", "Bundesliga"), "LIGUE1": ("fra.1", "Ligue 1"), "UCL": ("uefa.champions", "Champions League"),
    "UEL": ("uefa.europa", "Europa League"), "UECL": ("uefa.europa.conf", "Conference League"),
    "MLS": ("usa.1", "MLS"), "EREDIVISIE": ("ned.1", "Eredivisie"), "LIGAPORTUGAL": ("por.1", "Liga Portugal"),
    "EFLCHAMPIONSHIP": ("eng.2", "Championship"), "SCOTTISHPREM": ("sco.1", "Scottish Prem"),
    "LIGAMX": ("mex.1", "Liga MX"), "BRASILEIRO": ("bra.1", "Brasileirão"), "SUPERLIG": ("tur.1", "Süper Lig"),
    "BELGIANPL": ("bel.1", "Belgian Pro League"),
}
CUPS = {"uefa.champions", "uefa.europa", "uefa.europa.conf"}

_s = requests.Session()
_s.headers["User-Agent"] = "Mozilla/5.0 (soccer-sim)"


def _fetch(url, params=None, ttl=None):
    key = hashlib.md5((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists() and (ttl is None or time.time() - path.stat().st_mtime < ttl):
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    for attempt in range(4):
        try:
            r = _s.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1)); continue
            if r.status_code in (400, 404):
                return None
            r.raise_for_status()
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text)
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(1 + attempt)
    return None


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    drop = {"fc", "afc", "cf", "sc", "ac", "cd", "ud", "sd", "rc", "fk", "sk", "bk", "if", "club", "de", "the", "and", "calcio", "1", "us", "ss", "as", "cfc", "sv", "vfl", "vfb", "tsg", "rcd", "ca", "cr", "ec", "se"}
    return " ".join(w for w in s.split() if w not in drop)


def name_score(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.9
    ta, tb = set(a.split()), set(b.split())
    jac = len(ta & tb) / len(ta | tb)
    return max(jac, SequenceMatcher(None, a, b).ratio() * 0.95)


def _parse_event(ev: dict, slug: str) -> dict | None:
    try:
        c = ev["competitions"][0]
        teams = {x["homeAway"]: x for x in c["competitors"]}
        h, a = teams["home"], teams["away"]
        st = c["status"]["type"]
        row = {"id": ev["id"], "league": slug, "date": ev["date"], "state": st.get("state"), "status": st.get("name"),
               "neutral": bool(c.get("neutralSite")),
               "home_id": h["team"]["id"], "away_id": a["team"]["id"],
               "home": h["team"].get("displayName"), "away": a["team"].get("displayName"),
               "home_short": h["team"].get("shortDisplayName"), "away_short": a["team"].get("shortDisplayName"),
               "home_abbr": h["team"].get("abbreviation"), "away_abbr": a["team"].get("abbreviation"),
               "hg": None, "ag": None}
        if st.get("completed") and st.get("name") in ("STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_AET", "STATUS_FINAL_PEN", "STATUS_FT"):
            # 90-minute score: ESPN reports the AET score for extra-time games; keep those out of the goal model
            if st.get("name") in ("STATUS_FINAL_AET", "STATUS_FINAL_PEN"):
                return None
            row["hg"], row["ag"] = int(h.get("score") or 0), int(a.get("score") or 0)
        return row
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def scoreboard(slug: str, d: date, today: date) -> list[dict]:
    ttl = None if d < today - timedelta(days=3) else (3600 if d < today else 600)
    j = _fetch(f"{ESPN}/{slug}/scoreboard", {"dates": d.strftime("%Y%m%d")}, ttl=ttl) or {}
    return [r for r in (_parse_event(e, slug) for e in j.get("events") or []) if r]


def load_matches(days_back=270, days_ahead=2, verbose=True) -> list[dict]:
    today = datetime.now(timezone.utc).date()
    jobs = [(slug, today + timedelta(days=k)) for slug, _ in LEAGUES.values() for k in range(-days_back, days_ahead + 1)]
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda j: scoreboard(j[0], j[1], today), jobs))
    seen, out = set(), []
    for rows in res:
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"]); out.append(r)
    if verbose:
        print(f"soccer: {len(out)} ESPN matches from {len(jobs)} league-days")
    return out


# ------------------------------------------------------------------ Kalshi
def kalshi_markets(series: str) -> list[dict]:
    out, cursor = [], None
    for _ in range(10):
        p = {"series_ticker": series, "status": "open", "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        j = _fetch(f"{KALSHI}/markets", p, ttl=300) or {}
        out += j.get("markets") or []
        cursor = j.get("cursor")
        if not cursor or not j.get("markets"):
            break
        time.sleep(0.2)
    return out


def price(m: dict) -> dict:
    f = lambda k: float(m[k]) if m.get(k) not in (None, "") else None
    bid, ask, last = f("yes_bid_dollars"), f("yes_ask_dollars"), f("last_price_dollars")
    return {"bid": bid or 0.0, "ask": ask, "last": last, "volume": float(m.get("volume_fp") or 0)}


def load_kalshi(verbose=True) -> dict:
    """event key (e.g. 'EPL|26DEC02NEWMUN') -> {'league', 'teams': [name, name], 'win': {name: price}, 'tie', 'totals': {2.5: price}, 'btts'}"""
    ev: dict[str, dict] = {}
    for pre in LEAGUES:
        for kind in ("GAME", "TOTAL", "BTTS"):
            for m in kalshi_markets(f"KX{pre}{kind}"):
                suffix = m["event_ticker"].split("-", 1)[-1]
                e = ev.setdefault(f"{pre}|{suffix}", {"league": pre, "suffix": suffix, "teams": [], "win": {}, "tie": None,
                                                       "totals": {}, "btts": None, "time": m.get("occurrence_datetime")})
                pr = price(m)
                if kind == "GAME":
                    who = m.get("yes_sub_title") or ""
                    if who.lower() in ("tie", "draw"):
                        e["tie"] = pr
                    else:
                        e["win"][who] = pr
                        if who not in e["teams"]:
                            e["teams"].append(who)
                elif kind == "TOTAL" and m.get("floor_strike") is not None:
                    e["totals"][float(m["floor_strike"])] = pr
                elif kind == "BTTS":
                    e["btts"] = pr
                if not e["teams"]:
                    mt = re.search(r"in the (.+?) vs (.+?) (?:professional|[A-Z0-9 ]+ (?:soccer|match))", m.get("rules_primary") or "")
                    if mt:
                        e["rule_teams"] = [mt.group(1).strip(), mt.group(2).strip()]
    for e in ev.values():
        if len(e["teams"]) < 2 and e.get("rule_teams"):
            e["teams"] = e["rule_teams"]
    if verbose:
        print(f"soccer: {len(ev)} open Kalshi soccer events")
    return ev


def _ticker_date(suffix: str) -> date | None:
    m = re.match(r"(\d{2})([A-Z]{3})(\d{2})", suffix)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(0), "%y%b%d").date()
    except ValueError:
        return None


def match_kalshi(fixtures: list[dict], kal: dict) -> dict:
    """ESPN fixture id -> (kalshi event, home name on Kalshi, away name on Kalshi)"""
    slug_to_pre = {v[0]: k for k, v in LEAGUES.items()}
    out = {}
    for e in kal.values():
        d = _ticker_date(e["suffix"])
        if not d or len(e["teams"]) < 2:
            continue
        best, bs = None, 0.0
        for f in fixtures:
            if slug_to_pre.get(f["league"]) != e["league"]:
                continue
            fd = datetime.fromisoformat(f["date"].replace("Z", "+00:00")).date()
            if abs((fd - d).days) > 1:
                continue
            t0, t1 = e["teams"][:2]
            for hk, ak in ((t0, t1), (t1, t0)):
                s = min(max(name_score(hk, f["home"]), name_score(hk, f["home_short"] or "")),
                        max(name_score(ak, f["away"]), name_score(ak, f["away_short"] or "")))
                if s > bs:
                    best, bs, pair = f, s, (hk, ak)
        if best and bs >= 0.55:
            prev = out.get(best["id"])
            if not prev or prev[3] < bs:
                out[best["id"]] = (e, pair[0], pair[1], bs)
    return out
