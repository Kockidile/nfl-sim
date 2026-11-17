"""Tennis match history from three sources, merged into one table.

1. Jeff Sackmann's ATP/WTA archive (GitHub mirror): every level incl. Challengers/ITF,
   with serve stats. Static snapshot that ends in early June 2026.
2. ESPN scoreboards: tour-level (ATP/WTA) results after the archive ends, with set scores.
3. Kalshi settled match markets: winners of Challenger/ITF/tour matches after the archive ends.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(os.environ.get("NFL_SIM_CACHE", Path.home() / ".nfl_sim_cache")) / "tennis"
ARCHIVE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
YEARS = (2023, 2024, 2025, 2026)
FILES = {"M": ["atp/atp_matches_{y}.csv", "atp/atp_matches_qual_chall_{y}.csv", "atp/atp_matches_futures_{y}.csv"],
         "W": ["wta/wta_matches_{y}.csv", "wta/wta_matches_qual_itf_{y}.csv"]}
STAT_COLS = ["svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]

_s = requests.Session()
_s.headers["User-Agent"] = "Mozilla/5.0 (personal tennis research)"


# --------------------------------------------------------------------------- utils
def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[-.']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _fetch(url, params=None, ttl=None, as_json=True, pause=0.0):
    key = hashlib.md5((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = CACHE / (key + (".json" if as_json else ".txt"))
    if path.exists() and (ttl is None or time.time() - path.stat().st_mtime < ttl):
        txt = path.read_text()
        return json.loads(txt) if as_json else txt
    for attempt in range(5):
        try:
            r = _s.get(url, params=params, timeout=40)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1)); continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text)
            if pause:
                time.sleep(pause)
            return r.json() if as_json else r.text
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    if path.exists():
        txt = path.read_text()
        return json.loads(txt) if as_json else txt
    return None


SURF_KEYWORDS = {
    "Clay": ["roland garros", "french open", "monte carlo", "monte-carlo", "madrid", "rome", "italian", "internazionali",
             "barcelona", "hamburg", "gstaad", "kitzbuhel", "umag", "bastad", "nordea", "estoril", "generali", "bucharest",
             "marrakech", "houston", "geneva", "lyon", "munich", "rabat", "strasbourg", "parma", "palermo", "prague",
             "iasi", "rio", "buenos aires", "santiago", "cordoba", "bogota", "charleston", "sao paulo", "lausanne",
             "budapest", "warsaw", "bad gastein", "rouen", "oeiras", "cagliari", "aix", "bordeaux", "heilbronn"],
    "Grass": ["wimbledon", "halle", "queen", "hertogenbosch", "libema", "mallorca", "eastbourne", "boss open",
              "terra wortmann", "berlin", "bad homburg", "nottingham", "birmingham", "ilkley", "surbiton", "newport"],
}


def surface_from_name(name: str, archive_map: dict | None = None) -> str | None:
    n = " " + norm(name) + " "
    has = lambda w: f" {norm(w)} " in n
    for surf, words in SURF_KEYWORDS.items():   # well-known events first
        if any(has(w) for w in words):
            return surf
    if archive_map:
        for tok in sorted(archive_map, key=len, reverse=True):
            if tok and has(tok):
                return archive_map[tok]
    return None


def _sets(score: str):
    out = []
    for part in str(score).split():
        m = re.match(r"^(\d+)-(\d+)", part)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


# --------------------------------------------------------------------------- 1. archive
def load_archive() -> pd.DataFrame:
    frames = []
    for tour, pats in FILES.items():
        for y in YEARS:
            for p in pats:
                txt = _fetch(f"{ARCHIVE}/{p.format(y=y)}", as_json=False)
                if not txt:
                    continue
                df = pd.read_csv(io.StringIO(txt), low_memory=False)
                df["tour"] = tour
                frames.append(df)
    a = pd.concat(frames, ignore_index=True)
    a = a[~a.score.astype(str).str.contains("W/O|DEF|Walkover|unfinished", case=False, na=True)]
    a["retired"] = a.score.astype(str).str.contains("RET", case=False)
    a["date"] = pd.to_datetime(a.tourney_date.astype(str), format="%Y%m%d", errors="coerce")
    # matches inside a tournament share a date; order by round so ratings update in sequence
    order = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "R128": 4, "R64": 5, "R32": 6, "R16": 7, "RR": 8, "QF": 9, "SF": 10, "BR": 11, "F": 12}
    a["date"] = a.date + pd.to_timedelta(a["round"].map(order).fillna(6), unit="D") * 0.5
    out = pd.DataFrame({
        "date": a.date, "tour": a.tour, "source": "archive", "level": a.tourney_level.astype(str),
        "tourney": a.tourney_name.astype(str), "surface": a.surface.fillna("Hard").replace({"Carpet": "Hard"}),
        "best_of": a.best_of.fillna(3).astype(int), "w_name": a.winner_name, "l_name": a.loser_name,
        "score": a.score.astype(str), "retired": a.retired, "minutes": pd.to_numeric(a.minutes, errors="coerce"),
        "w_rank": pd.to_numeric(a.winner_rank, errors="coerce"), "l_rank": pd.to_numeric(a.loser_rank, errors="coerce"),
    })
    for side in ("w", "l"):
        for c in STAT_COLS:
            out[f"{side}_{c}"] = pd.to_numeric(a[f"{side}_{c}"], errors="coerce")
    return out.dropna(subset=["date"])


def archive_surface_map(arch: pd.DataFrame) -> dict:
    m = {}
    for name, surf in arch.sort_values("date")[["tourney", "surface"]].drop_duplicates("tourney", keep="last").itertuples(index=False):
        tok = re.sub(r"\b(ch|cup|open|masters|w\d+|m\d+|itf|wta|atp|challenger)\b", " ", norm(name))
        tok = re.sub(r"\s+", " ", tok).strip()
        if len(tok) >= 4:
            m[tok] = surf
    return m


# --------------------------------------------------------------------------- 2. ESPN tour results
def load_espn(since: date, surf_map: dict) -> pd.DataFrame:
    rows, seen = [], set()
    today = datetime.now(timezone.utc).date()
    for tour, code in (("M", "atp"), ("W", "wta")):
        d = since
        while d <= today:
            ttl = None if d < today - timedelta(days=4) else 3 * 3600
            j = _fetch(ESPN.format(tour=code), {"dates": d.strftime("%Y%m%d")}, ttl=ttl) or {}
            for ev in j.get("events", []):
                tname = ev.get("name", "")
                for gr in ev.get("groupings", []):
                    gname = (gr.get("grouping") or {}).get("displayName", "")
                    if "Singles" not in gname:
                        continue
                    for c in gr.get("competitions", []):
                        if c.get("id") in seen or not ((c.get("status") or {}).get("type") or {}).get("completed"):
                            continue
                        comps = [x for x in c.get("competitors", []) if x.get("type") == "athlete"]
                        if len(comps) != 2 or not any(x.get("winner") for x in comps):
                            continue
                        seen.add(c.get("id"))
                        w = next(x for x in comps if x.get("winner"))
                        l = next(x for x in comps if x is not w)
                        ws = [int(s.get("value", 0)) for s in w.get("linescores", [])]
                        ls = [int(s.get("value", 0)) for s in l.get("linescores", [])]
                        note = " ".join(n.get("text", "") for n in c.get("notes", []))
                        if "walkover" in note.lower() or not ws:
                            continue
                        rows.append({"date": pd.to_datetime(c.get("date")).tz_localize(None), "tour": tour, "source": "espn",
                                     "level": "T", "tourney": tname, "surface": surface_from_name(tname, surf_map) or "Hard",
                                     "best_of": 5 if (tour == "M" and (c.get("format") or {}).get("regulation", {}).get("periods") == 5 and ev.get("major")) else 3,
                                     "w_name": (w.get("athlete") or {}).get("displayName"), "l_name": (l.get("athlete") or {}).get("displayName"),
                                     "score": " ".join(f"{a}-{b}" for a, b in zip(ws, ls)), "retired": "ret" in note.lower(),
                                     "w_rank": (w.get("curatedRank") or {}).get("current"), "l_rank": (l.get("curatedRank") or {}).get("current")})
            d += timedelta(days=3)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3. Kalshi settled results
MATCH_SERIES = {"KXATPMATCH": "M", "KXWTAMATCH": "W", "KXATPCHALLENGERMATCH": "M", "KXCHALLENGERMATCH": "M",
                "KXWTACHALLENGERMATCH": "W", "KXITFMATCH": "M", "KXITFWMATCH": "W", "KXUNITEDCUPMATCH": "?",
                "KXDAVISCUPMATCH": "M", "KXLAVERCUPMATCH": "M", "KXTENNISEXHIBITION": "?", "KXEXHIBITIONMEN": "M"}
RULE_RE = re.compile(r"If (.+?) wins the (.+?) professional tennis match in the (\d{4}) (.+?) after", re.S)


def kalshi_events(series: str, status: str, max_pages=60, stop_before: datetime | None = None, ttl=3600):
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"series_ticker": series, "status": status, "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        j = _fetch(f"{KALSHI}/events", params, ttl=ttl, pause=0.35) or {}
        evs = j.get("events") or []
        out += evs
        cursor = j.get("cursor")
        if not cursor or not evs:
            break
        if stop_before:
            ds = [_event_time(e) for e in evs]
            ds = [d for d in ds if d]
            if ds and max(ds) < stop_before:
                break
    return out


def _event_time(ev):
    for m in ev.get("markets") or []:
        for k in ("occurrence_datetime", "expected_expiration_time"):
            if m.get(k):
                try:
                    return datetime.fromisoformat(m[k].replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    pass
    return None


def parse_match_event(ev: dict) -> dict | None:
    """Turn a Kalshi match event into {players:[a,b], tourney, start, markets:{name:market}}."""
    mk = [m for m in ev.get("markets") or [] if m.get("market_type", "binary") == "binary"]
    if len(mk) != 2 or "/" in (ev.get("title") or ""):
        return None
    names, tourney = [], ""
    for m in mk:
        r = RULE_RE.search(m.get("rules_primary", ""))
        nm = (r.group(1) if r else None) or m.get("yes_sub_title")
        if r:
            tourney = r.group(4)
        names.append(nm)
    if not all(names):
        return None
    return {"event": ev.get("event_ticker"), "title": ev.get("title"), "players": names, "tourney": tourney,
            "start": _event_time(ev), "markets": dict(zip(names, mk))}


def price(m: dict) -> dict:
    f = lambda k: float(m[k]) if m.get(k) not in (None, "") else None
    bid, ask, last = f("yes_bid_dollars"), f("yes_ask_dollars"), f("last_price_dollars")
    if bid is None and m.get("yes_bid") is not None:  # older cent fields
        bid, ask, last = m.get("yes_bid", 0) / 100, m.get("yes_ask", 0) / 100, m.get("last_price", 0) / 100
    mid = (bid + ask) / 2 if bid and ask and ask > bid else (last or None)
    return {"bid": bid, "ask": ask, "last": last, "mid": mid,
            "volume": float(m.get("volume_fp") or m.get("volume") or 0)}


def load_kalshi_results(since: datetime, surf_map: dict) -> pd.DataFrame:
    rows = []
    for series, tour in MATCH_SERIES.items():
        for ev in kalshi_events(series, "settled", stop_before=since, ttl=12 * 3600):
            p = parse_match_event(ev)
            if not p or not p["start"] or p["start"] < since:
                continue
            res = {n: (m.get("result") or "").lower() for n, m in p["markets"].items()}
            win = [n for n, r in res.items() if r == "yes"]
            lose = [n for n, r in res.items() if r == "no"]
            if len(win) != 1 or len(lose) != 1:
                continue  # voided / 50-50 settlements
            rows.append({"date": p["start"], "tour": tour, "source": "kalshi", "level": series, "tourney": p["tourney"],
                         "surface": surface_from_name(p["tourney"], surf_map) or "Hard",
                         "best_of": 5 if tour == "M" and re.search(r"australian open|roland|french open|wimbledon|us open", p["tourney"], re.I) else 3,
                         "w_name": win[0], "l_name": lose[0], "score": "", "retired": False})
    return pd.DataFrame(rows)


def load_all(verbose=True) -> tuple[pd.DataFrame, dict, date]:
    arch = load_archive()
    smap = archive_surface_map(arch)
    end = arch.date.max().date()
    if verbose:
        print(f"  archive: {len(arch):,} matches through {end}")
    frames = [arch]
    try:
        e = load_espn(end - timedelta(days=2), smap)
        frames.append(e)
        if verbose:
            print(f"  ESPN tour results since archive: {len(e):,}")
    except Exception as ex:  # noqa: BLE001
        print("  ESPN failed:", ex)
    try:
        k = load_kalshi_results(datetime.combine(end, datetime.min.time()), smap)
        frames.append(k)
        if verbose:
            print(f"  Kalshi settled results since archive: {len(k):,}")
    except Exception as ex:  # noqa: BLE001
        print("  Kalshi results failed:", ex)
    allm = pd.concat(frames, ignore_index=True)
    allm["w_key"] = allm.w_name.map(norm)
    allm["l_key"] = allm.l_name.map(norm)
    allm = allm[(allm.w_key != "") & (allm.l_key != "") & (allm.w_key != allm.l_key)]
    # de-duplicate the same match reported by ESPN and Kalshi
    allm["pair"] = [" | ".join(sorted(p)) for p in zip(allm.w_key, allm.l_key)]
    allm["day"] = pd.to_datetime(allm.date).dt.floor("D")
    allm = allm.sort_values(["date", "source"])
    dup = allm[allm.source != "archive"].duplicated(subset=["pair"], keep="first") & \
        allm[allm.source != "archive"].groupby("pair").day.diff().abs().le(pd.Timedelta(days=3))
    allm = allm.drop(index=dup[dup].index)
    return allm.sort_values("date").reset_index(drop=True), smap, end
