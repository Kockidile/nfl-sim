"""One-off: save samples of every data source so parsers can be written against real shapes."""
import csv, io, json, sys, datetime as dt
from pathlib import Path
import requests

out = Path("debug"); out.mkdir(exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0 (personal research)"
K = "https://api.elections.kalshi.com/trade-api/v2"
log = {}

def get(url, **p):
    try:
        r = S.get(url, params=p, timeout=30); log[url] = r.status_code
        return r
    except Exception as e:
        log[url] = str(e); return None

# Kalshi: series list, tennis series, open events + markets
r = get(f"{K}/series", category="Sports")
series = (r.json().get("series") if r is not None and r.ok else []) or []
tennis = [s for s in series if any(w in (json.dumps(s).lower()) for w in ("tennis", "atp", "wta", "itf", "challenger", "wimbledon", "us open", "australian open", "roland"))]
(out / "kalshi_series_tennis.json").write_text(json.dumps(tennis, indent=1)[:200000])
log["n_series_sports"] = len(series); log["n_series_tennis"] = len(tennis)
samples = {}
for s in tennis[:60]:
    t = s.get("ticker")
    r = get(f"{K}/events", series_ticker=t, status="open", with_nested_markets="true", limit=3)
    if r is not None and r.ok:
        ev = r.json().get("events") or []
        if ev:
            samples[t] = ev[:2]
(out / "kalshi_events_sample.json").write_text(json.dumps(samples, indent=1)[:600000])
log["series_with_open_events"] = list(samples)

# ESPN tennis
for tour in ("atp", "wta"):
    r = get(f"https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard")
    if r is not None and r.ok:
        (out / f"espn_{tour}_scoreboard.json").write_text(json.dumps(r.json(), indent=1)[:300000])
    d = (dt.date.today() - dt.timedelta(days=10)).strftime("%Y%m%d")
    r = get(f"https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard", dates=d)
    if r is not None and r.ok:
        (out / f"espn_{tour}_scoreboard_{d}.json").write_text(json.dumps(r.json(), indent=1)[:300000])

# Sackmann freshness
RAW = "https://raw.githubusercontent.com/JeffSackmann"
for repo, f in [("tennis_atp", "atp_matches_2026.csv"), ("tennis_atp", "atp_matches_qual_chall_2026.csv"),
                ("tennis_wta", "wta_matches_2026.csv"), ("tennis_wta", "wta_matches_qual_itf_2026.csv"),
                ("tennis_atp", "atp_matches_2025.csv")]:
    r = get(f"{RAW}/{repo}/master/{f}")
    if r is not None and r.ok:
        rows = list(csv.DictReader(io.StringIO(r.text)))
        dates = sorted(x.get("tourney_date", "") for x in rows)
        log[f] = {"rows": len(rows), "first": dates[:1], "last": dates[-1:], "cols": list(rows[0].keys()) if rows else [],
                  "last_tourneys": sorted({(x["tourney_date"], x["tourney_name"]) for x in rows})[-6:]}
(out / "log.json").write_text(json.dumps(log, indent=1))
print(json.dumps(log, indent=1)[:3000])
