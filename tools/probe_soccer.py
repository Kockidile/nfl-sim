import requests, json, re
K = "https://api.elections.kalshi.com/trade-api/v2"
s = requests.Session()
ser = s.get(f"{K}/series", params={"category": "Sports"}, timeout=60).json().get("series") or []
soc = [x for x in ser if "Soccer" in (x.get("tags") or [])]
games = sorted(x["ticker"] for x in soc if x["ticker"].endswith("GAME"))
print("GAMESER", len(games), " ".join(games))
cnt = {}
for t in games:
    e = s.get(f"{K}/events", params={"series_ticker": t, "status": "open", "limit": 200}, timeout=60).json().get("events") or []
    if e: cnt[t] = len(e)
print("OPEN", json.dumps(cnt))
for t in ["KXEPLGAME", "KXEPLTOTAL", "KXEPLBTTS", "KXUCLGAME", "KXEPLSPREAD"]:
    e = s.get(f"{K}/events", params={"series_ticker": t, "status": "open", "with_nested_markets": "true", "limit": 2}, timeout=60).json().get("events") or []
    for ev in e[:1]:
        print("\n== EV", t, ev.get("event_ticker"), "|", ev.get("title"), "|", ev.get("sub_title"), "|", ev.get("strike_date"))
        for m in (ev.get("markets") or [])[:6]:
            print("   M", m.get("ticker"), "|", m.get("yes_sub_title"), "|", m.get("yes_bid_dollars"), m.get("yes_ask_dollars"), m.get("volume_fp"), "|", m.get("occurrence_datetime"), m.get("expected_expiration_time"), "|", m.get("floor_strike"), m.get("strike_type"), "|", (m.get("rules_primary") or "")[:220])
E = "https://site.api.espn.com/apis/site/v2/sports/soccer"
for lg in ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "usa.1", "uefa.champions", "uefa.europa", "ned.1", "por.1"]:
    r = s.get(f"{E}/{lg}/scoreboard", params={"dates": "20261101-20261130", "limit": 500}, timeout=60)
    ev = r.json().get("events", []) if r.ok else []
    c = ev[0]["competitions"][0] if ev else {}
    print("ESPN", lg, r.status_code, len(ev), ev[0]["name"] if ev else "", ev[0]["date"] if ev else "", [ (x["team"]["abbreviation"], x.get("score"), x.get("homeAway")) for x in c.get("competitors", [])], c.get("status",{}).get("type",{}).get("name"))
