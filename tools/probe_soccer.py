import requests, json, time
K = "https://api.elections.kalshi.com/trade-api/v2"
s = requests.Session()
for t in ["KXEPLGAME", "KXEPLTOTAL", "KXUCLGAME", "KXSERIEAGAME", "KXLALIGAGAME"]:
    j = s.get(f"{K}/markets", params={"series_ticker": t, "status": "open", "limit": 200}, timeout=60).json()
    ms = j.get("markets") or []
    print("\n== MK", t, "open markets", len(ms))
    for m in ms[:7]:
        print("   M", m.get("ticker"), "|", m.get("event_ticker"), "|", m.get("title"), "|", m.get("yes_sub_title"), "|", m.get("yes_bid_dollars"), m.get("yes_ask_dollars"), m.get("volume_fp"), "|", m.get("occurrence_datetime"), m.get("expected_expiration_time"), m.get("close_time"), "|", m.get("floor_strike"), m.get("strike_type"), "|", (m.get("rules_primary") or "")[:260])
    j = s.get(f"{K}/events", params={"series_ticker": t, "limit": 5}, timeout=60).json()
    print("   EVENTS(any status)", [(e.get("event_ticker"), e.get("sub_title")) for e in (j.get("events") or [])[:5]])
E = "https://site.api.espn.com/apis/site/v2/sports/soccer"
for lg in ["eng.1", "uefa.champions", "ita.1"]:
    for params in ({"dates": "20261122"}, {"dates": "20261101-20261124"}, {}):
        r = s.get(f"{E}/{lg}/scoreboard", params=params, timeout=60)
        ev = r.json().get("events", []) if r.ok else []
        c = ev[0]["competitions"][0] if ev else {}
        print("ESPN", lg, params, r.status_code, len(ev), ev[0]["name"] if ev else r.text[:120], ev[0]["date"] if ev else "", [(x["team"]["abbreviation"], x.get("score"), x.get("homeAway")) for x in c.get("competitors", [])], c.get("status", {}).get("type", {}).get("name"))
r = s.get("https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings", timeout=60); print("STAND", r.status_code, r.text[:200])
