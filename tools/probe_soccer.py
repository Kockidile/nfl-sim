import requests, json, re
K = "https://api.elections.kalshi.com/trade-api/v2"
s = requests.Session()
j = s.get(f"{K}/series", params={"category": "Sports"}, timeout=60).json()
ser = j.get("series") or []
print("sports series", len(ser))
soc = [x for x in ser if re.search(r"soccer|premier|la ?liga|serie a|bundesliga|ligue|mls|champions|europa|uefa|fifa|epl|football club|fc\b", (x.get("title","")+" "+x.get("ticker","")+" "+" ".join(x.get("tags") or [])), re.I)]
for x in soc:
    print("SER", x["ticker"], "|", x.get("title"), "|", x.get("tags"), "|", x.get("frequency"))
for x in soc:
    t = x["ticker"]
    e = s.get(f"{K}/events", params={"series_ticker": t, "status": "open", "with_nested_markets": "true", "limit": 3}, timeout=60).json().get("events") or []
    if not e: continue
    ev = e[0]
    print("\n== EV", t, ev.get("event_ticker"), ev.get("title"), ev.get("sub_title"), "n_open_events?", len(e))
    for m in (ev.get("markets") or [])[:4]:
        print("   M", m.get("ticker"), "|", m.get("yes_sub_title"), "|", m.get("yes_ask_dollars"), m.get("volume_fp"), "|", m.get("occurrence_datetime") or m.get("expected_expiration_time"), "|", (m.get("rules_primary") or "")[:200])
E = "https://site.api.espn.com/apis/site/v2/sports/soccer"
for lg in ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "usa.1", "uefa.champions", "uefa.europa"]:
    r = s.get(f"{E}/{lg}/scoreboard", params={"dates": "20261101-20261130", "limit": 500}, timeout=60)
    ev = r.json().get("events", []) if r.ok else []
    print("ESPN", lg, r.status_code, len(ev), ev[0]["name"] if ev else "", ev[0]["competitions"][0]["competitors"][0].get("score") if ev else "")
