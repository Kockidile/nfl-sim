import requests, json
K = "https://api.elections.kalshi.com/trade-api/v2"
s = requests.Session()
ser = s.get(f"{K}/series", params={"category": "Sports"}, timeout=60).json().get("series") or []
tk = {x["ticker"] for x in ser}
for lg in ["EPL","LALIGA","SERIEA","BUNDESLIGA","LIGUE1","UCL","UEL","UECL","MLS","EREDIVISIE","LIGAPORTUGAL","EFLCHAMPIONSHIP","SCOTTISHPREM","LIGAMX","BRASILEIRO","SUPERLIG","BELGIANPL"]:
    print("HAS", lg, [suf for suf in ["GAME","TOTAL","BTTS","SPREAD"] if f"KX{lg}{suf}" in tk])
for t in ["KXEPLGAME", "KXEPLTOTAL", "KXEPLBTTS"]:
    ms = s.get(f"{K}/markets", params={"series_ticker": t, "status": "open", "limit": 200}, timeout=60).json().get("markets") or []
    print("MK", t, len(ms))
    for m in ms[:7]:
        print("MM", json.dumps({k: m.get(k) for k in ["ticker","event_ticker","title","yes_sub_title","no_sub_title","yes_bid_dollars","yes_ask_dollars","volume_fp","occurrence_datetime","expected_expiration_time","floor_strike","strike_type","custom_strike"]}))
    if ms: print("RULE", (ms[0].get("rules_primary") or "")[:400])
E = "https://site.api.espn.com/apis/site/v2/sports/soccer"
for lg in ["eng.1", "uefa.champions"]:
    for d in ["20261122", "20261125", "20261126"]:
        r = s.get(f"{E}/{lg}/scoreboard", params={"dates": d}, timeout=60)
        for ev in (r.json().get("events", []) if r.ok else [])[:2]:
            c = ev["competitions"][0]
            print("ESPN", lg, d, ev["id"], ev["name"], ev["date"], c["status"]["type"]["name"], [(x["team"]["displayName"], x["team"]["abbreviation"], x.get("score"), x.get("homeAway")) for x in c["competitors"]])
