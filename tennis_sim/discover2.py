import json, requests, csv, io
from pathlib import Path
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"
out = {}
for repo in ("JeffSackmann/tennis_atp", "JeffSackmann/tennis_wta", "Aneeshers/tennis-sackmann-archive"):
    r = S.get(f"https://api.github.com/repos/{repo}", timeout=30)
    info = r.json() if r.ok else {"status": r.status_code}
    out[repo] = {k: info.get(k) for k in ("default_branch", "pushed_at", "updated_at", "status", "message")}
    br = info.get("default_branch", "master")
    t = S.get(f"https://api.github.com/repos/{repo}/git/trees/{br}?recursive=1", timeout=30)
    if t.ok:
        files = [x["path"] for x in t.json().get("tree", []) if x["path"].endswith(".csv")]
        out[repo]["n_csv"] = len(files)
        out[repo]["recent"] = sorted([f for f in files if "2025" in f or "2026" in f])[:40]
    c = S.get(f"https://api.github.com/repos/{repo}/commits", params={"per_page": 3}, timeout=30)
    if c.ok:
        out[repo]["commits"] = [(x["commit"]["committer"]["date"], x["commit"]["message"][:80]) for x in c.json()]
# sample the newest match file found
for repo, info in out.items():
    br = "master"
    for f in reversed(info.get("recent", [])):
        if "matches" in f and ("2026" in f):
            r = S.get(f"https://raw.githubusercontent.com/{repo}/{(S.get(f'https://api.github.com/repos/{repo}').json() or {}).get('default_branch','master')}/{f}", timeout=60)
            if r.ok:
                rows = list(csv.DictReader(io.StringIO(r.text)))
                ds = sorted(x.get("tourney_date","") for x in rows)
                info.setdefault("samples", {})[f] = {"rows": len(rows), "last": ds[-1:], "cols": list(rows[0])[:60] if rows else []}
            else:
                info.setdefault("samples", {})[f] = r.status_code
# ESPN: does a date in mid-season return events?
for d in ("20260720", "20260315"):
    r = S.get("https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard", params={"dates": d}, timeout=30)
    if r.ok:
        j = r.json(); out[f"espn_{d}"] = [(e.get("name"), e.get("date"), sum(len(g.get("competitions", [])) for g in e.get("groupings", []))) for e in j.get("events", [])]
        cal = (j.get("leagues") or [{}])[0].get("calendar") or []
        out["espn_calendar_len"] = len(cal); out["espn_calendar_sample"] = cal[:3]
Path("debug/log2.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str)[:4000])
