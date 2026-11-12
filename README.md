# NFL Sim — game & player-prop Monte Carlo engine

## Phone web app

The app lives in `docs/` and is served by GitHub Pages. A GitHub Action
(`.github/workflows/update-data.yml`) pulls fresh ESPN, injury and weather
data every 3 hours (hourly on Sundays), runs the model, and commits
`docs/data.json`. The page then runs the simulations on your phone, so
injury changes and parlay legs re-price instantly.

- **Refresh now:** GitHub → Actions → *Update NFL data* → *Run workflow*.
- **Add to Home Screen:** open the site in Safari → Share → *Add to Home Screen*.


Pulls live data from ESPN (schedule, venues, box scores, rosters, injuries) and
Open-Meteo (kickoff weather), builds opponent-adjusted offense/defense ratings,
and simulates every game 20,000 times to produce win probabilities, fair
spreads/totals, player yardage distributions, prop probabilities, and
correlated same-game-parlay odds.

No API keys needed.

## Install

```bash
cd nfl_sim
python -m pip install -r requirements.txt
```

## Commands

```bash
python -m nfl_sim games                      # this week's matchups, venue, kickoff weather
python -m nfl_sim rankings --sort passdef    # team ranks: off|def|pass|rush|passdef|rushdef
python -m nfl_sim injuries KC                # injury report + play probability
python -m nfl_sim players KC                 # weekly pass/rush/rec averages (season + last 3)
python -m nfl_sim sim                        # simulate the whole slate
python -m nfl_sim sim --game KC BUF --csv out/
python -m nfl_sim sim --game KC --line "Patrick Mahomes:pass_yds:245.5"
python -m nfl_sim sim --game KC \
    --parlay ML:KC --parlay "PROP:Patrick Mahomes:pass_yds:over:245.5" --parlay TOTAL:over:44.5
python -m nfl_sim backtest --start 4         # score the model on past weeks
python -m nfl_sim --week 12 sim              # any week; --season 2025 for past seasons
```

Stats: `pass_yds`, `rush_yds`, `rec_yds`.
Parlay legs: `ML:TEAM`, `SPREAD:TEAM:-3.5`, `TOTAL:over:47.5`, `TEAMTOTAL:TEAM:over:24.5`,
`PROP:Player Name:stat:over|under:line` (fuzzy name matching).

The first run downloads every completed box score of the season (~1–2 min).
After that, finished games come from the cache (`~/.nfl_sim_cache`, override with `NFL_SIM_CACHE`).

## How the model works

1. **Team ratings**: points, passing yards and rushing yards for and against,
   *opponent-adjusted* (300 yards against the #1 pass defense counts for more than 300 against #32),
   regressed toward league average early in the season.
2. **Matchup**: team offense × opponent defense ÷ league average → expected team yards and points.
   Each player's recency-weighted average is scaled by that matchup factor.
3. **Weather**: wind above 10 mph, cold below 25°F and rain cut passing (and scoring) and slightly
   boost rushing. Domes and retractable roofs are weather-neutral. Neutral-site/international
   games are geocoded from ESPN's venue city.
4. **Injuries**: each simulation samples who plays (Out 0%, Doubtful 15%, Questionable 75%…).
   A missing player's volume is redistributed to teammates; a missing starting QB hands passing
   to the backup at 80% and reduces receiver yards and team points. Players who already missed
   their team's last two games are treated as priced in. Opposing defenders ruled out this week add points.
5. **Correlation**: shared "team form" and "game pace" draws tie a QB to his receivers and to the
   team's score, so same-game parlays get a real joint probability instead of multiplying legs.

All tuning knobs are in `nfl_sim/config.py`. Change them, then re-run `backtest` to see whether
Brier score, spread/total error, and player-yard error improve.

## Testing offline

`python tests/smoke_test.py` runs every command against a fake ESPN that returns the same
JSON shapes, without needing the network.

## Caveats

- ESPN's JSON endpoints are public but undocumented, so they can change. Parsers are defensive.
  If something comes back empty, the likely fix is a field name in `espn.py`.
- Players with no stats this season (new signings, rookies stepping in) aren't projected,
  except a backup QB placeholder.
- Defensive injuries are weighted roughly, because box scores don't show who starts on defense.
- This is a research tool. The starting weights are reasonable guesses, not fitted values, so
  backtest before trusting any edge.
