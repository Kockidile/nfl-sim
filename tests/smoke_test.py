"""End-to-end run against the offline ESPN fake:  python tests/smoke_test.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_espn  # noqa: E402

fake_espn.install()
from nfl_sim.cli import main  # noqa: E402

for argv in (["games"],
             ["-q", "rankings", "--sort", "passdef"],
             ["injuries", "ARI"],
             ["-q", "players", "ARI"],
             ["-q", "sim", "--game", "ARI", "--n", "5000", "--seed", "3",
              "--line", "ARI RB1:rush_yds:60.5",
              "--parlay", "ML:ARI", "--parlay", "PROP:ARI RB1:rush_yds:over:50.5", "--parlay", "TOTAL:over:40.5"],
             ["-q", "sim", "--n", "2000", "--no-players"],
             ["-q", "backtest", "--start", "4"]):
    print(f"\n\n$ python -m nfl_sim {' '.join(argv)}")
    main(argv)
