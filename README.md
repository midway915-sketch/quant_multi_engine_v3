
# quant_multi_engine_v3

This repo provides a simple grid runner with:
- look-ahead safe momentum signals (shift(1))
- weekly rebalance schedule (W-FRI)
- progress + ETA logging for grid runs
- artifacts including picks_top2_weekly.csv with realized returns

## Run locally
python -m scripts.run_grid --config config/default.yml --grid config/grid.yml --out out/grid_run --save-picks
