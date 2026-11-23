# PAST

Official code for **PAST: Posterior-Conditioned Residual Learning for Long-Horizon Ballistic Trajectory Prediction From Short Radar Tracks** (IEEE Transactions on Aerospace and Electronic Systems, under review).

PAST forecasts a ballistic trajectory to impact from a 1 s radar track. A drag-estimating unscented Kalman filter (UKF) supplies a physical forecast (the anchor) and its posterior; a sequence network learns the open-loop correction of the anchor, conditioned on anchor context and filter uncertainty. Sensor-randomized re-observation trains the correction across radar conditions. The residual interface accepts any backbone; validation selects TSMixer (PAST-TSMixer).

On the paper's high-fidelity benchmark, PAST-TSMixer reaches 17.5 m average displacement error versus 39.9 m for the drag-UKF, lowers 90th-percentile impact-point error from 67.1 to 52.6 m, and lowers mean time-to-impact error from 0.229 to 0.079 s.

## Contents

| Path | Role |
|---|---|
| `utils/radar.py` | Polar radar measurements, converted-measurement CA Kalman tracker, sensor-randomized re-observation (`radar_track_window(..., sensor_rng=...)`) |
| `utils/ballistic_physics.py` | Drag point-mass model, RK4, drag-UKF (`ukf_drag_track`), physical baselines |
| `tools/physics_baselines.py` | Baselines B1-B6, UKF-G (`--b7`), MM-UKF-G (`--b8`), UKF-GBM (`--b9`), drag-UKF anchors and posterior (uncertainty) features |
| `data_provider/` | Kinematic descriptors, anchor context, uncertainty features, train-only normalization |
| `models/` | TSMixer, PNMixer (`--model PASTMixer`), Transformer, TimeMixer, LSTM, iTransformer behind one residual interface; CE-LSTM comparison |
| `exp/`, `run.py` | Training with the masked physical-unit loss (RMSE + 0.5 ADE) and evaluation |
| `tools/summarize.py`, `tools/impact_regions.py` | Paired bootstrap confidence intervals; split-conformal impact regions |
| `configs/paper_protocol.json` | Paper settings: benchmark, anchor, training flags, forecasting stages, backbone sizes, seeds |

The benchmark trajectory generator and the generated benchmark data are not part of this release. `scripts/generate_demo_data.py` produces generic, non-operational trajectories in the same file format so that the full pipeline can be run end to end.

## Installation

```bash
python -m pip install -r requirements-dev.txt
```

Python 3.10 or later; a CUDA GPU is optional.

## End-to-end demo

```bash
# 1. Generic demo data with sensor-randomized re-observation of the training split
python scripts/generate_demo_data.py --output data/demo --reobserve
# 2. Physical baselines B1-B6 and validation selection of the UKF settings
python tools/physics_baselines.py --data-dir data/demo --out results/demo_physics --horizon 500 --b5-fix
# 3. Drag-UKF anchors and uncertainty features for every split
python tools/physics_baselines.py --data-dir data/demo --out results/demo_physics --horizon 500 --anchors-only --report results/demo_physics/report.json --anchor-model ukf --anchor-unc
# 4. PAST-TSMixer (small width for a CPU run)
python run.py --dataset_dir data/demo --seq_len 100 --label_len 0 --pred_len 500 --model TSMixer --d_model 64 --d_ff 128 --e_layers 2 --train_epochs 5 --patience 5 --batch_size 32 --input_state kinematics --output_mode physics_anchor --anchor_context fit_unc --loss_mode physical_composite --ade_loss_weight 0.5 --fde_loss_weight 0 --select_metric val_rmse_m --residual_gate none --use_gpu 0 --model_id demo
```

The demo checks the software path only; its numbers are unrelated to the paper.

## Paper configuration

`configs/paper_protocol.json` lists the exact training flags. A paper run combines `train_args`, one entry of `stages` (direct learning, physics residual, anchor context, PAST), one entry of `backbones`, and a seed, with `--pred_len 2000`. The main instance is the PAST stage with the TSMixer backbone, trained on the original-plus-re-observed training set with `--scaler_dataset_dir` pointing to the original training set.

## Data format

Each split (`train.pt`, `val.pt`, `test.pt`) is a dictionary:

- `X`: clean positions `(N, L + H, 3)` in meters, observation window followed by the future, held at the impact point after impact;
- `X_observed`: tracked window `(N, L, 3)`;
- `valid_len`: future steps up to and including impact `(N,)`;
- `polar`, `detected`: raw range-azimuth-elevation measurements and detection flags over the warm-up and window;
- `meta`: per-trajectory lists including `domain`, `radar_site_xyz_m`, `t_go_s`, and, for re-observed copies, `sensor_noise_scale`.

Normalization statistics come from the training split only.

## Tests

```bash
pytest
```

## Citation

See [`CITATION.cff`](CITATION.cff); the bibliographic record will be updated on publication.

## License

MIT; see [`LICENSE`](LICENSE).
