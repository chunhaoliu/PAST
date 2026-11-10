"""Generate generic, non-operational demo data for software smoke tests.

Trajectories follow the generic constant-drag point-mass model of utils/ballistic_physics.py with arbitrary
launch and drag values; they are not projectile models and are not the paper benchmark. Each trajectory is
observed through the radar and tracking chain of utils/radar.py, so the demo files have the same format as the
benchmark files: clean window + future positions ``X``, tracked window ``X_observed``, ``valid_len`` up to
impact, raw polar measurements and detection flags of the warm-up + window span, and metadata.
``--reobserve`` appends one sensor-randomized re-observation of every training trajectory.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.ballistic_physics import integrate  # noqa: E402
from utils.radar import DT, radar_track_window  # noqa: E402

DOMAINS = {'train': ('nominal',), 'val': ('nominal',), 'test': ('nominal', 'extrapolative')}


def flights(rng, count):
    """Clean generic flights (count, steps, 3) and their ground-impact step."""
    speed, elevation = rng.uniform(200.0, 350.0, count), np.deg2rad(rng.uniform(35.0, 60.0, count))
    azimuth = np.deg2rad(rng.uniform(-5.0, 5.0, count))
    state = np.zeros((count, 6))
    state[:, 3], state[:, 4] = speed * np.cos(elevation) * np.cos(azimuth), speed * np.cos(elevation) * np.sin(azimuth)
    state[:, 5] = speed * np.sin(elevation)
    beta = np.exp(rng.uniform(np.log(2e-5), np.log(2e-4), count))
    path = integrate(state, beta, 8000)
    impact = np.argmax(path[..., 2] <= 0.0, axis=1)
    if np.any(impact == 0):
        raise RuntimeError('demo flight did not reach the ground')
    path[np.arange(8000)[None, :] > impact[:, None]] = np.nan
    return path, impact


def make_split(rng, sensor_rng, count, domain, seq_len, pred_len, warm, min_go, reobserve):
    path, impact = flights(rng, count)
    go = rng.integers(min_go, pred_len + 1, count)  # remaining steps, impact step included
    end = impact - go                                # last observed step
    if np.any(end - seq_len - warm < 0):
        raise RuntimeError('demo flight too short for the requested window')
    span = np.stack([path[i, end[i] - seq_len - warm + 1:end[i] + 1] for i in range(count)])
    future = np.empty((count, pred_len, 3))
    for i in range(count):
        future[i, :go[i]] = path[i, end[i] + 1:end[i] + 1 + go[i]]
        future[i, go[i]:] = future[i, go[i] - 1]
    future[np.arange(count), go - 1, 2] = 0.0
    window = span[:, -seq_len:]
    copies = [radar_track_window(rng, span, seq_len, domain)]
    if reobserve:
        copies.append(radar_track_window(rng, span, seq_len, domain, sensor_rng=sensor_rng))
    parts = []
    for tracked, meta, raw in copies:
        base = {'domain': [domain] * count, 'projectile': ['demo'] * count, 'weapon': ['demo'] * count,
                'wind': ['none'] * count, 't_go_s': (go * DT).astype(np.float32)}
        parts.append({'X': np.concatenate([window, future], axis=1).astype(np.float32), 'X_observed': tracked,
                      'valid_len': go.astype(np.int64), 'polar': raw['polar'], 'detected': raw['detected'],
                      'meta': {**base, **meta}})
    return parts


def merge(parts):
    keys = set.intersection(*(set(p['meta']) for p in parts))
    out = {k: torch.from_numpy(np.concatenate([p[k] for p in parts])) for k in ('X', 'X_observed', 'valid_len', 'polar', 'detected')}
    meta = {}
    for k in sorted(keys):
        values = [p['meta'][k] for p in parts]
        meta[k] = sum(values, []) if isinstance(values[0], list) else np.concatenate(values)
    if any('sensor_noise_scale' in p['meta'] for p in parts):  # original copies use the nominal radar
        meta['sensor_noise_scale'] = np.concatenate([p['meta'].get('sensor_noise_scale', np.ones(len(p['X']), np.float32))
                                                     for p in parts])
    out['meta'] = meta
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/demo')
    parser.add_argument('--seq-len', type=int, default=100)
    parser.add_argument('--pred-len', type=int, default=500)
    parser.add_argument('--warm-steps', type=int, default=100)
    parser.add_argument('--count', type=int, default=64, help='Training trajectories; val/test domains get count/4')
    parser.add_argument('--reobserve', action='store_true', help='Append sensor-randomized training re-observations')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng, sensor_rng = np.random.default_rng(args.seed), np.random.default_rng(args.seed + 1)
    protocol = {'protocol_id': 'software_demo_only', 'default_observation_mode': 'tracked',
                'sample_interval_s': DT, 'observation_steps': args.seq_len, 'prediction_steps': args.pred_len,
                'warm_up_steps': args.warm_steps, 'seed': args.seed, 'reobserve': args.reobserve,
                'purpose': 'Generic non-operational data for interface and smoke testing; not used in the paper.'}
    for name, domains in DOMAINS.items():
        count = args.count if name == 'train' else args.count // 4
        parts = []
        for domain in domains:
            parts += make_split(rng, sensor_rng, count, domain, args.seq_len, args.pred_len, args.warm_steps,
                                args.pred_len // 4, args.reobserve and name == 'train')
        data = merge(parts)
        data['protocol'] = protocol
        torch.save(data, output / f'{name}.pt')
        print(name, tuple(data['X'].shape))
    (output / 'protocol_manifest.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
