"""Radar measurement and converted-measurement tracking chain.

* the filter starts ``warm_steps`` before the observation window, so the
  network sees a converged track;
* two-point differencing initialization with its consistent covariance;
* converted polar measurements with the full 3x3 Jacobian covariance;
* a white-noise-jerk constant-acceleration (CA) model or a white-noise
  acceleration constant-velocity model with known gravity (CVG);
* the filter is vectorized over the batch.

Only the observation-window positions are given to the predictors.  The raw
polar measurements and detection flags of the whole tracked span are returned
so that model-based baselines can use exactly the same information.
"""

from __future__ import annotations

import numpy as np

DT = 0.01
G = 9.81

NOMINAL_SENSOR = {'severity': 1.0, 'detection_probability': 0.98}
EXTRAPOLATIVE_SENSOR = {'severity': 1.6, 'detection_probability': 0.90}
ANGLE_STD_DEG = 0.04


def sensor_for_domain(domain):
    if domain == 'nominal':
        return dict(NOMINAL_SENSOR)
    if domain == 'extrapolative':
        return dict(EXTRAPOLATIVE_SENSOR)
    raise ValueError(f'Unsupported domain={domain!r}')


def sample_radar_sites(rng, n_samples):
    return np.column_stack(
        [
            rng.uniform(-6000.0, -2500.0, n_samples),
            rng.uniform(-2500.0, 2500.0, n_samples),
            rng.uniform(20.0, 80.0, n_samples),
        ]
    )


def _per_row(value):
    """Scalar sensor parameter, or one value per trajectory broadcast over (N, T)."""
    value = np.asarray(value, dtype=np.float64)
    return value if value.ndim == 0 else value[:, None]


def measure_polar(rng, clean, sites, severity, detection_probability):
    """Noisy range/azimuth/elevation measurements of clean (N, T, 3) positions.

    ``severity`` and ``detection_probability`` are scalars or (N,) arrays.
    """
    severity, detection_probability = _per_row(severity), _per_row(detection_probability)
    relative = clean.astype(np.float64) - sites[:, None, :]
    horizontal = np.linalg.norm(relative[..., :2], axis=-1)
    ranges = np.linalg.norm(relative, axis=-1)
    azimuth = np.arctan2(relative[..., 1], relative[..., 0])
    elevation = np.arctan2(relative[..., 2], horizontal)
    range_std = severity * (3.0 + 2.5e-4 * ranges)
    angle_std = np.deg2rad(ANGLE_STD_DEG * severity)
    polar = np.stack(
        [
            ranges + rng.normal(0.0, 1.0, ranges.shape) * range_std,
            azimuth + rng.normal(0.0, 1.0, azimuth.shape) * angle_std,
            elevation + rng.normal(0.0, 1.0, elevation.shape) * angle_std,
        ],
        axis=-1,
    )
    detected = rng.random(ranges.shape) < detection_probability
    detected[:, :2] = True
    return polar, detected, angle_std if angle_std.ndim == 0 else angle_std[:, 0]


def polar_to_cartesian(polar, sites, severity, angle_std):
    """Converted measurements and their Jacobian covariance, shapes (N,T,3), (N,T,3,3)."""
    severity, angle_std = _per_row(severity), _per_row(angle_std)
    r, az, el = polar[..., 0], polar[..., 1], polar[..., 2]
    ce, se, ca, sa = np.cos(el), np.sin(el), np.cos(az), np.sin(az)
    cart = np.stack([r * ce * ca, r * ce * sa, r * se], axis=-1) + sites[:, None, :]
    jac = np.zeros(polar.shape[:-1] + (3, 3))
    jac[..., 0, 0], jac[..., 0, 1], jac[..., 0, 2] = ce * ca, -r * ce * sa, -r * se * ca
    jac[..., 1, 0], jac[..., 1, 1], jac[..., 1, 2] = ce * sa, r * ce * ca, -r * se * sa
    jac[..., 2, 0], jac[..., 2, 2] = se, r * ce
    range_std = severity * (3.0 + 2.5e-4 * np.maximum(r, 0.0))
    polar_var = np.stack(
        [range_std**2, np.broadcast_to(angle_std**2, r.shape), np.broadcast_to(angle_std**2, r.shape)],
        axis=-1,
    )
    cov = np.einsum('...ij,...j,...kj->...ik', jac, polar_var, jac)
    return cart, cov


def _motion_model(model, q):
    """Per-axis transition, process noise and gravity input for one step."""
    dt = DT
    if model == 'ca':
        f1 = np.array([[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        q1 = q * np.array(
            [
                [dt**5 / 20, dt**4 / 8, dt**3 / 6],
                [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                [dt**3 / 6, dt**2 / 2, dt],
            ]
        )
        u1 = np.zeros(3)
    elif model == 'cvg':
        f1 = np.array([[1.0, dt], [0.0, 1.0]])
        q1 = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        u1 = np.array([0.5 * dt**2, dt])
    else:
        raise ValueError(f'Unsupported tracker model={model!r}')
    order = f1.shape[0]
    # State layout: [x-axis block, y-axis block, z-axis block].
    transition = np.kron(np.eye(3), f1)
    process = np.kron(np.eye(3), q1)
    gravity = np.zeros(3 * order)
    if model == 'cvg':
        gravity[2 * order:] = -G * u1
    observation = np.zeros((3, 3 * order))
    for axis in range(3):
        observation[axis, axis * order] = 1.0
    return transition, process, gravity, observation, order


def kalman_track(cart, cov, detected, model='ca', q=None):
    """Run a converted-measurement Kalman filter over (N, T) measurements.

    Returns filtered positions (N, T, 3) and velocities (N, T, 3).
    """
    if q is None:
        q = DEFAULT_Q[model]
    transition, process, gravity, observation, order = _motion_model(model, q)
    n, steps, _ = cart.shape
    dim = 3 * order

    # Two-point differencing initialization from the first two samples
    # (both forced to be detections by ``measure_polar``).
    state = np.zeros((n, dim))
    covariance = np.zeros((n, dim, dim))
    position_idx = [axis * order for axis in range(3)]
    velocity_idx = [axis * order + 1 for axis in range(3)]
    state[:, position_idx] = cart[:, 1]
    state[:, velocity_idx] = (cart[:, 1] - cart[:, 0]) / DT
    covariance[np.ix_(range(n), position_idx, position_idx)] = cov[:, 1]
    covariance[np.ix_(range(n), velocity_idx, velocity_idx)] = (cov[:, 0] + cov[:, 1]) / DT**2
    covariance[np.ix_(range(n), velocity_idx, position_idx)] = cov[:, 1] / DT
    covariance[np.ix_(range(n), position_idx, velocity_idx)] = cov[:, 1] / DT
    if model == 'ca':
        accel_idx = [axis * order + 2 for axis in range(3)]
        state[:, accel_idx[2]] = -G
        for index in accel_idx:
            covariance[:, index, index] = 50.0**2

    positions = np.empty((n, steps, 3))
    velocities = np.empty((n, steps, 3))
    positions[:, 0] = cart[:, 0]
    velocities[:, 0] = state[:, velocity_idx]
    positions[:, 1] = state[:, position_idx]
    velocities[:, 1] = state[:, velocity_idx]
    identity = np.eye(dim)
    for step in range(2, steps):
        state = state @ transition.T + gravity
        covariance = transition @ covariance @ transition.T + process
        hit = detected[:, step]
        if np.any(hit):
            p = covariance[hit]
            innovation = cart[hit, step] - state[hit] @ observation.T
            s = observation @ p @ observation.T + cov[hit, step]
            gain = p @ observation.T @ np.linalg.inv(s)
            state[hit] = state[hit] + np.einsum('nij,nj->ni', gain, innovation)
            joseph = identity - gain @ observation
            covariance[hit] = (
                joseph @ p @ np.swapaxes(joseph, 1, 2)
                + gain @ cov[hit, step] @ np.swapaxes(gain, 1, 2)
            )
        positions[:, step] = state[:, position_idx]
        velocities[:, step] = state[:, velocity_idx]
    return positions, velocities


# Process-noise intensities selected on separate tuning trajectories; units
# m^2/s^5 (CA) and m^2/s^3 (CVG). With ``sensor_rng``, radar_track_window
# implements sensor-randomized re-observation.
DEFAULT_Q = {'ca': 2.0, 'cvg': 20.0}
DEFAULT_MODEL = 'ca'


def radar_track_window(
    rng,
    clean_span,
    window_steps,
    domain,
    model=DEFAULT_MODEL,
    q=None,
    noise_scale=1.0,
    pd=None,
    revisit=1,
    sensor_rng=None,
):
    """Track clean (N, warm+window, 3) spans and return the final window.

    The first ``span - window_steps`` samples are the warm-up period. With
    ``sensor_rng``, each trajectory gets its own degraded radar (25% nominal;
    otherwise noise x LogU[0.5, 3], Pd ~ U[0.5, 1], revisit from {1, 2, 5, 10}),
    drawn from that separate generator so ``rng`` and the truth are unchanged.
    """
    n, span, _ = clean_span.shape
    if window_steps > span:
        raise ValueError('window_steps exceeds the tracked span')
    sensor = sensor_for_domain(domain)
    sensor['severity'] *= noise_scale
    if pd is not None:
        sensor['detection_probability'] = pd
    randomized = {}
    if sensor_rng is not None:
        nominal = sensor_rng.random(n) < 0.25
        randomized = {
            'sensor_noise_scale': np.where(nominal, 1.0, np.exp(sensor_rng.uniform(np.log(0.5), np.log(3.0), n))),
            'sensor_pd': np.where(nominal, sensor['detection_probability'], sensor_rng.uniform(0.5, 1.0, n)),
            'sensor_revisit': np.where(nominal, 1, sensor_rng.choice([1, 2, 5, 10], n)),
        }
        sensor['severity'] = sensor['severity'] * randomized['sensor_noise_scale']
        sensor['detection_probability'] = randomized['sensor_pd']
    sites = sample_radar_sites(rng, n)
    polar, detected, angle_std = measure_polar(
        rng, clean_span, sites, sensor['severity'], sensor['detection_probability']
    )
    if revisit != 1:
        detected &= np.arange(span)[None, :] % revisit == 0
        # Preserve the two measured samples required by two-point initialization.
        detected[:, :2] = True
    if randomized:
        detected &= np.arange(span)[None, :] % randomized['sensor_revisit'][:, None] == 0
        detected[:, :2] = True
    cart, cov = polar_to_cartesian(polar, sites, sensor['severity'], angle_std)
    positions, velocities = kalman_track(cart, cov, detected, model=model, q=q)

    window = slice(span - window_steps, span)
    clean64 = clean_span.astype(np.float64)
    true_velocity_end = (clean64[:, -1] - clean64[:, -2]) / DT
    metadata = {
        'radar_site_xyz_m': sites.astype(np.float32),
        'detection_rate': detected[:, window].mean(axis=1).astype(np.float32),
        'raw_measurement_rmse_m': np.sqrt(
            np.mean((cart[:, window] - clean64[:, window]) ** 2, axis=(1, 2))
        ).astype(np.float32),
        'track_rmse_m': np.sqrt(
            np.mean((positions[:, window] - clean64[:, window]) ** 2, axis=(1, 2))
        ).astype(np.float32),
        'end_position_error_m': np.linalg.norm(
            positions[:, -1] - clean64[:, -1], axis=1
        ).astype(np.float32),
        'end_velocity_error_mps': np.linalg.norm(
            velocities[:, -1] - true_velocity_end, axis=1
        ).astype(np.float32),
        **{key: value.astype(np.float32) for key, value in randomized.items()},
    }
    raw = {
        'polar': polar.astype(np.float32),
        'detected': detected,
        'tracked_velocity_end': velocities[:, -1].astype(np.float32),
    }
    return positions[:, window].astype(np.float32), metadata, raw
