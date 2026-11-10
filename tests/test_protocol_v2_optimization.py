"""Physics fit/context contracts and candidate sweep coverage (CPU, no real data)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_provider.data_factory import data_provider
from tools import evaluate_protocol_v2_physics_baselines as physics
from utils.ballistic_physics import DT, _window_residual, drag_window_trajectory, rk4_step


def test_conformal_quantile_and_closed_disk_coverage():
    from tools.conformal_protocol_v2 import conformal_quantile, region_summary, calibrate

    # n=10: finite-sample ranks are 10 (90%) and 9 (80%), without interpolation.
    scores = np.array([7., 2., 10., 1., 5., 9., 3., 6., 8., 4.])
    assert conformal_quantile(scores, 0.1) == 10.
    assert conformal_quantile(scores, 0.2) == 9.
    assert np.isinf(conformal_quantile([1., 2.], 0.1))
    assert region_summary(np.array([0., 1., 2., 3.]), np.array([0., 1., 1., 2.])) == {
        'n': 4, 'covered': 2, 'coverage': 0.5, 'median_radius_m': 1.}
    val = {'error': scores, 'spread': np.full(10, 2.), 'correction': np.full(10, 3.)}
    test = {'error': np.array([9., 10., 11.]), 'spread': np.array([2., 6., 2.]),
            'correction': np.array([3., 9., 3.]),
            'meta': {'projectile': ['held'] * 3, 'domain': ['nominal', 'extrapolative', 'nominal']}}
    result = calibrate(val, test, holdout=True)['variants']
    assert result['V1']['d_m'] == 2.  # Calibration medians only, unaffected by test.
    assert result['V2']['d_m'] == 3.
    for variant in result.values():
        assert variant['levels']['90']['groups']['overall']['coverage'] == 2 / 3
    assert result['V1']['levels']['80']['groups']['extrapolative']['median_radius_m'] == 18.
    assert result['V2']['levels']['80']['groups']['extrapolative']['median_radius_m'] == 18.
    assert result['V0']['levels']['80']['groups']['extrapolative']['coverage'] == 0.
    for invalid in ([], [-1.], [np.nan]):
        with pytest.raises(ValueError):
            conformal_quantile(invalid, 0.1)


def test_window_helper_matches_original_backward_integration():
    params = np.array([[100., 200., 300., 50., 20., 30., 1e-4],
                       [10., 20., 30., 60., -20., 5., 2e-4]])
    state = params[:, :6].copy()
    expected = np.empty((2, 100, 3))
    expected[:, -1] = state[:, :3]
    for step in range(98, -1, -1):
        state = rk4_step(state, params[:, 6], -DT)
        expected[:, step] = state[:, :3]
    assert np.array_equal(drag_window_trajectory(params), expected)
    track = expected + 0.1
    prior = (1e-4, 5e-5, 4.)
    residual = np.concatenate([(expected - track).reshape(2, -1),
                               (4 * (params[:, 6] - prior[0]) / prior[1])[:, None]], axis=1)
    assert np.array_equal(_window_residual(params, track, prior), residual)


@pytest.mark.parametrize('scale', [0, 1])
def test_dataset_context_scaling_filtering_and_train_only_beta(tmp_path, scale):
    rng = np.random.default_rng(10)
    arrays = {}
    for split in ('train', 'val', 'test'):
        clean = rng.normal(size=(3, 2100, 3)).astype(np.float32) * [10, 20, 30] + [100, 200, 300]
        observed = clean[:, :100].astype(np.float32) + rng.normal(size=(3, 100, 3)).astype(np.float32)
        anchor = rng.normal(size=(3, 2000, 3)).astype(np.float32) + [500, 600, 700]
        fit = observed - rng.normal(size=(3, 100, 3)).astype(np.float32)
        beta = np.array([1., 2., 3.]) if split == 'train' else np.array([20., 30., 40.])
        torch.save({'X': torch.from_numpy(clean.astype(np.float32)), 'X_observed': torch.from_numpy(observed),
                    'valid_len': torch.tensor([2000, 1000, 500]),
                    'meta': {'weapon': ['a', 'b', 'a'], 'wind': ['constant'] * 3}}, tmp_path / f'{split}.pt')
        for key, value in (('anchor_drag', anchor), ('anchor_fit_window', fit), ('anchor_beta', beta)):
            np.save(tmp_path / f'{key}_{split}.npy', value)
        np.save(tmp_path / f'anchor_unc_{split}.npy', rng.normal(size=(3, 4)).astype(np.float32))
        arrays[split] = clean, observed, anchor, fit, beta
    args = SimpleNamespace(dataset_dir=str(tmp_path), root_path=str(tmp_path), output_mode='physics_anchor',
                           anchor_context='fit', observation_mode='tracked', seq_len=100, label_len=0,
                           pred_len=2000, scale=scale, batch_size=2, num_workers=0, weapon_filter='b')
    dataset, loader = data_provider(args, 'val')
    clean, obs, anchor, fit, beta = arrays['val']
    train = arrays['train'][0].astype(np.float32)
    std = train.std(axis=(0, 1)) + 1e-8 if scale else np.ones(3)
    train_beta = arrays['train'][4]
    expected = np.concatenate([(obs[1] - fit[1]) / std,
                               (anchor[1, 19::20] - obs[1, 99]) / std,
                               np.full((100, 1), (beta[1] - train_beta.mean()) / train_beta.std())], axis=-1)
    assert len(dataset) == 1
    assert len(dataset[0]) == 6
    np.testing.assert_allclose(dataset[0][5]['context'].numpy(), expected, rtol=1e-6, atol=1e-6)
    assert next(iter(loader))[5]['context'].shape == (1, 100, 7)
    args.anchor_context = 'none'
    legacy, _ = data_provider(args, 'val')
    assert torch.is_tensor(legacy[0][5])
    for i in range(5):
        assert torch.equal(legacy[0][i], dataset[0][i])
    assert torch.equal(legacy[0][5], dataset[0][5]['anchor'])
    args.anchor_context = 'fit_unc'
    unc_set, _ = data_provider(args, 'val')
    train_unc = np.load(tmp_path / 'anchor_unc_train.npy').astype(np.float64)[:, :3]
    val_unc = np.load(tmp_path / 'anchor_unc_val.npy').astype(np.float64)[1, :3]
    context = unc_set[0][5]['context'].numpy()
    np.testing.assert_allclose(context[:, :7], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(context[:, 7:], np.broadcast_to((val_unc - train_unc.mean(0)) / train_unc.std(0), (100, 3)),
                               rtol=1e-5, atol=1e-5)


def test_pastmixer_anchor_film_starts_as_unconditioned_model():
    from models.PASTMixer import Model
    base = dict(task_name='ballistic_forecast', seq_len=20, pred_len=30, enc_in=3, c_out=3, d_model=16, d_ff=32,
                e_layers=2, dropout=0.0, context_channels=10)
    torch.manual_seed(0)
    plain = Model(SimpleNamespace(**base)).eval()
    torch.manual_seed(0)
    film = Model(SimpleNamespace(**base, past_anchor_film=1)).eval()
    film.load_state_dict(plain.state_dict(), strict=False)
    x, ctx = torch.randn(4, 20, 3), torch.randn(4, 20, 10)
    torch.testing.assert_close(film(x, None, None, None, context=ctx), plain(x, None, None, None, context=ctx))
    with pytest.raises(ValueError):
        Model(SimpleNamespace(**{**base, 'context_channels': 0}, past_anchor_film=1))


def test_pastmixer_time_basis_is_orthonormal_and_optional():
    from models.PASTMixer import Model
    base = dict(task_name='ballistic_forecast', seq_len=20, pred_len=30, enc_in=3, c_out=3, d_model=16, d_ff=32,
                e_layers=2, dropout=0.0, context_channels=10, past_temporal_mode='simple')
    torch.manual_seed(0)
    plain = Model(SimpleNamespace(**base)).eval()
    torch.manual_seed(0)
    off = Model(SimpleNamespace(**base, past_time_basis=0)).eval()
    x, ctx = torch.randn(4, 20, 3), torch.randn(4, 20, 10)
    torch.testing.assert_close(off(x, None, None, None, context=ctx), plain(x, None, None, None, context=ctx))
    model = Model(SimpleNamespace(**base, past_time_basis=6, past_anchor_film=1)).eval()
    torch.testing.assert_close(model.time_basis.T @ model.time_basis, torch.eye(6), atol=1e-5, rtol=0)
    assert model(x, None, None, None, context=ctx).shape == (4, 30, 3)
    assert 'time_basis' not in model.state_dict()
    with pytest.raises(ValueError):
        Model(SimpleNamespace(**{**base, 'past_temporal_mode': 'multiscale'}, past_time_basis=6))


def test_anchors_only_reuses_report_without_selection_or_evaluation(tmp_path, monkeypatch):
    report = {'beta_prior': {'mean': 1e-4, 'std': 3e-5}, 'selected': {'B4_noise': 4.}}
    path = tmp_path / 'report.json'
    report_text = json.dumps(report)
    path.write_text(report_text)
    for split in ('train', 'val', 'test'):
        (tmp_path / f'{split}.pt').touch()
    calls = []
    params = np.array([[100., 200., 300., 10., 20., 30., 1e-4]])
    monkeypatch.setattr(physics, 'load', lambda directory, split: {'obs': np.zeros((1, 100, 3))})
    def fit(obs, prior, **_):
        calls.append(prior)
        return params
    monkeypatch.setattr(physics, 'fit_drag_window', fit)
    def forbidden(*args, **kwargs):
        pytest.fail('anchors-only must not select priors/settings or evaluate baselines')
    monkeypatch.setattr(physics, 'beta_prior_from_train', forbidden)
    monkeypatch.setattr(physics, 'evaluate', forbidden)
    monkeypatch.setattr(physics, 'run_b5', forbidden)
    monkeypatch.setattr(sys, 'argv', ['physics', '--anchors-only', '--data-dir', str(tmp_path),
                                     '--report', str(path), '--horizon', '10'])
    physics.main()
    assert calls == [(1e-4, 3e-5, 4.)] * 3
    for split in ('train', 'val', 'test'):
        np.testing.assert_array_equal(np.load(tmp_path / f'anchor_fit_window_{split}.npy'),
                                      drag_window_trajectory(params).astype(np.float32))
        assert np.load(tmp_path / f'anchor_drag_{split}.npy').shape == (1, 10, 3)
        np.testing.assert_array_equal(np.load(tmp_path / f'anchor_beta_{split}.npy'), params[:, 6].astype(np.float32))
    assert path.read_text() == report_text


def test_paired_bootstrap_identical_and_weighted():
    from tools.summarize_protocol_v2 import ade_totals, paired_bootstrap

    true = np.zeros((3, 4, 3))
    pred = true.copy()
    pred[:, :, 0] = np.array([1., 2., 3.])[:, None]
    lengths = np.array([1, 2, 4])
    pred[0, 1:] = 1e6  # Invalid tail never affects ADE or bootstrap.
    totals = ade_totals(pred, true, lengths)
    np.testing.assert_array_equal(totals, [1, 4, 12])
    left = {2021: totals, 2022: totals * 2}
    identical = paired_bootstrap(left, left, lengths, 100)
    assert identical['mean'] == 0
    assert identical['ci_low'] <= 0 <= identical['ci_high']
    baseline = {None: np.zeros(3)}
    result = paired_bootstrap(left, baseline, lengths, 100)
    assert result['mean'] == pytest.approx(1.5 * 17 / 7)
    assert result['ci_low'] > 0
    assert result == paired_bootstrap(left, baseline, lengths, 100)
    reverse = paired_bootstrap(baseline, left, lengths, 100)
    assert reverse['mean'] == -result['mean']
    assert reverse['ci_low'] == -result['ci_high']
    with pytest.raises(ValueError, match='matching seed'):
        paired_bootstrap(left, {2021: totals}, lengths, 100)


def test_paired_impact_bootstrap_crossing_quantiles_and_resampling():
    from tools.summarize_protocol_v2 import paired_impact_bootstrap
    from utils.metrics import evaluate, impact_metrics

    true = np.zeros((4, 4, 3))
    true[:, :, 2] = [4, 2, 0, -2]
    lengths = np.full(4, 3)
    left = {}
    for seed, scale in ((2021, 1), (2022, 2)):
        pred = true.copy()
        pred[:, :, 0] = scale * np.array([1., 4., 8., 20.])[:, None] + 2 * np.arange(4)
        pred[:, :, 2] = [3, 1, -1, -3]
        left[seed] = impact_metrics(pred, true, lengths)['impact_err_m']
        # Crossing is halfway between steps 1 and 2, rather than at true impact.
        np.testing.assert_array_equal(left[seed], scale * np.array([1., 4., 8., 20.]) + 3)
        assert np.median(left[seed]) == evaluate(pred, true, lengths)['overall']['impact_err_median_m']
    right = {2021: np.array([10., 1., 6., 30.]), 2022: np.array([3., 20., 1., 12.])}
    a, b = np.stack(list(left.values())), np.stack(list(right.values()))
    reductions = (np.mean, np.median, lambda x, axis: np.percentile(x, 90, axis=axis))
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(2000):
        indices = rng.integers(4, size=4)
        seeds = rng.integers(2, size=2)
        samples.append([np.mean([reduce(a[s, indices], axis=0) - reduce(b[s, indices], axis=0)
                                 for s in seeds]) for reduce in reductions])
    result = paired_impact_bootstrap(left, right, 2000)
    for j, (key, reduce) in enumerate(zip(('mean', 'median', 'p90'), reductions)):
        assert result[key]['difference'] == pytest.approx((reduce(a, axis=1) - reduce(b, axis=1)).mean())
        np.testing.assert_allclose([result[key]['ci_low'], result[key]['ci_high']],
                                   np.percentile(np.asarray(samples)[:, j], [2.5, 97.5]))
    assert result['median']['difference'] != np.median(a - b, axis=1).mean()
    assert paired_impact_bootstrap(left, left, 20) == {
        key: {'difference': 0., 'ci_low': 0., 'ci_high': 0.} for key in result}
    baseline = {None: right[2021]}
    forward = paired_impact_bootstrap(left, baseline, 100)
    reverse = paired_impact_bootstrap(baseline, left, 100)
    assert forward == paired_impact_bootstrap(left, {s: right[2021] for s in left}, 100)
    for key in result:
        assert reverse[key]['difference'] == -forward[key]['difference']
        assert reverse[key]['ci_low'] == pytest.approx(-forward[key]['ci_high'])
        assert reverse[key]['ci_high'] == pytest.approx(-forward[key]['ci_low'])
    with pytest.raises(ValueError, match='matching seed'):
        paired_impact_bootstrap(left, {2021: right[2021]}, 20)
    with pytest.raises(ValueError, match='align'):
        paired_impact_bootstrap(left, {None: np.zeros(3)}, 20)
    with pytest.raises(ValueError, match='positive'):
        paired_impact_bootstrap(left, right, 0)


def test_pooled_worst_class_bootstrap_recomputes_each_methods_maximum():
    from tools.summarize_protocol_v2 import paired_pool_bootstrap

    lengths = {'a': np.array([1, 2]), 'b': np.array([2, 1])}
    left = {'a': {2021: {'totals': np.array([0., 18.]), 'impact': np.array([0., 9.])},
                  2022: {'totals': np.array([0., 24.]), 'impact': np.array([0., 12.])}},
            'b': {2021: {'totals': np.array([16., 0.]), 'impact': np.array([8., 0.])},
                  2022: {'totals': np.array([20., 0.]), 'impact': np.array([10., 0.])}}}
    right = {'a': {None: {'totals': np.array([0., 3.]), 'impact': np.array([0., 1.5])}},
             'b': {None: {'totals': np.array([21., 0.]), 'impact': np.array([10.5, 0.])}}}
    result = paired_pool_bootstrap(left, right, lengths, 2000)
    # Left's worst class is a; right's is b. Max of differences would give 6.
    assert result['worst_class_ade']['difference'] == pytest.approx(0.)
    assert result['class_mean_ade']['difference'] == pytest.approx(2.5)
    assert result['impact_p90']['difference'] == pytest.approx(2.25)
    rng = np.random.default_rng(0)
    expected = []
    worst_classes = set()
    for _ in range(2000):
        draws = {c: rng.integers(2, size=2) for c in lengths}
        seeds = rng.integers(2, size=2) + 2021
        differences = []
        for seed in seeds:
            a = [left[c][seed]['totals'][draws[c]].sum() / lengths[c][draws[c]].sum() for c in lengths]
            b = [right[c][None]['totals'][draws[c]].sum() / lengths[c][draws[c]].sum() for c in lengths]
            worst_classes.add(int(np.argmax(a)))
            a_impact = np.concatenate([left[c][seed]['impact'][draws[c]] for c in lengths])
            b_impact = np.concatenate([right[c][None]['impact'][draws[c]] for c in lengths])
            differences.append([np.mean(a) - np.mean(b), max(a) - max(b),
                                np.percentile(a_impact, 90) - np.percentile(b_impact, 90)])
        expected.append(np.mean(differences, axis=0))
    assert worst_classes == {0, 1}
    for j, key in enumerate(('class_mean_ade', 'worst_class_ade', 'impact_p90')):
        np.testing.assert_allclose([result[key]['ci_low'], result[key]['ci_high']],
                                   np.percentile(np.asarray(expected)[:, j], [2.5, 97.5]))
    repeated = {c: {s: runs[None] for s in (2021, 2022)} for c, runs in right.items()}
    assert result == paired_pool_bootstrap(left, repeated, lengths, 2000)
    assert all(stat == {'difference': 0., 'ci_low': 0., 'ci_high': 0.}
               for stat in paired_pool_bootstrap(left, left, lengths, 20).values())
    with pytest.raises(ValueError, match='matching seed'):
        paired_pool_bootstrap(left, {c: {2021: runs[None]} for c, runs in right.items()}, lengths, 20)


def test_signed_impact_components_use_existing_crossing_and_terminal_direction():
    from tools.summarize_protocol_v2 import signed_impact_components
    from utils.metrics import impact_metrics

    true = np.array([[[0., 0., 3.], [1., 0., 1.], [1., 0., 0.]],
                     [[0., 0., 3.], [0., 1., 1.], [0., 2., 0.]]])
    pred = np.array([[[2., -2., 3.], [4., -2., 1.], [6., -2., -1.]],
                     [[1., 3., 3.], [1., 4., 2.], [1., 5., 1.]]])
    lengths = np.array([3, 3])
    signed = signed_impact_components(pred, true, lengths)
    np.testing.assert_allclose(signed['along'], [4., 2.])
    np.testing.assert_allclose(signed['cross'], [2., 1.])
    np.testing.assert_allclose(signed['tti'], [-0.005, -0.01])
    metrics = impact_metrics(pred, true, lengths)
    np.testing.assert_allclose(np.hypot(signed['along'], signed['cross']), metrics['impact_err_m'])
    np.testing.assert_allclose(np.abs(signed['tti']), metrics['tti_abs_err_s'])


def test_summary_arrays_prefix_wind_and_bootstrap(tmp_path, monkeypatch, capsys):
    from tools import summarize_protocol_v2 as summary
    from utils.metrics import evaluate, metric

    true = np.zeros((2, 3, 3), dtype=np.float32)
    true[:, :, 2] = [2, 1, 0]
    lengths = np.array([2, 3])
    meta = {'domain': ['nominal', 'extrapolative'], 'projectile': ['a', 'b'], 'wind': ['linear', 'random'], 't_go_s': [5., 20.]}
    torch.save({'meta': meta}, tmp_path / 'test.pt')
    expected = []
    for name, offset in [('main_R5c_PASTMixer_s2021', 1), ('main_R5c_PASTMixer_s2022', 3),
                         ('abl_pos_PASTMixer_s2021', 100)]:
        run_dir = tmp_path / 'results' / f'ballistic_forecast_{name}_PASTMixer_sl100'
        run_dir.mkdir(parents=True)
        pred = true.copy()
        pred[:, :, 0] += offset
        for filename, value in [('pred', pred), ('true', true), ('valid_len', lengths)]:
            np.save(run_dir / f'{filename}.npy', value)
        if name.startswith('main'):
            expected.append(evaluate(pred, true, lengths, meta)['overall']['ade_m'])
    physics_dir = tmp_path / 'physics'
    physics_dir.mkdir()
    for name, value in [('pred_B4_DragLS', true), ('true_test', true), ('valid_len_test', lengths)]:
        np.save(physics_dir / f'{name}.npy', value)
    # A stale report must not influence recomputed baseline metrics.
    (physics_dir / 'report.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['summary', '--out', str(tmp_path), '--data-dir', str(tmp_path),
                                     '--physics', str(physics_dir), '--prefix', 'main_', '--bootstrap', '50',
                                     '--pairs', 'R5c:PASTMixer,B4_DragLS;R5c:PASTMixer,R5c:PASTMixer'])
    summary.main()
    report = (tmp_path / 'summary_main.md').read_text(encoding='utf-8')
    assert 'wind: linear' in report and 'wind: random' in report
    assert 'Lead time: [5,10) s' in report and 'Lead time: [15,20] s' in report
    assert '2.0 ± 1.4' in report and 'abl_pos' not in report
    assert '| R5c:PASTMixer | R5c:PASTMixer | 0.000 | [0.000, 0.000] |' in report
    runs = json.loads((tmp_path / 'summary_main_runs.json').read_text(encoding='utf-8'))
    assert len(runs) == 2
    stats = summary.seed_stats(runs, ('overall',))['ade_m']
    assert stats == pytest.approx((np.mean(expected), np.std(expected, ddof=1)))
    assert metric(true, true) == (0, 0, 0, 0)  # Legacy API survives the merge.
    capsys.readouterr()

    for seed in (2021, 2022):
        run_dir = tmp_path / 'results' / f'ballistic_forecast_best_R5c_TimeMixer_s{seed}_TimeMixer_sl100'
        run_dir.mkdir()
        pred = true.copy()
        pred[:, :, 0] = 5
        for filename, value in [('pred', pred), ('true', true), ('valid_len', lengths)]:
            np.save(run_dir / f'{filename}.npy', value)
    monkeypatch.setattr(sys, 'argv', ['summary', '--out', str(tmp_path), '--data-dir', str(tmp_path),
                                     '--physics', str(physics_dir), '--prefix', 'main_,best_', '--bootstrap', '50',
                                     '--pairs', 'R5c:PASTMixer,B4_DragLS', '--group-by', 'projectile',
                                     '--runner-up', 'R5c:PASTMixer', '--impact-bootstrap',
                                     '--physics-methods', 'B4_DragLS', '--report', str(tmp_path / 'class.md')])
    summary.main()
    grouped = (tmp_path / 'class.md').read_text(encoding='utf-8')
    assert 'projectile: a / nominal (n=1)' in grouped
    assert 'projectile: b / extrapolative (n=1)' in grouped
    assert 'projectile: a / all (n=1)' in grouped
    assert 'wind: linear' not in grouped
    assert '| R5c:PASTMixer | R5c:TimeMixer | -3.000 | [-4.000, -2.000] |' in grouped
    assert '| R5c:PASTMixer | B4_DragLS | median | 2.000 | [1.000, 3.000] |' in grouped
    assert not (tmp_path / 'summary_main_best_runs.json').exists()
    assert (tmp_path / 'summary_main.md').read_text(encoding='utf-8') == report
    capsys.readouterr()


@pytest.mark.parametrize('length', [50, 100, 200])
def test_stress_test_only_uses_original_scaler_and_beta(tmp_path, length):
    original, stress = tmp_path / 'original', tmp_path / 'stress'
    original.mkdir()
    stress.mkdir()
    rng = np.random.default_rng(42)
    train = rng.normal(size=(3, length + 2000, 3)).astype(np.float32) * 10 + 100
    torch.save({'X': torch.from_numpy(train)}, original / 'train.pt')
    np.save(original / 'anchor_beta_train.npy', [1., 2., 3.])
    obs = rng.normal(size=(2, length, 3)).astype(np.float32)
    anchor = rng.normal(size=(2, 2000, 3)).astype(np.float32)
    fit = obs - 2
    beta = np.array([20., 30.])
    torch.save({'X': torch.zeros(2, length + 2000, 3), 'X_observed': torch.from_numpy(obs),
                'valid_len': torch.tensor([500, 2000])}, stress / 'test.pt')
    for key, values in [('anchor_drag', anchor), ('anchor_fit_window', fit), ('anchor_beta', beta)]:
        np.save(stress / f'{key}_test.npy', values)
    args = SimpleNamespace(dataset_dir=str(stress), scaler_dataset_dir='original', root_path=str(tmp_path),
                           output_mode='physics_anchor', anchor_context='fit', observation_mode='tracked',
                           seq_len=length, label_len=0, pred_len=2000, scale=True, batch_size=2, num_workers=0)
    dataset, _ = data_provider(args, 'test')
    np.testing.assert_array_equal(dataset.scaler.mean, train.mean(axis=(0, 1)))
    std = train.std(axis=(0, 1)) + 1e-8
    indices = np.arange(1, length + 1) * 2000 // length - 1
    expected = np.concatenate([(obs - fit) / std, (anchor[:, indices] - obs[:, -1:]) / std,
                               np.broadcast_to(((beta - 2) / np.std([1., 2., 3.]))[:, None, None],
                                               (2, length, 1))], axis=-1)
    np.testing.assert_allclose(dataset.context, expected, rtol=1e-6)
    np.testing.assert_allclose(dataset[0][5]['anchor'], (anchor[0] - dataset.scaler.mean) / std, rtol=1e-6)


def test_lead_time_bins_include_boundaries():
    from utils.metrics import evaluate
    true = np.zeros((6, 3, 3))
    pred = true.copy()
    pred[:, :, 0] = np.arange(1, 7)[:, None]
    result = evaluate(pred, true, np.full(6, 3), {'t_go_s': [5, 9.99, 10, 14.99, 15, 20]})['t_go_s']
    assert [v['ade_m'] for v in result.values()] == [1.5, 3.5, 5.5]


@pytest.mark.parametrize('length', [50, 200])
def test_physics_window_lengths_and_train_prior(length):
    from utils.ballistic_physics import fit_drag_window
    params = np.array([[100., 200., 300., 50., 20., 30., 1e-4]])
    obs = drag_window_trajectory(params, length)
    fitted = fit_drag_window(obs, (1e-4, 3e-5, 4.))
    np.testing.assert_allclose(drag_window_trajectory(fitted, length), obs, atol=0.01)
    split = {'obs': obs, 'X': np.zeros((1, length + 500, 3))}
    assert physics.truth(split, 500).shape == (1, 500, 3)
    # Use a valid clean forecast to exercise the dynamic beta-prior window indices.
    from utils.ballistic_physics import drag_predict
    split['X'] = np.concatenate([obs, drag_predict(params, 500)], axis=1)
    assert physics.beta_prior_from_train(split, n=1)['n'] == 1


@pytest.mark.parametrize('anchors_only', [False, True])
def test_physics_test_only_directory_uses_reference_train_and_val(tmp_path, monkeypatch, anchors_only):
    reference, stress = tmp_path / 'reference', tmp_path / 'stress'
    reference.mkdir()
    stress.mkdir()
    (stress / 'test.pt').touch()
    split = {'obs': np.zeros((2, 50, 3)), 'X': np.zeros((2, 60, 3)),
             'valid_len': np.full(2, 10), 'meta': {}}
    loaded, anchors = [], []
    def load(directory, name):
        loaded.append((Path(directory), name))
        return split
    monkeypatch.setattr(physics, 'load', load)
    monkeypatch.setattr(physics, 'beta_prior_from_train', lambda train, **_: {'mean': 1e-4, 'std': 3e-5})
    monkeypatch.setattr(physics, 'run_b4', lambda data, prior, noise, horizon, *_:
                        (physics.truth(data, horizon), np.full(2, prior['mean'])))
    monkeypatch.setattr(physics, 'save_anchors', lambda directory, name, *args: anchors.append((Path(directory), name)))
    monkeypatch.setattr(physics, 'run_b5', lambda *a, **k: pytest.fail('fast/anchors-only must skip UKF'))
    monkeypatch.setattr(sys, 'argv', ['physics', '--data-dir', str(stress), '--prior-dir', str(reference),
                                     '--out', str(tmp_path / 'physics'), '--horizon', '10', '--fast'] +
                                    (['--anchors-only'] if anchors_only else []))
    physics.main()
    assert loaded == [(reference, 'train'), (reference, 'val'), (stress, 'test')]
    assert anchors == [(stress, 'test')]
    if not anchors_only:
        report = json.loads((tmp_path / 'physics/report.json').read_text())
        assert report['prior_dir'] == str(reference.resolve())
        assert 'B4_DragLS' in report['test'] and 'B5_DragUKF' not in report['test']
