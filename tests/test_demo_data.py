import sys

import numpy as np
import torch

from scripts import generate_demo_data as demo


def test_demo_format_and_reobservation_preserve_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['demo', '--output', str(tmp_path), '--count', '8', '--pred-len', '300',
                                      '--reobserve'])
    demo.main()
    train = torch.load(tmp_path / 'train.pt', weights_only=False)
    test = torch.load(tmp_path / 'test.pt', weights_only=False)
    assert train['X'].shape == (16, 400, 3) and train['X_observed'].shape == (16, 100, 3)
    assert train['polar'].shape == (16, 200, 3) and train['detected'].shape == (16, 200)
    assert int(train['valid_len'].max()) <= 300 and int(train['valid_len'].min()) >= 75
    # The re-observed copy keeps the truth but sees a different radar.
    np.testing.assert_array_equal(train['X'][:8], train['X'][8:])
    assert not torch.equal(train['X_observed'][:8], train['X_observed'][8:])
    assert np.all(np.asarray(train['meta']['sensor_noise_scale'][:8]) == 1.0)
    assert sorted(set(test['meta']['domain'])) == ['extrapolative', 'nominal']
