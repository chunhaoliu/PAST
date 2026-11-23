import os
import sys
import types
import unittest

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.PASTMixer import (
    Model as PASTMixerModel,
    PASTMixerBlock,
    SimpleTemporalMixing,
)


class EchoTemporal(nn.Module):
    def forward(self, x):
        return x


class PASTMixerBlockResidualTests(unittest.TestCase):
    def test_temporal_stage_does_not_apply_duplicate_residual(self):
        configs = types.SimpleNamespace(
            d_model=8,
            seq_len=5,
            d_ff=16,
            dropout=0.0,
        )
        block = PASTMixerBlock(configs)
        block.temporal = EchoTemporal()
        block.channel = nn.Identity()

        x = torch.randn(2, configs.seq_len, configs.d_model)
        y = block(x)

        self.assertTrue(
            torch.allclose(y, x),
            'Temporal mixing stage should contribute one residual path, not duplicate the input.',
        )


class PASTMixerAblationTests(unittest.TestCase):
    def test_simple_temporal_ablation_keeps_shape(self):
        configs = types.SimpleNamespace(
            d_model=8,
            seq_len=5,
            d_ff=16,
            dropout=0.0,
        )
        block = PASTMixerBlock(configs)
        x = torch.randn(2, configs.seq_len, configs.d_model)
        y = block(x)

        self.assertIsInstance(block.temporal, SimpleTemporalMixing)
        self.assertEqual(y.shape, x.shape)

    def test_pre_norm_adds_temporal_channel_and_output_normalization(self):
        configs = types.SimpleNamespace(
            d_model=8,
            seq_len=5,
            d_ff=16,
            dropout=0.0,
            past_norm_mode='pre',
        )
        block = PASTMixerBlock(configs)
        self.assertIsInstance(block.temporal_norm, nn.LayerNorm)
        self.assertIsInstance(block.channel_norm, nn.LayerNorm)

        model_configs = types.SimpleNamespace(
            task_name='ballistic_forecast',
            seq_len=5,
            pred_len=4,
            enc_in=3,
            c_out=3,
            d_model=8,
            d_ff=16,
            e_layers=1,
            dropout=0.0,
            past_norm_mode='pre',
            past_state_mode='kinematics',
            past_state_lag=2,
            past_output_mode='displacement',
            dt=0.1,
        )
        model = PASTMixerModel(model_configs)
        self.assertIsInstance(model.final_norm, nn.LayerNorm)


if __name__ == '__main__':
    unittest.main()
