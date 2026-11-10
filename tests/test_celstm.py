import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from exp.exp_ballistic import Exp_Ballistic, MaskedMSELoss
from models import build_model
from models.CELSTM import CELSTMLayer, Model


def configs(**updates):
    values = dict(
        model='CELSTM', task_name='ballistic_forecast', enc_in=3, c_out=3,
        e_layers=2, d_model=8, pred_len=5, label_len=0, alpha=0.9,
        celstm_alpha_mode='learn', input_state='position', output_mode='absolute',
        loss_mode='mse', select_metric='val_rmse_m', scale=1, use_gpu=0,
        learning_rate=1e-3, batch_size=128, patience=30, train_epochs=11,
        num_workers=0, train_subset=0,
    )
    values.update(updates)
    return Namespace(**values)


class CELSTMTests(unittest.TestCase):
    def test_shapes_and_registration(self):
        model = build_model(configs())
        observed = torch.randn(2, 4, 3)
        sequence, states = model.sequence(observed)
        self.assertEqual(sequence.shape, (2, 4, 3))
        self.assertEqual(model(observed).shape, (2, 5, 3))
        self.assertEqual(len(states), 2)
        self.assertFalse(any(isinstance(layer, torch.nn.Dropout) for layer in model.modules()))
        self.assertIsInstance(model.fc1, torch.nn.Linear)
        self.assertEqual(model.fc1.out_features, 6)

    def test_raw_hidden_mixing_including_initial_state_and_chunk_boundary(self):
        layer = CELSTMLayer(3, 8)
        x = torch.randn(2, 7, 3)
        raw, raw_state = layer.lstm(x)
        previous = torch.cat([torch.zeros_like(raw[:, :1]), raw[:, :-1]], 1)
        mixed, state = layer(x)
        torch.testing.assert_close(mixed, 0.9 * previous + 0.1 * raw)
        torch.testing.assert_close(state[0], raw_state[0])
        torch.testing.assert_close(state[1], raw_state[1])
        prefix, prefix_state = layer(x[:, :4])
        suffix, _ = layer(x[:, 4:], prefix_state)
        torch.testing.assert_close(mixed, torch.cat([prefix, suffix], 1))

    def test_alpha_modes(self):
        for mode in ('learn', 'fixed'):
            model = Model(configs(celstm_alpha_mode=mode))
            model.sequence(torch.randn(2, 4, 3))[0].square().mean().backward()
            for layer in model.layers:
                self.assertAlmostEqual(layer.alpha.item(), 0.9)
                self.assertEqual(layer.alpha.requires_grad, mode == 'learn')
                self.assertEqual(layer.alpha.grad is not None, mode == 'learn')
                self.assertIn('alpha', layer.state_dict())
        with self.assertRaises(ValueError):
            CELSTMLayer(3, 8, 'invalid')

    def test_teacher_forcing_shift_window_mask_and_padding(self):
        model = Model(configs())
        observed = torch.randn(2, 4, 3)
        future = torch.randn(2, 5, 3)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1.0]])
        pred, target, weights = model.teacher_forced(observed, future, mask)
        self.assertEqual(pred.shape, (2, 8, 3))
        torch.testing.assert_close(target[:, :3], observed[:, 1:])
        torch.testing.assert_close(target[:, 3:5], future[:, :2])
        torch.testing.assert_close(weights, torch.cat([torch.ones(2, 3), mask], 1))
        sequence = torch.cat([observed, torch.where(mask[..., None].bool(), future, 0.0)], 1)
        torch.testing.assert_close(pred, model.sequence(sequence[:, :-1])[0])
        loss = MaskedMSELoss()(pred, target, weights)
        expected = (pred - target).square()[weights.bool()].mean()
        torch.testing.assert_close(loss, expected)
        corrupted = future.clone()
        corrupted[~mask.bool()] = float('nan')
        other_pred, other_target, other_weights = model.teacher_forced(observed, corrupted, mask)
        self.assertTrue(torch.equal(pred, other_pred))
        self.assertTrue(torch.equal(loss, MaskedMSELoss()(other_pred, other_target, other_weights)))

    def test_closed_loop_equals_manual_raw_lstm_steps(self):
        model = Model(configs()).double().eval()
        observed = torch.randn(2, 4, 3, dtype=torch.float64)
        states = [(torch.zeros(2, 8, dtype=torch.float64), torch.zeros(2, 8, dtype=torch.float64))
                  for _ in model.layers]

        def step(x):
            for index, layer in enumerate(model.layers):
                h, c = states[index]
                lstm = layer.lstm
                gates = F.linear(x, lstm.weight_ih_l0, lstm.bias_ih_l0)
                gates = gates + F.linear(h, lstm.weight_hh_l0, lstm.bias_hh_l0)
                i, f, g, o = gates.chunk(4, -1)
                next_c = f.sigmoid() * c + i.sigmoid() * g.tanh()
                next_h = o.sigmoid() * next_c.tanh()
                x = layer.alpha * h + (1 - layer.alpha) * next_h
                states[index] = (next_h, next_c)
            return model.fc2(model.fc1(x).sigmoid())

        with torch.no_grad():
            for x in observed.unbind(1):
                prediction = step(x)
            manual = [prediction]
            for _ in range(1, model.pred_len):
                prediction = step(prediction)
                manual.append(prediction)
            torch.testing.assert_close(model(observed), torch.stack(manual, 1), rtol=1e-10, atol=1e-12)

    def test_no_future_leakage_through_validation_and_test_batches(self):
        exp = Exp_Ballistic(configs())
        observed = torch.randn(2, 4, 3)
        future = torch.randn(2, 5, 3)
        mask = torch.ones(2, 5)
        predictions = []
        handle = exp.model.register_forward_hook(lambda module, inputs, output: predictions.append(output.detach().clone()))
        batch = (observed, future, torch.zeros(2, 4, 1), torch.zeros(2, 5, 1), mask, torch.zeros(2, 1, 3))
        corrupt = (observed, future * 10000 + 10000, batch[2], batch[3], mask, batch[5])
        try:
            exp._vali([batch], MaskedMSELoss())
            exp._vali([corrupt], MaskedMSELoss())
            data = Namespace(scale=False)
            with mock.patch.object(exp, '_get_data', return_value=(data, [batch])), \
                    mock.patch('exp.exp_ballistic.write_evaluation_artifacts'):
                exp.test('first')
            with mock.patch.object(exp, '_get_data', return_value=(data, [corrupt])), \
                    mock.patch('exp.exp_ballistic.write_evaluation_artifacts'):
                exp.test('second')
        finally:
            handle.remove()
        self.assertEqual(len(predictions), 4)
        for prediction in predictions[1:]:
            self.assertTrue(torch.equal(predictions[0], prediction))
        with torch.no_grad():
            self.assertTrue(torch.equal(exp.model(observed, None, future),
                                        exp.model(observed, None, future * float('nan'))))

    def test_optimizer_schedule_training_path_and_best_restore(self):
        exp = Exp_Ballistic(configs())
        optimizer = exp._select_optimizer()
        decay = {id(p) for group in optimizer.param_groups if group['weight_decay'] == 1e-4
                 for p in group['params']}
        expected = {id(p) for name, p in exp.model.named_parameters()
                    if '.lstm.weight_ih_' in name or '.lstm.weight_hh_' in name}
        self.assertEqual(decay, expected)
        self.assertEqual(optimizer.defaults['betas'], (0.9, 0.999))
        self.assertFalse(optimizer.defaults['decoupled_weight_decay'])
        batch = (torch.randn(2, 4, 3), torch.randn(2, 5, 3), torch.zeros(2, 4, 1),
                 torch.zeros(2, 5, 1), torch.ones(2, 5), torch.zeros(2, 1, 3))
        data = Namespace(scaler=Namespace(std=np.ones(3)))
        rates, best_state = [], {}

        def validation(*unused):
            rates.append(optimizer.param_groups[0]['lr'])
            if len(rates) == 2:
                best_state.update({key: value.clone() for key, value in exp.model.state_dict().items()})
                return 0.1, 1.0
            return 0.2, 2.0

        with tempfile.TemporaryDirectory() as path:
            exp.args.checkpoints = path
            with mock.patch.object(exp, '_get_data', return_value=(data, [batch])), \
                    mock.patch.object(exp, '_select_optimizer', return_value=optimizer), \
                    mock.patch.object(exp, '_vali', side_effect=validation), \
                    mock.patch.object(exp, '_predict', side_effect=AssertionError('training rolled out')), \
                    mock.patch('torch.nn.utils.clip_grad_norm_') as clip:
                exp.train('tiny')
            clip.assert_not_called()
        self.assertEqual(rates, [1e-3] * 10 + [5e-4])
        self.assertEqual(exp.args.train_epochs_used, 11)
        self.assertEqual(exp.args.best_epoch, 2)
        for key, value in exp.model.state_dict().items():
            self.assertTrue(torch.equal(value, best_state[key]))

    def test_direct_decoder_head_and_training_path(self):
        self.assertNotIn('horizon_embedding', Model(configs()).state_dict())
        model = build_model(configs(celstm_decoder='direct', celstm_head_width=4))
        observed = torch.randn(2, 4, 3)
        output = model(observed)
        self.assertEqual(output.shape, (2, 5, 3))
        last = model.encode(observed)[0][:, -1]
        for step in range(5):
            z = torch.cat([last, model.horizon_embedding[step].expand(2, -1)], -1)
            torch.testing.assert_close(output[:, step], model.fc2(torch.sigmoid(model.fc1(z))))
        exp = Exp_Ballistic(configs(celstm_decoder='direct', train_epochs=1))
        batch = (observed, torch.randn(2, 5, 3), torch.zeros(2, 4, 1),
                 torch.zeros(2, 5, 1), torch.ones(2, 5), torch.zeros(2, 1, 3))
        data = Namespace(scaler=Namespace(std=np.ones(3)))
        with tempfile.TemporaryDirectory() as path:
            exp.args.checkpoints = path
            with mock.patch.object(exp, '_get_data', return_value=(data, [batch])), \
                    mock.patch.object(exp, '_vali', return_value=(0.1, 1.0)), \
                    mock.patch.object(exp.model, 'teacher_forced', side_effect=AssertionError('teacher forcing')), \
                    mock.patch('torch.nn.utils.clip_grad_norm_') as clip:
                exp.train('tiny')
            clip.assert_not_called()
        self.assertEqual(exp.args.best_epoch, 1)

    def test_reject_adapted_io(self):
        for changes in ({'input_state': 'kinematics'}, {'output_mode': 'displacement'},
                        {'output_mode': 'physics_anchor'}, {'e_layers': 3}, {'label_len': 1}):
            with self.assertRaises(ValueError):
                build_model(configs(**changes))


if __name__ == '__main__':
    unittest.main()
