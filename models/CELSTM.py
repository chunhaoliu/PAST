"""CE-LSTM sequence regression from Ren et al., Acta Armamentarii (2023).

The one-step teacher-forcing/closed-loop decoding protocol is an interpretation
of the papers' MATLAB sequence-regression description. Recurrent states always
contain raw LSTM outputs. Only the inter-layer outputs are context-enhanced.
With celstm_decoder='direct', the last context-enhanced output feeds the same
FC-sigmoid-FC head once per horizon step (with a learned step embedding), so the
whole horizon is emitted in one pass instead of by closed-loop rollout.
"""

import torch
from torch import nn


class CELSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, alpha_mode='learn', alpha=0.9):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        if alpha_mode == 'learn':
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        elif alpha_mode == 'fixed':
            self.register_buffer('alpha', torch.tensor(float(alpha)))
        else:
            raise ValueError('celstm_alpha_mode must be learn or fixed')

    def forward(self, x, state=None):
        raw, next_state = self.lstm(x, state)
        # Zero initial hidden state supplies h_0 at the start of each sample.
        previous = torch.zeros_like(raw[:, :1]) if state is None else state[0].transpose(0, 1)
        previous = torch.cat([previous, raw[:, :-1]], dim=1)
        mixed = self.alpha * previous + (1.0 - self.alpha) * raw
        return mixed, next_state


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.enc_in != 3 or configs.c_out != 3 or configs.e_layers != 2:
            raise ValueError('CELSTM requires three position channels and e_layers=2')
        if getattr(configs, 'input_state', 'position') not in ('position', 'model_default'):
            raise ValueError('CELSTM requires input_state=position')
        if getattr(configs, 'output_mode', 'absolute') not in ('absolute', 'model_default'):
            raise ValueError('CELSTM requires output_mode=absolute')
        if getattr(configs, 'label_len', 0) != 0:
            raise ValueError('CELSTM requires label_len=0')
        self.pred_len = configs.pred_len
        width = configs.d_model
        mode = getattr(configs, 'celstm_alpha_mode', 'learn')
        alpha = float(getattr(configs, 'alpha', 0.9))
        self.layers = nn.ModuleList([
            CELSTMLayer(3, width, mode, alpha),
            CELSTMLayer(width, width, mode, alpha),
        ])
        self.decoder = getattr(configs, 'celstm_decoder', 'closed_loop')
        head = int(getattr(configs, 'celstm_head_width', 6))
        if self.decoder == 'direct':
            self.horizon_embedding = nn.Parameter(torch.randn(self.pred_len, width) * 0.02)
            self.fc1 = nn.Linear(2 * width, head)
        elif self.decoder == 'closed_loop':
            self.fc1 = nn.Linear(width, head)
        else:
            raise ValueError('celstm_decoder must be closed_loop or direct')
        self.fc2 = nn.Linear(head, 3)

    def encode(self, x, states=None):
        """Return context-enhanced outputs of the second layer and raw states."""
        if states is None:
            states = [None, None]
        next_states = []
        for layer, state in zip(self.layers, states):
            x, next_state = layer(x, state)
            next_states.append(next_state)
        return x, next_states

    def sequence(self, x, states=None):
        """Return one next-position estimate per input step and raw states."""
        x, next_states = self.encode(x, states)
        return self.fc2(torch.sigmoid(self.fc1(x))), next_states

    def teacher_forced(self, observed, future, future_mask):
        """Build s[:-1] -> s[1:] with a mask including the tracked prefix."""
        # Invalid padding cannot influence valid outputs of this causal network.
        future = torch.where(future_mask[..., None].bool(), future, 0.0)
        sequence = torch.cat([observed, future], dim=1)
        target_mask = torch.cat([
            torch.ones_like(observed[:, 1:, 0]), future_mask,
        ], dim=1)
        prediction, _ = self.sequence(sequence[:, :-1])
        return prediction, sequence[:, 1:], target_mask

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """Forecast using only the observed position window."""
        if self.decoder == 'direct':
            context = self.encode(x_enc)[0][:, -1:].expand(-1, self.pred_len, -1)
            horizon = self.horizon_embedding.unsqueeze(0).expand(x_enc.shape[0], -1, -1)
            return self.fc2(torch.sigmoid(self.fc1(torch.cat([context, horizon], dim=-1))))
        prediction, states = self.sequence(x_enc)
        next_position = prediction[:, -1:]
        forecast = [next_position]
        for _ in range(1, self.pred_len):
            next_position, states = self.sequence(next_position, states)
            forecast.append(next_position)
        return torch.cat(forecast, dim=1)
