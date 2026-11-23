"""PNMixer: pre-layer-normalized temporal-channel mixer backbone (PAST-Mixer in the original submission).

References: TSMixer (KDD'23), TimeMixer (ICLR'24).
"""
import torch
import torch.nn as nn

from models.kinematic_features import build_state_features


class SimpleTemporalMixing(nn.Module):
    """TSMixer-style temporal MLP."""
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.temporal(x)


class ChannelMixing(nn.Module):
    """Channel mixing on embedded features: d_model -> d_ff -> d_model."""
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 2
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, L, d_model)
        return x + self.mlp(x)


class PASTMixerBlock(nn.Module):
    """Temporal mixing followed by channel mixing, each with pre-normalization."""
    def __init__(self, configs):
        super().__init__()
        self.temporal = SimpleTemporalMixing(configs.d_model, configs.seq_len, configs.dropout)
        self.channel = ChannelMixing(configs.d_model, configs.d_ff, configs.dropout)
        self.use_pre_norm = str(getattr(configs, 'past_norm_mode', 'none')).lower() == 'pre'
        self.temporal_norm = nn.LayerNorm(configs.d_model) if self.use_pre_norm else nn.Identity()
        self.channel_norm = nn.LayerNorm(configs.d_model) if self.use_pre_norm else nn.Identity()

    def forward(self, x):
        # x: (B, L, d_model); the temporal stage mixes along L.
        temporal_input = self.temporal_norm(x)
        temporal_output = self.temporal(temporal_input.transpose(1, 2)).transpose(1, 2)
        x = x + (temporal_output - temporal_input)

        channel_input = self.channel_norm(x)
        channel_output = self.channel(channel_input)
        return x + (channel_output - channel_input)


class Model(nn.Module):
    """Embed the observed state, apply the mixer blocks, and project to the prediction horizon."""
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.n_layers = configs.e_layers
        self.dropout = configs.dropout
        self.dt = float(getattr(configs, 'dt', 0.01))
        self.past_state_mode = str(getattr(configs, 'past_state_mode', 'position')).lower()
        self.past_state_lag = int(getattr(configs, 'past_state_lag', 5))
        self.past_norm_mode = str(getattr(configs, 'past_norm_mode', 'none')).lower()
        self.past_output_mode = str(getattr(configs, 'past_output_mode', 'absolute')).lower()
        if self.past_state_mode not in {'position', 'kinematics'}:
            raise ValueError(f'Unsupported past_state_mode={self.past_state_mode}')
        if self.past_norm_mode not in {'none', 'pre'}:
            raise ValueError(f'Unsupported past_norm_mode={self.past_norm_mode}')
        if self.past_output_mode not in {'absolute', 'displacement'}:
            raise ValueError(f'Unsupported past_output_mode={self.past_output_mode}')
        self.state_dim = 3 if self.past_state_mode == 'position' else 9
        self.context_channels = int(getattr(configs, 'context_channels', 0))
        self.state_dim += self.context_channels

        self.kinematic_state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
        )
        self.embedding = self.kinematic_state_encoder  # Saved checkpoints store both names.
        self.blocks = nn.ModuleList([PASTMixerBlock(configs) for _ in range(configs.e_layers)])
        self.final_norm = nn.LayerNorm(configs.d_model) if self.past_norm_mode == 'pre' else nn.Identity()
        self.residual_proj = nn.Linear(configs.d_model, configs.enc_in)
        self.projection = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.pred_len),
            nn.Dropout(configs.dropout),
        )

    def forecast(self, x_enc, context=None):
        """Map the observed trajectory (B, seq_len, enc_in) to the forecast (B, pred_len, enc_in)."""
        state = build_state_features(x_enc, mode=self.past_state_mode, lag=self.past_state_lag, dt=self.dt)
        if self.context_channels:
            if context is None or context.shape != (*x_enc.shape[:2], self.context_channels):
                raise ValueError('Missing or incorrectly shaped PASTMixer context')
            state = torch.cat([state, context], dim=-1)
        x = self.kinematic_state_encoder(state)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        output = self.projection(x.transpose(1, 2)).transpose(1, 2)  # (B, pred_len, d_model)
        output = self.residual_proj(output)
        if self.past_output_mode == 'displacement':
            output = output + x_enc[:, -1:, :]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, context=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast', 'ballistic_forecast'):
            return self.forecast(x_enc, context=context)[:, -self.pred_len:, :]
        raise ValueError(f'Task {self.task_name} not supported')
