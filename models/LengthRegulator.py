import torch
import torch.nn.functional as F
from torch import distributions
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from configs.FastSpeechConfig import FastSpeechConfig


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config: FastSpeechConfig):
        super(DurationPredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout2

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class PitchPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config: FastSpeechConfig):
        super(PitchPredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout2

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class EnergyPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config: FastSpeechConfig):
        super(EnergyPredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout2

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config, train_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor(model_config)
        self.train_config = train_config

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = (duration_predictor_output * alpha + 0.5).int()
            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack([torch.tensor([i+1 for i in range(output.size(1))])]).long().to(self.train_config.device)
            return output, mel_pos


class AllRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config, train_config):
        super(AllRegulator, self).__init__()
        self.length_regulator = LengthRegulator(model_config, train_config)
        self.pitch_predictor = PitchPredictor(model_config)
        self.energy_predictor = EnergyPredictor(model_config)
        self.duration_predictor = DurationPredictor(model_config)
        n_bins = model_config.encoder_dim
        pitch_min = -2.917079304729967
        pitch_max = 11.391254536985784
        energy_min = -1.431044578552246
        energy_max = 8.184337615966797
        self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        self.energy_bins = nn.Parameter(
            torch.linspace(energy_min, energy_max, n_bins - 1),
            requires_grad=False,
        )

        self.pitch_embedding = nn.Embedding(
            model_config.encoder_dim, model_config.decoder_dim
        )
        self.energy_embedding = nn.Embedding(
            model_config.encoder_dim, model_config.decoder_dim
        )

    def LR(self, x, duration_predictor_output, mel_max_length=None):
#         print('duration_predictor_output', duration_predictor_output.shape)
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def get_pitch_embedding(self, x, target, control):
        prediction = self.pitch_predictor(x)
        if target is not None:
#             print('target', target.shape)
#             print('bucket', torch.bucketize(target, self.pitch_bins).shape)
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
#             print('embed', embedding.shape)
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, control):
        prediction = self.energy_predictor(x)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(self, x, mel_max_length=None, pitch_target=None, energy_target=None, duration_target=None,
                p_control=1.0, e_control=1.0, d_control=1.0,):
#         print('start all regulator, x', x.shape)
#         log_duration_prediction = self.duration_predictor(x)
#         print('start pitch embed')
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, p_control
        )
#         print('pitch pred and embed', pitch_prediction.shape, pitch_embedding.shape)
        x = x + pitch_embedding

        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, e_control
        )
        x = x + energy_embedding

#         if duration_target is not None:
#             x, mel_len = self.length_regulator(x, duration_target, mel_max_length)
#             duration_rounded = duration_target
#         else:
#             duration_rounded = torch.clamp(
#                 (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
#                 min=0,
#             )
#             x, mel_len = self.length_regulator(x, duration_rounded, mel_max_length)
        if mel_max_length is not None:
            x, duration_predict_output = self.length_regulator(x, d_control, duration_target, mel_max_length)
            return x, pitch_prediction, energy_prediction, duration_predict_output
        else:
            x, mel_pos = self.length_regulator(x, d_control)
            return x, mel_pos
