import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, mel_target, duration_predictor_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())

        return mel_loss, duration_predictor_loss


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, pitch_predicted, energy_predicted, mel_target,
                duration_predictor_target, pitch_predictor_target, energy_predictor_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())

        pitch_predictor_loss = self.l1_loss(pitch_predicted,
                                            pitch_predictor_target.float())

        energy_predictor_loss = self.l1_loss(energy_predicted,
                                             energy_predictor_target.float())

        return mel_loss, duration_predictor_loss, pitch_predictor_loss, energy_predictor_loss
