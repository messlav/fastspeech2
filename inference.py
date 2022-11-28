import waveglow
import argparse
import os
import torch
from tqdm import tqdm

import text
import utils_fastspeech
from configs.FastSpeechConfig import FastSpeechConfig
from configs.MelSpectrogramConfig import MelSpectrogramConfig
from configs.TrainConfig import TrainConfig
from models.FastSpeechModel import FastSpeech2
from utils.inference import get_data, synthesis2

train_config = TrainConfig()


def main(checkpoint, extra_phrase):
    WaveGlow = utils_fastspeech.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    model_config = FastSpeechConfig()
    train_config = TrainConfig()
    mel_config = MelSpectrogramConfig()

    model = FastSpeech2(model_config, mel_config)
    model = model.to(train_config.device)

    model.load_state_dict(torch.load(checkpoint, map_location='cuda:0')['model'])
    model = model.eval()

    data_list = get_data()
    if extra_phrase != '':
        data_list += [text.text_to_sequence(extra_phrase, train_config.text_cleaners)]

    os.makedirs("results", exist_ok=True)
    for q, phn in tqdm(enumerate(data_list)):
        for speed in [0.8, 1., 1.2]:
            mel, mel_cuda = synthesis2(model, phn, d_control=speed)
            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                f'results/audio_№_{q}_speed_{speed}.wav'
            )
        for pitch in [0.8, 1.2]:
            mel, mel_cuda = synthesis2(model, phn, p_control=pitch)
            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                f'results/audio_№_{q}_pitch_{pitch}.wav'
            )
        for energy in [0.8, 1.2]:
            mel, mel_cuda = synthesis2(model, phn, e_control=energy)
            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                f'results/audio_№_{q}_energy_{energy}.wav'
            )

        mel, mel_cuda = synthesis2(model, phn, 0.8, 0.8, 0.8)
        waveglow.inference.inference(
            mel_cuda, WaveGlow,
            f'results/audio_№_{q}_all_{0.8}.wav'
        )

        mel, mel_cuda = synthesis2(model, phn, 1.2, 1.2, 1.2)
        waveglow.inference.inference(
            mel_cuda, WaveGlow,
            f'results/audio_№_{q}_all_{1.2}.wav'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='checkpoint.pth.tar')
    parser.add_argument("--extra_phrase", type=str, default='')
    args = parser.parse_args()
    main(args.checkpoint, args.extra_phrase)
