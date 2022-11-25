import waveglow
import text
import audio
import torch
import os
from tqdm import tqdm

import utils_fastspeech
from configs.TrainConfig import TrainConfig
from configs.FastSpeechConfig import FastSpeechConfig
from models.FastSpeechModel import FastSpeech
from utils.inference import get_data, synthesis


def main(checkpoint):
    WaveGlow = utils_fastspeech.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    model_config = FastSpeechConfig()
    train_config = TrainConfig()

    model = FastSpeech(model_config)
    model = model.to(train_config.device)

    model.load_state_dict(torch.load(checkpoint, map_location='cuda:0')['model'])
    model = model.eval()

    data_list = get_data()
    for speed in [0.8, 1., 1.3]:
        for i, phn in tqdm(enumerate(data_list)):
            mel, mel_cuda = synthesis(model, phn, speed)

            os.makedirs("results", exist_ok=True)

            audio.tools.inv_mel_spec(
                mel, f"results/s={speed}_{i}.wav"
            )

            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                f"results/s={speed}_{i}_waveglow.wav"
            )
