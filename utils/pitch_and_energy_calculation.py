import os
import librosa
import numpy as np
import pyworld as pw
from tqdm import tqdm
from utils_fastspeech import process_text
import torch

import audio as Audio


config = {
    'path': {
        'raw_path': '../fastspeech2_large_files/data/LJSpeech-1.1/wavs',
        'preprocessed_path': '../fastspeech2_large_files/try_preprocess',
    },
    'preprocessing': {
        'audio': {
            'sampling_rate': 22050
        },
        'stft': {
            'hop_length': 256,
            'filter_length': 1024,
            'win_length': 1024,
        },
        'pitch': {
            'feature': 'phoneme_level',
            'normalization': True
        },
        'energy': {
            'feature': 'phoneme_level',
            "normalization": True
        },
        'mel': {
            'n_mel_channels': 80,
            'mel_fmin': 0,
            'mel_fmax': 8000
        }
    },
}


def main():
    print("Processing Data ...")
    sampling_rate = 22050
    hop_length = 256

    STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    # Compute pitch, energy
    text = process_text('../data/train.txt')
    all_wavs = sorted(os.listdir('../data/LJSpeech-1.1/wavs/'))
    # os.makedirs('../fastspeech2_large_files/pitches', exist_ok=True)
    # os.makedirs('./fastspeech2_large_files/energies', exist_ok=True)
    for i in tqdm(range(len(text))):
        wav_now = all_wavs[i]
        wav, _ = librosa.load(f'../data/LJSpeech-1.1/wavs/{wav_now}')
        duration = np.load(os.path.join(
            "../alignments", str(i) + ".npy"))

        pitch, t = pw.dio(
            wav.astype(np.float64),
            sampling_rate,
            frame_period=hop_length / sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)
        pitch = pitch[: sum(duration)]

        audio = torch.clip(torch.FloatTensor(wav).unsqueeze(0), -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        energy = STFT.energy(audio)

        energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
        energy = energy[: sum(duration)]

        np.save('../pitches/ljspeech-pitch-%05d.npy' % (i + 1), pitch)
        np.save('../energies/ljspeech-energy-%05d.npy' % (i + 1), energy)


if __name__ == '__main__':
    main()