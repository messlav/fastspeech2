import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
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


def smth():
    wavs_dir = '../fastspeech2_large_files/data/LJSpeech-1.1/wavs'
    print("Processing Data ...")
    out = list()
    n_frames = 0
    pitch_scaler = StandardScaler()
    energy_scaler = StandardScaler()
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

    # Compute pitch, energy, duration, and mel-spectrogram
    text_file = '../fastspeech2_large_files/data/train.txt'
    mel_ground_truth = '../fastspeech2_large_files/mels'
    text = process_text(text_file)
    all_wavs = sorted(os.listdir('../fastspeech2_large_files/data/LJSpeech-1.1/wavs/'))
    # os.makedirs('../fastspeech2_large_files/pitches', exist_ok=True)
    # os.makedirs('./fastspeech2_large_files/energies', exist_ok=True)
    for i in tqdm(range(len(text))):
        text_now = text[i]
        wav_now = all_wavs[i]
        wav, _ = librosa.load(f'../fastspeech2_large_files/data/LJSpeech-1.1/wavs/{wav_now}')
        duration = np.load(os.path.join(
            "./alignments", str(i) + ".npy"))

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            sampling_rate,
            frame_period=hop_length / sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)

        pitch = pitch[: sum(duration)]
        # print('pitch', pitch.shape)

        mel_gt_name = os.path.join(
            mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1))
        mel_gt_target = np.load(mel_gt_name)
        # print(mel_gt_target.shape)
        # print(wav.shape)
        audio = torch.clip(torch.FloatTensor(wav).unsqueeze(0), -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        # print('audio', audio.shape)
        # energy = STFT.energy(torch.from_numpy(wav))
        energy = STFT.energy(audio)
        # melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
        energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
        energy = energy[: sum(duration)]

        # print(melspec.shape)
        # print('energy', energy.shape)
        # break
        # print("ljspeech-mel-%05d.npy" % (i+1))
        np.save('../fastspeech2_large_files/pitches/ljspeech-pitch-%05d.npy' % (i + 1), pitch)
        np.save('../fastspeech2_large_files/energies/ljspeech-energy-%05d.npy' % (i + 1), energy)
        # np.save('test3.npy', energy)
        # print('done')
        # break


if __name__ == '__main__':
    smth()
