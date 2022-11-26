from try_preprocessor import Preprocessor


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


if __name__ == "__main__":
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
