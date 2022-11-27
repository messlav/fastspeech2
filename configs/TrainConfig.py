from dataclasses import dataclass
import torch


@dataclass
class TrainConfig:
    checkpoint_path = "./model_new"
    logger_path = "./logger"
    mel_ground_truth = "./mels"
    alignment_path = "./alignments"
    pitch_path = './pitches2'
    energies_path = './energies2'
    data_path = './data/train.txt'
    
    wandb_project = 'fastspeech'
    
    text_cleaners = ['english_cleaners']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cuda:0'

    batch_size = 16
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 3000
    log_step = 5
    clear_Time = 20

    batch_expand_size = 32
