import torch
import numpy as np

from configs.TrainConfig import TrainConfig
import text

train_config = TrainConfig()


def synthesis(model, phn, alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i + 1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)

    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def synthesis2(model, phn, p_control=1.0, e_control=1.0, d_control=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i + 1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)

    with torch.no_grad():
        mel = model.forward(sequence, src_pos, p_control=p_control, e_control=e_control, d_control=d_control)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data():
    tests = [
        "A defibrillator is a device that gives a high energy electric \
        shock to the heart of someone who is in cardiac arrest",

        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",

        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function \
         defined between probability distributions on a given metric space"
    ]
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)

    return data_list
