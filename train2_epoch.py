from tqdm import tqdm
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler  import OneCycleLR

from configs.TrainConfig import TrainConfig
from configs.MelSpectrogramConfig import MelSpectrogramConfig
from configs.FastSpeechConfig import FastSpeechConfig
from utils.dataload2 import get_data_to_buffer, BufferDataset, collate_fn_tensor
from utils.wandb_writer import WanDBWriter
import utils_fastspeech
from utils.inference import get_data, synthesis2
import waveglow

from models.FastSpeechModel import FastSpeech, FastSpeech2
from metrics.FastSpeechLoss import FastSpeech2Loss


def main():
    # configs
    mel_config = MelSpectrogramConfig()
    model_config = FastSpeechConfig()
    train_config = TrainConfig()
    print(train_config.device)
    # data
    buffer = get_data_to_buffer(train_config)
    dataset = BufferDataset(buffer)
    training_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn_tensor,
        drop_last=True,
        num_workers=0
    )
    # model
    model = FastSpeech2(model_config, mel_config)
    model = model.to(train_config.device)
    # loss and hyperparameters
    fastspeech_loss = FastSpeech2Loss()
    current_step = 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9)

    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
    })
    # WaveGlow
    WaveGlow = utils_fastspeech.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()
    data_list = get_data()
    # train
    os.makedirs(train_config.checkpoint_path, exist_ok=True)
    logger = WanDBWriter(train_config)
    tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)
    for epoch in range(train_config.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                duration = db["duration"].int().to(train_config.device)
                pitch = db["pitch"].int().to(train_config.device)
                energy = db["energy"].int().to(train_config.device)
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]
                while True:
                    current_step += 1
                    tqdm_bar.update(1)

                    logger.set_step(current_step)
                    # Forward
                    mel_output, duration_predictor_output, pitch_predictor_output, energy_predictor_output = model(character,
                                                                  src_pos,
                                                                  mel_pos=mel_pos,
                                                                  mel_max_length=max_mel_len,
                                                                  length_target=duration,
                                                                  pitch_target=pitch,
                                                                  energy_target=energy)

                    # Calc Loss
                    mel_loss, duration_loss, pitch_loss, energy_loss = fastspeech_loss(
                        mel_output,
                        duration_predictor_output,
                        pitch_predictor_output,
                        energy_predictor_output,
                        mel_target,
                        duration,
                        pitch,
                        energy)
                    total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                    # Logger
                    t_l = total_loss.detach().cpu().numpy()
                    m_l = mel_loss.detach().cpu().numpy()
                    d_l = duration_loss.detach().cpu().numpy()
                    p_l = pitch_loss.detach().cpu().numpy()
                    e_l = energy_loss.detach().cpu().numpy()

                    logger.add_scalar("duration_loss", d_l)
                    logger.add_scalar("mel_loss", m_l)
                    logger.add_scalar("total_loss", t_l)
                    logger.add_scalar("pitch_loss", p_l)
                    logger.add_scalar("energy_loss", e_l)

                    # Backward
                    total_loss.backward()

                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.grad_clip_thresh)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    if current_step % train_config.save_step == 0:
                        # torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                        # )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                        # print("save model at step %d ..." % current_step)

                        model = model.eval()
                        for q, phn in tqdm(enumerate(data_list)):
                            for speed in [0.8, 1., 1.2]:
                                mel, mel_cuda = synthesis2(model, phn, speed)
                                audio_inference = waveglow.inference.inference_return_audio(
                                    mel_cuda, WaveGlow
                                )
                                logger.add_audio(f'audio_№_{q}_speed_{speed}', audio_inference, 22050)
                            for pitch in [0.8, 1.2]:
                                mel, mel_cuda = synthesis2(model, phn, alpha2=pitch)
                                audio_inference = waveglow.inference.inference_return_audio(
                                    mel_cuda, WaveGlow
                                )
                                logger.add_audio(f'audio_№_{q}_pitch_{pitch}', audio_inference, 22050)
                            for energy in [0.8, 1.2]:
                                mel, mel_cuda = synthesis2(model, phn, alpha3=energy)
                                audio_inference = waveglow.inference.inference_return_audio(
                                    mel_cuda, WaveGlow
                                )
                                logger.add_audio(f'audio_№_{q}_energy_{energy}', audio_inference, 22050)

                            mel, mel_cuda = synthesis2(model, phn, 0.8, 0.8, 0.8)
                            audio_inference = waveglow.inference.inference_return_audio(
                                mel_cuda, WaveGlow
                            )
                            logger.add_audio(f'audio_№_{q}_all_{0.8}', audio_inference, 22050)

                            mel, mel_cuda = synthesis2(model, phn, 1.2, 1.2, 1.2)
                            audio_inference = waveglow.inference.inference_return_audio(
                                mel_cuda, WaveGlow
                            )
                            logger.add_audio(f'audio_№_{q}_all_{1.2}', audio_inference, 22050)

                        model = model.train()


if __name__ == '__main__':
    main()
