# FastSpeech 2
[wandb report and runs](https://wandb.ai/messlav/fastspeech/reports/TTS-Fastspeech2--VmlldzozMDQ5NDE1?accessToken=5of000h5qagb5mbnc0qacbx2m8qmffzi07uwcycjkktj5fjbgbpvz4ar1bva7591)
## Installation guide

```console
git clone https://github.com/messlav/fastspeech2.git
cd fastspeech2
pip install -r requirements.txt
git clone https://github.com/xcmyz/FastSpeech.git
mkdir -p waveglow/pretrained_model/
mv FastSpeech/waveglow/* waveglow/
pip install gdown==4.5.4 --no-cache-dir
```

cringe with gdown version
```python
import gdown

gdown.download('https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx')
gdown.download('https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx')
gdown.download('https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j')
gdown.download('https://drive.google.com/uc?id=1VNjwAi8KcEfYJBcKHq3zP4Lab-DXIKGg',
               'checkpoint_old.pth.tar')
gdown.download('https://drive.google.com/uc?id=1HW8Fyt5Kse0fZxyn6a943LO3ZlulmJmF',
               'checkpoint_final.pth.tar')
```

```console
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
mv train.txt data/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt
tar -xvf mel.tar.gz
```

## Inference

You can pass your own weights and phrase
```console
python3 inference.py --checkpoint 'checkpoint.pth.tar' --extra_phrase 'Hello! Privet! Salam Alleikum! Shalom!'
```

## Training

```console
python3 train.py
```

## Example

[colab](https://colab.research.google.com/drive/1z_2O81Y0kogEWd8QAGccsDHGR_6zAbP3?usp=sharing)

[same notebook in github]()
