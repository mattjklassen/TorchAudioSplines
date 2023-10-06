# ----- Brief Description -----
# 
# import audio file and apply yin to chunks of samples
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----


import torch
import torchyin
import torchaudio
import math

import wave
import matplotlib.pyplot as plt
import numpy as np
import sys

from playsound import playsound

filename = sys.argv[1]
playsound(filename)

waveform, sample_rate = torchaudio.load(filename)

metadata = torchaudio.info(filename)
print(metadata)

print("sample_rate:  ", sample_rate)

yinput = torch.Tensor(waveform)
print("shape of yinput:  ", yinput.shape)

pitch = torchyin.estimate(
    yinput,
    sample_rate=sample_rate,
    pitch_min=60,
    pitch_max=1000,
)

print("shape of pitch tensor:  ", pitch.shape)
print(pitch)

print("some examples of resolution for yin pitch:")
print("i   44100/i")
for i in range(95,106) :
    print(i, " : ", 44100/i)

