# In this function we receive audio from mic or read wave file, then predict f_0 and select a cycle.
# Then this cycle is used to generate a waveform at the frequency f_0.

# use:  arg_max = getArgMax(waveform, RATE, N, hop_size) to get f_0
# use:  getCycles(waveform, rate, freq) to get list of cycles
# modify:  getCycles(waveform, rate, freq) and plotCycleSpline(waveform, cycle_num, a, b, n) 
# to return B-spline coefficients

# start with output.wav and get cycles list with bcoeffs

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

print(" first print statement")

from argMaxSpec import plotSpecArgMax, getArgMax
# from cycleSpline import plotCycleSpline
from matplotlib.backends.backend_pdf import PdfPages
from getCyclesFunction import getCycles

# compute bcoeffs for one cycle = [a, b] using waveform data and dim=n
# return a list of bcoeffs of size n


def getBcoeffs(waveform, cycle, n) :

    bcoeffs = []
    a = cycle[0]
    b = cycle[1]
    return bcoeffs


# assume waveform comes from read of this type: 
waveform, sample_rate = torchaudio.load("output.wav")
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
segments = torch.tensor_split(waveform, 16, dim=1)
waveform = segments[1]

np_waveform = waveform.numpy() 
num_channels, num_frames = np_waveform.shape
# i^th sample value is now np_waveform[0,i] (but becomes data below)
# number of samples = num_frames
    
print("waveform.shape:  ", waveform.shape)
squeezed = torch.squeeze(waveform)
print("squeezed.shape:  ", squeezed.shape)
data = torch.squeeze(waveform).numpy()
print("data.shape:  ", data.shape)
print("first 3 data values: ", data[0], data[1], data[2])


