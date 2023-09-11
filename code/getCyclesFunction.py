# In the function findCycles we find "cycles" in an audio segment given weak f_0 (fundamental frequency).
# By "cycle" we mean a time interval [a,b] (with a and b time values in float samples)
# where time is measured from 0 in the audio segment, and where b-a has length in samples
# predicted by f_0, so b-a is approximately sample_rate * 1/f_0 (samples/cycle). 

# Cycles are computed based on zero crossings between samples, where the audio graph is computed
# using linear interpolation between samples.  Zero crossings are assumed to have the pattern of
# positive slope at t, so crossing of type f(x_i) < 0 and f(x_{i+1}) > 0 and x_i < t < x_{i+1}.
# Cycles of this type are found at each such zero crossing of positive slope, by finding the next
# such zero crossing which is closest to the project number of cycles per sample. Cycles may also overlap.

# We call f_0 "weak" since it can be applied in an audio segment which is not harmonic, meaning
# that it might not be deemed to have a fundamental frequency.  For example, a noisy segment may not
# have a clear f_0 but it can still be measured with STFT to produce weak f_0 which can simply be
# defined as an average arg_max of frequency bins.  Additionally, the values of the audio signal
# over the short interval of one cycle can still be used to represent the signal in the entire segment
# or an interval larger than one segment, using cycle interpolation.

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
# import fpdf

from argMaxSpec import plotSpecArgMax, getArgMax
from cycleSpline import plotCycleSpline
from matplotlib.backends.backend_pdf import PdfPages

# input parameters:

# 1. waveform is a torch tensor which should come from a line like:
# waveform, sample_rate = torchaudio.load("output.wav"), output.wav is 2 sec at 16K sample rate
# shape of waveform is:  torch.Size([1, length]), where length is number of frames
# also np_waveform = waveform.numpy() and num_channels, num_frames = np_waveform.shape
# or could do segments = torch.tensor_split(waveform, 16, dim=1)
# then do waveform = segments[i], then for output.wav get segment size = 2000 samples

# 2. rate = audio sample rate

# 3. freq = predicted weak f_0
# it may also be interesting to have a measure of energy in the cycle, which can be computed
# as an average sum of squares of sample values, or spectral values

# output is written to pdf, one cycle per page

def getCycles(waveform, rate, freq) :

    # need to find time value a between samples t_i and t_{i+1} with y_i < 0 < y_{i+1}
    a = 0.0  # left endpoint of cycle
    b = 0.0  # right endpoint of cycle
    cycle_length = 0.0
    np_waveform = waveform.numpy() 
    num_channels, num_frames = np_waveform.shape
    # freq is cycles/sec, rate is samples/sec 
    # rate/freq is samples/cycle = cycle length in samples
    cycle_length = float(rate) / freq  # cycle length in samples

    # print("shape of np_waveform  ", np_waveform.shape)
    y0 = 0.0
    y1 = 0.0
    zero = 0.0
    end_pts = []
    zeros = []
    cycles = []

    # loop over samples in waveform to find a and b
    for i in range(int(num_frames - 2)) :
        y0 = np_waveform[0,i]
        y1 = np_waveform[0,i+1]
        if (y0 < 0) and (y1 > 0) :
#            print("sample ", i, " : ", y0)
#            print("sample ", i+1, " : ", y1)
            m = y1 - y0  # line is y(t) = y0 + mt = 0 when t = -y0/m
            zero = float(i) - y0 / m
            end_pts.append([y0,y1])
            zeros.append(zero)
    # print("zeros:")
    # print(zeros)

    num_zeros = len(zeros)
    last_zero = zeros[num_zeros-1]
    for i in range(num_zeros-1) :
        exceeded = False
        temp = zeros[i] + cycle_length
        if temp > last_zero :
            # print("temp exceeds last_zero")
            exceeded = True
        j = 0
        while zeros[j] < temp :
            j += 1
            if j > num_zeros - 1 :
                j = num_zeros - 1
                break
        closest = j
        if abs(zeros[j] - temp) > abs(zeros[j-1] - temp) :
            closest = j-1
        if exceeded :
            closest = num_zeros - 1
        if closest == i :
            closest = i + 1
        if closest > num_zeros - 1 :
            closest = num_zeros - 1
        diff = zeros[closest] - zeros[i]
        # each cycle is a list [a, b]
        cycles.append([zeros[i],zeros[closest]])

    return cycles

