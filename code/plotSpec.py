# ----- Brief Description -----
# 
# This first draft of script to read in audio file and plot spectrum.
# reads in audio file left.wav and computes magnitude spectrum with stft and plots it.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# ----- ----- ----- ----- -----

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

import os
import shutil
import requests

waveform, sample_rate = torchaudio.load("../audio/left.wav")
print("sample rate of waveform: ")
print(sample_rate)
print("shape of waveform: ")
print(waveform.shape)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape

N = 1024
print("FFT size is: ", N)
hop_size = 256
print("hop size is: ", hop_size)
spec = torch.stft(waveform, n_fft=N, hop_length=hop_size, return_complex=True)
print("shape of spectrogram from torch.stft(waveform, 1024)")
print(spec.shape)
num_bins = int(N/2) + 1
print("num_bins = int(N/2)+1: ", num_bins)
num_hops = int(num_frames/hop_size)+1
print("num_hops = :", num_hops)

print("spec[0].dtype:")
print(spec[0].dtype)
print("spec[0].size():")
print(spec[0].size())
print("spec[0,0].size():")
print(spec[0,0].size())
print("spec[0,0]:")
print(spec[0,0])
print("spec[0,1]:")
print(spec[0,1])
print("spec[0,1,2]:")
print(spec[0,1,2])

# sample rate of waveform: 
# 16000
# shape of waveform: 
# torch.Size([1, 6987])
# shape of spectrogram from torch.stft(waveform, 1024)
# torch.Size([1, 513, 28])
# 6987/256. = 27.29, and hop size (pytorch default) is N/4 = 256, N = 1024
# Also N//2+1 = 513, the number of "unique" FFT bins
# spec[0] is shape spec[0].size():
# torch.Size([513, 28]) so is a 513 x 28 matrix, with 513 rows, 28 cols
# so each column is one FFT and there are 28 hops.
# We can construct a real magnitude spectrum by averaging values over the 28 FFT's

mag = spec.abs()
print("mag[0,0]:")
print(mag[0,0])
print("mag[0,1]:")
print(mag[0,1])

avgFFT = torch.zeros(num_bins)
for i in range(num_bins):
    temp = 0
    for j in range(num_hops):
        temp += mag[0,i,j] / num_bins
    if temp < 0.001 :
        temp = 0.001
    avgFFT[i] = temp

print("avgFFT:")
print(avgFFT)
print("argmax of FFT bins: ", torch.argmax(avgFFT))

# xvals = np.arange(start=0.0, stop=1.001, step=.001)
xvals = np.log(np.arange(start=1, stop=num_bins+1, step=1))
# print(xvals)
# print("size of xvals:  ", xvals.size)
yvals = 20*np.log10(avgFFT)
# print(yvals)
# print("size of yvals:  ", yvals.size)

X_Y_Spline = make_interp_spline(xvals, yvals)
# Returns evenly spaced numbers
# over a specified interval.
X_ = np.linspace(xvals.min(), xvals.max(), 500)
Y_ = X_Y_Spline(X_)
plt.figure(figsize=(15,8))
plt.plot(X_, Y_)

# plt.plot(xvals, yvals)
plt.title('avg FFT')
plt.xlabel('positive log frequency bins')
plt.ylabel('magnitude spectrum dB')
plt.show()

# This plot of avgFFT reaches a max at bin number 11 out of 513 = Nyquist which is 8000 Hz, 
# (11/513)*8000 = 171.5 or pretty close to what Audacity gives as 167 Hz. 
# avgFFT: tensor([0.0219, 0.0223, 0.0275, 0.0378, 0.0635, 0.2336, 0.1715, 0.1139, 0.1416, 0.2172, 0.5418, 0.9566 ...
# There is also a smaller peak at bin 5, or (5/513)*8000 = 78 Hz, pretty close to the 81 Hz in Audacity.

# dB is 20*log10(A2/A1) for amplitude ratio A2/A1.  For float sample values between -1 and 1, dB = 0 corresponds
# to float 1, so all values below 1 will be negative dB.  
    
