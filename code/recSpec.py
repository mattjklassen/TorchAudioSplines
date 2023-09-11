# this program records one second of audio and saves as output.wav and then reopens
# this file and does spectrogram and graph.
# sample rate is 16K

import pyaudio
import wave

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

import os
import shutil
import requests


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
# RATE = 44100
RATE = 16000

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
		channels=CHANNELS,
		rate=RATE,
		input=True,
		frames_per_buffer=CHUNK)

print("start recording ...")

frames = []
seconds = 1
i = 0
# for RATE 16000 and CHUNK 1024 we get RATE/CHUNK = 15.625, so int() = 15
# so below we would read 0 to 15, or 16 chunks, so 16*1024 = 16384 samples
# or just over one second
for i in range(0, int(RATE / CHUNK * seconds)+1):
    data = stream.read(CHUNK)
    frames.append(data)
print("number of chunks: ", i)
    
print("recording stopped")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open("output.wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# end recording and writing wav file from mic
# begin PyTorch reading of wav file and spectrogram

waveform, sample_rate = torchaudio.load("output.wav")
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

# PyTorch STFT = Short-Time Fourier Transform
spec = torch.stft(waveform, n_fft=N, hop_length=hop_size, return_complex=True)
print("shape of spectrogram from torch.stft(waveform, 1024)")
print(spec.shape)
num_bins = int(N/2) + 1
halfN = num_bins - 1
print("num_bins = int(N/2)+1: ", num_bins)
num_hops = int(num_frames/hop_size)+1
print("num_hops = :", num_hops)

# sample rate of waveform: 16000
# example wav file with 6987 samples:
# shape of waveform: torch.Size([1, 6987])
# shape of spectrogram from torch.stft(waveform, 1024):  torch.Size([1, 513, 28])
# 6987/256. = 27.29, and hop size (pytorch default) is N/4 = 256, N = 1024
# Also N//2+1 = 513, the number of "unique" FFT bins (which actually means number of "essential" bins
#   which can be used to reconstruct all N bins as complex, since -X_k = conj(X_k), k=0,...,N/2 for N even.)
# spec[0] is shape spec[0].size(): torch.Size([513, 28]) so is a 513 x 28 matrix, with 513 rows, 28 cols
# so each column is one FFT and there are 28 hops.
# We can construct a real magnitude spectrum by averaging values over the 28 FFT's

mag = spec.abs()

avgFFT = torch.zeros(num_bins)
for i in range(num_bins):
    temp = 0
    for j in range(num_hops):
        temp += mag[0,i,j] / num_bins
    if temp < 0.001 :
        temp = 0.001
    avgFFT[i] = temp

# print("avgFFT:")
# print(avgFFT)
print("argmax of FFT bins: ", torch.argmax(avgFFT))
arg_max = 0.0
arg_max = float(torch.argmax(avgFFT).numpy())
print("arg_max as numpy", arg_max)
# arg_max = np.exp(arg_max)
# print("exp():", arg_max)
arg_max /= halfN 
Nyquist = RATE / 2
arg_max *= Nyquist
print("arg_max in Hz", arg_max)

# xvals = np.arange(start=0.0, stop=1.001, step=.001)
xvals = np.log(np.arange(start=1, stop=num_bins+1, step=1))
# print(xvals)
# print("size of xvals:  ", xvals.size)
yvals = 20*np.log10(avgFFT)
# print(yvals)
# print("size of yvals:  ", yvals.size)

X_Y_Spline = make_interp_spline(xvals, yvals)
# Returns evenly spaced numbers over a specified interval.
X_ = np.linspace(xvals.min(), xvals.max(), 1000)
Y_ = X_Y_Spline(X_)
plt.figure(figsize=(15,8))
plt.plot(X_, Y_)
myTicks = []
myLabels = []
# myf = 31.25
myf = Nyquist / (2 ** 8)
for i in range(1,9) :
    myf *= 2
    mylabel = str(myf) + "Hz"
    myLabels.append(mylabel)
    myx = np.log(myf/Nyquist * halfN)
    mynewf = np.exp(myx)/halfN * Nyquist
    myTicks.append(myx)
#    print(myf, myx, mynewf)
plt.xticks(ticks=myTicks,labels=myLabels)
# plt.plot(xvals, yvals)
plt.title('avg FFT')
myXlabel = "positive log frequency, arg_max = "
myXlabel += str(arg_max)
myXlabel += " Hz"
plt.xlabel(myXlabel)
plt.ylabel('magnitude spectrum dB')
plt.show()

# example plot of avgFFT reaches a max at bin number 11 out of 512 = Nyquist which is 8000 Hz, 
# (11/512)*8000 = 171.875 or pretty close to what Audacity gives as 167 Hz. 
# avgFFT: tensor([0.0219, 0.0223, 0.0275, 0.0378, 0.0635, 0.2336, 0.1715, 0.1139, 0.1416, 0.2172, 0.5418, 0.9566 ...
# There is also a smaller peak at bin 5, or (5/513)*8000 = 78 Hz, pretty close to the 81 Hz in Audacity.

# dB is 20*log10(A2/A1) for amplitude ratio A2/A1.  For float sample values between -1 and 1, dB = 0 corresponds
# to float 1, so all values below 1 will be negative dB.  
    



