
import pyaudio
import wave

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def getArgMax(waveform, rate, N, hop_size) :

    np_waveform = waveform.numpy()
    num_channels, num_frames = np_waveform.shape
    
    # PyTorch STFT = Short-Time Fourier Transform
    spec = torch.stft(waveform, n_fft=N, hop_length=hop_size, return_complex=True)
    halfN = int(N/2)
    num_bins = halfN + 1
    num_hops = int((num_frames-N)/hop_size)+1
    
    mag = spec.abs()
    
# only use freq bins < (1/8)*Nyquist, so for sample rate 16000 this is 1000 Hz.
# this is to get a weak f_0 less than 1000 Hz.

    avgFFT = torch.zeros(num_bins)
    for i in range(int((num_bins-1)/8)):
        temp = 0
        for j in range(num_hops):
            temp += mag[0,i,j] / num_bins
        if temp < 0.0001 :
            temp = 0.0001
        avgFFT[i] = temp
    
    arg_max = 0.0
    arg_max = float(torch.argmax(avgFFT).numpy())
    arg_max /= halfN 
    Nyquist = rate / 2
    arg_max *= Nyquist

    return arg_max

    


def plotSpecArgMax(waveform, rate, N, hop_size) :

    np_waveform = waveform.numpy()
    num_channels, num_frames = np_waveform.shape
    
    # PyTorch STFT = Short-Time Fourier Transform
    spec = torch.stft(waveform, n_fft=N, hop_length=hop_size, return_complex=True)
    halfN = int(N/2)
    num_bins = halfN + 1
    num_hops = int((num_frames-N)/hop_size)+1
    
    mag = spec.abs()
    
    avgFFT = torch.zeros(num_bins)
    for i in range(num_bins):
        temp = 0
        for j in range(num_hops):
            temp += mag[0,i,j] / num_bins
        if temp < 0.0001 :
            temp = 0.0001
        avgFFT[i] = temp
    
    arg_max = 0.0
    arg_max = float(torch.argmax(avgFFT).numpy())
    arg_max /= halfN 
    Nyquist = rate / 2
    arg_max *= Nyquist

    xvals = np.log(np.arange(start=1, stop=num_bins+1, step=1))
    yvals = 20*np.log10(avgFFT)
    
    X_Y_Spline = make_interp_spline(xvals, yvals)
    # Returns evenly spaced numbers over a specified interval.
    X_ = np.linspace(xvals.min(), xvals.max(), 1000)
    Y_ = X_Y_Spline(X_)
    plt.figure(figsize=(15,8))
    plt.plot(X_, Y_)
    myTicks = []
    myLabels = []

    myf = Nyquist / (2 ** 8)
    for i in range(1,9) :
        myf *= 2
        mylabel = str(myf) + "Hz"
        myLabels.append(mylabel)
        myx = np.log(myf/Nyquist * halfN)
        mynewf = np.exp(myx)/halfN * Nyquist
        myTicks.append(myx)
    plt.xticks(ticks=myTicks,labels=myLabels)
    plt.title('avg FFT')
    myXlabel = "positive log frequency, arg_max = "
    myXlabel += str(arg_max)
    myXlabel += " Hz"
    plt.xlabel(myXlabel)
    plt.ylabel('magnitude spectrum dB')
    plt.show()
    return arg_max

    

