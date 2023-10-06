# ----- Brief Description -----
# 
# getArgMax(waveform, rate, N, hop_size) computes weak f_0 using STFT 
# where waveform is a torch tensor, returns argMax as weak f_0.
# only use freq bins < (1/scale) * Nyquist = 1000 Hz, so scale = Nyquist / 1000.
# this is to get a weak f_0 less than 1000 Hz.
# plotSpecArgMax does the same as getArgMax but also plots the magnitude spectrum
# which is being used to get argMax.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import pyaudio
import wave

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def getArgMax(waveform, rate, N, hop_size, max_f0) :

    np_waveform = waveform.numpy()
    num_channels, num_frames = np_waveform.shape
    
    # PyTorch STFT = Short-Time Fourier Transform
    hann_window = torch.hann_window(window_length=N, periodic=True)
    spec = torch.stft(waveform, n_fft=N, hop_length=hop_size, window=hann_window, return_complex=True)
    halfN = int(N/2)
    num_bins = halfN + 1
    num_hops = int((num_frames-N)/hop_size)+1
    
    mag = spec.abs()
    
# Resolution Issue:  As an example, if sample_rate = 44100, and N = 1024, we get num_bins = 513
# and this gives frequency resolution in chunks of 44100/513 = 43 Hz.  This is unacceptably
# poor resolution for musical purposes.  For instance, we have a guitar pluck at 445 Hz, which
# computes very close to 445 with Audacity by selecting various segments of different sizes,
# but we are getting around 430 Hz with this argMax, even at the level of each FFT in one segment.
# The bins closest to 445 are in fact: 430 and 473, so indeed 430 is the expected value.

# only use freq bins < (1/scale) * Nyquist = max_f0 Hz, so scale = Nyquist / max_f0.
# This is to get a weak f_0 less than max_f0 Hz.

    # max_f0 = 800
    Nyquist = rate / 2
    scale = Nyquist / max_f0
    scaled_range = int((num_bins)/scale)
    avgFFT = torch.zeros(num_bins)
    arg_max = 0.0

    hops = 0.0
    for j in range(num_hops) :
        magFFT = torch.zeros(num_bins)
        for i in range(scaled_range) :
            magFFT[i] =  mag[0,i,j]
        # find argMax for hop j and convert to Hz
        temp = float(torch.argmax(magFFT).numpy())
        temp /= halfN
        temp *= Nyquist
        print("f0 for hop j = ", j, ": ", temp) 
        arg_max += temp
        hops += 1
    arg_max /= hops
         
    return arg_max

def plotSpecArgMax(waveform, rate, N, hop_size, max_f0) :

    np_waveform = waveform.numpy()
    num_channels, num_frames = np_waveform.shape
    
    # PyTorch STFT = Short-Time Fourier Transform
    hann_window = torch.hann_window(window_length=N, periodic=True)
    spec = torch.stft(waveform, n_fft=N, hop_length=hop_size, window=hann_window, return_complex=True)
    halfN = int(N/2)
    num_bins = halfN + 1
    num_hops = int((num_frames-N)/hop_size)+1
    
    mag = spec.abs()
    
# only use freq bins < (1/scale) * Nyquist = max_f0 Hz, so scale = Nyquist / max_f0.
# This is to get a weak f_0 less than 1000 Hz.

    Nyquist = rate / 2
    scale = Nyquist / max_f0
    scaled_range = int((num_bins)/scale)
    avgFFT = torch.zeros(num_bins)
    arg_max = 0.0

    hops = 0.0
    for j in range(num_hops) :
        magFFT = torch.zeros(num_bins)
        for i in range(scaled_range) :
            magFFT[i] =  mag[0,i,j]
        # find argMax for hop j and convert to Hz
        temp = float(torch.argmax(magFFT).numpy())
        temp /= halfN
        temp *= Nyquist
        print("f0 for hop j = ", j, ": ", temp) 
        arg_max += temp
        hops += 1
    arg_max /= hops
     
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

    

