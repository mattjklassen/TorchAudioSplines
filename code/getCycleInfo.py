# ----- Brief Description -----
# 
# breaking up findCycles.py into separate functions which we can call in material.py
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from argMaxSpec import plotSpecArgMax, getArgMax
from cycleSpline import plotCycleSpline
from matplotlib.backends.backend_pdf import PdfPages
from getCycles import getCycles

# path = "../audio/input.wav"
path = "../audio/A445.wav"
waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)

# specify segment_size (as opposed to number of segments)
segment_size = 2048
num_segments = int(num_frames / segment_size)
# num_segments is actually number of segments of size segment_size
# which can be one less than len(segments)

# split waveform into segments, each of length segment_size (with a remainder of smaller size)
    
def get_segments(waveform, sample_rate, segment_size) # here waveform is whole audio file
    np_waveform = waveform.numpy()
    num_channels, num_frames = np_waveform.shape
    num_segments = int(num_frames / seg_size)
    segments = torch.split(waveform, segment_size, dim=1)
    # we changed from using torch.tensor_split which splits based on number of segments 
    return segments  # may include one extra segment of size < segment_size
    # we can manage this by running loops on segments 0 to num_segments - 1 (rather than len(segments) - 1) 

segments = get_segments(waveform, sample_rate, segment_size)

print("waveform is split into ", len(segments), " segments")
print("size of segment 0: ", len(segments[0]))
print("size of segment ", len(segments)-1, ": ", len(segments[len(segments)-1]))

N = 1024
hop_size = 128

# txt1 and txt2 are based on globals from read of wav file
txt1 = "Audio File read: " + path 
txt1 += "      Length in seconds: " + str(length) 
txt1 += "      Sample Rate:  " + str(sample_rate)
txt2 = "Number of Segments:  " + str(num_segments)
txt2 += "      Segment Size:  " + str(segment_size)
txt2 += "      FFT Size:  " + str(N)
txt2 += "      Hop Size:  " + str(hop_size)

# use this function to process segments of size = segment_size only
def process_segment(segment, current_segment, segment_size, sample_rate, n, N, hop_size, txt1, txt2) :

    RATE = sample_rate
    
    waveform = segment
    np_waveform = waveform.numpy()
    segment_start = segment_size * current_segment
    segment_end = segment_start + segment_size
    # i^th sample value is now data[i]
    num_channels, num_frames = np_waveform.shape
    
    # get the weak f_0 (approx fundamental frequency < max_f0) with getArgMax
    max_f0 = 800
    arg_max = getArgMax(waveform, RATE, N, hop_size, max_f0)  # waveform is a segment

    arg_max_str = f'{arg_max:.2f}'
    samples_per_cycle_guess = RATE / arg_max
    spc_str = f'{samples_per_cycle_guess:.2f}'
    num_hops = int((num_frames - N)/hop_size)+1
    print("arg_max:  ", arg_max_str)
    print("samples per cycle guess:  ", spc_str)
    print("num_frames: ", num_frames)
    print("FFT size N: ", N)
    print("hop_size: ", hop_size)
    print("number of hops: ", num_hops)
    print("(num_hops-1) * hop_size + N = ", (num_hops - 1) * hop_size + N)
    
    # get cycles according to predicted weak f_0
    cycles = getCycles(waveform, RATE, arg_max)
    num_cycles = len(cycles)

    # next: need to do cycleReport for segment, and write bcoeffs for each cycle in segment 
    cycleReport(waveform, sample_rate, txt1, txt2, n, cycles, current_segment, arg_max)


def cycleReport(waveform, sample_rate, txt1, txt2, n, cycles, current_segment, arg_max) :

    # waveform input is one segment, coming from audio file
    # print pdf with title page first, then plot of waveform, and plots of all cycles found:
    pp = PdfPages('../doc/out.pdf')
    
    arg_max_str = f'{arg_max:.2f}'
    samples_per_cycle_guess = sample_rate / arg_max
    num_cycles = len(cycles)

    np_waveform = waveform.numpy() 
    data = torch.squeeze(waveform).numpy()

    firstPage = plt.figure(figsize=(15,8))
    firstPage.clf()

    txt3 = "Data for Segment " + str(current_segment) + ":"
    txt3 += "     Weak f_0:  " + arg_max_str + " Hz"
    txt3 += "     Target Samples per Cycle:  " + str(round(samples_per_cycle_guess,1)).ljust(8, ' ')
    txt3 += "    Number of Cycles:  " + str(num_cycles)
    txt4 = "Cycle Number:" 
    txt4 = str(txt4).ljust(25, ' ')
    txt5 = "Samples per Cycle:"
    txt5 = str(txt5).ljust(25, ' ')
    
    firstPage.text(0.1,0.8,txt1, transform=firstPage.transFigure, size=14)
    firstPage.text(0.1,0.75,txt2, transform=firstPage.transFigure, size=14)
    firstPage.text(0.1,0.65,txt3, transform=firstPage.transFigure, size=14)
    
    lines = int(num_cycles / 10)
    remainder = num_cycles % 10
    # print("lines: ", lines)
    # print("remainder: ", remainder)
    # print("lines * 10 + remainder - num_cycles :  ", lines * 10 + remainder - num_cycles)
    
    def printOneLine(line) :
        start = 0.23
        firstPage.text(0.1, 0.55 - line * 0.1, txt4, transform=firstPage.transFigure, size=14)
        firstPage.text(0.1, 0.5  - line * 0.1, txt5, transform=firstPage.transFigure, size=14)
        for i in range(10) :
            j = i + 10 * line
            a = cycles[j][0]
            b = cycles[j][1]
            samples = int(b-a)
            txt6 = str(j).rjust(8, ' ')
            txt7 = str(samples).rjust(8, ' ')
            firstPage.text(start + i * 0.05, 0.55 - line * 0.1, txt6, transform=firstPage.transFigure, size=14)
            firstPage.text(start + i * 0.05, 0.5  - line * 0.1, txt7, transform=firstPage.transFigure, size=14)
    
    def printLastLine() :
        start = 0.23
        firstPage.text(0.1, 0.55 - lines * 0.1, txt4, transform=firstPage.transFigure, size=14)
        firstPage.text(0.1, 0.5  - lines * 0.1, txt5, transform=firstPage.transFigure, size=14)
        for i in range(remainder) :
            j = i + 10 * lines
            a = cycles[j][0]
            b = cycles[j][1]
            samples = int(b-a)
            txt6 = str(j).rjust(8, ' ')
            txt7 = str(samples).rjust(8, ' ')
            firstPage.text(start + i * 0.05, 0.55 - lines * 0.1, txt6, transform=firstPage.transFigure, size=14)
            firstPage.text(start + i * 0.05, 0.5  - lines * 0.1, txt7, transform=firstPage.transFigure, size=14)
    
    for i in range(lines) :
        printOneLine(i)
    
    if remainder > 0 :
        printLastLine()
    
    plt.savefig(pp, format='pdf')
    
    # title page is finished
    # next do waveform segment graph page
    
    fig = plt.figure(figsize=(15,8))
    times = np.linspace(0, num_frames, num=num_frames)
    samples = np.zeros(num_frames)
    for i in range(num_frames) :
        samples[i] = data[i]
    
    plt.plot(times, samples)
    segment_title = "segment " + str(current_segment) + "  : "
    segment_title += str(segment_size) + " samples: (" + str(int(segment_start)) + " to " + str(int(segment_end)) + ")"
    plt.title(segment_title)
    plt.ylabel("sample float values")
    plt.xlabel("time in samples")
    plt.savefig(pp, format='pdf')
    
    # waveform segment graph page is finished
    # next do one page per cycle graph
    
    for i in range(num_cycles) :
        a = cycles[i][0]
        b = cycles[i][1]
        # a_str = f'{a:.2f}'
        # b_str = f'{b:.2f}'
        # print("cycle ", i, " a: ", a_str, " b: ", b_str)
        n = 30
        fig = plotCycleSpline(waveform, sample_rate, i, a, b, n)
        plt.savefig(pp, format='pdf')
        plt.close()
    
    pp.close()
    
