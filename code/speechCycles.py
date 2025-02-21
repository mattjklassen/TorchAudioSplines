# ----- Brief Description -----
# 
# This script is built from findCycles.py but we change a few things:
# In that script we did three main things:
# 1. find weak f_0 with getArgMax() (for waveform about 2048 samples)
# 2. find cycles (in waveform) with getCycles()
# 3. produce output summary and graphs of cycles to pdf
# Here we use 1 second audio at 16k sample rate for speech and do
# one FFT of size 1024 for each chunk of size 1K, and find cycles based on that guess.
# We also do some culling out of segments of audio with too low magnitude to be useful,
# say below 0.01, and simply ignore those parts.  Then we do various computations in order to
# choose a representative cycle to be used for reconstruction.  Here we restrict the f_0 guess to 
# be between 80 and 240 Hz.
# We can still capture f_0 of say 300 Hz by using 150 Hz (an octave lower) so this would be
# cycle length 106 (16K/150) instead of 53 (16K/300).  80 Hz <-> 200 samp,  250 Hz <-> 64 samp.
# or we could go up to 320 Hz <-> 50 samp.
#
# ----- ----- ----- ----- -----

# ----- More Details -----------------------------
#
# From some observations with speech recordings, we can see that selection of a representative cycle
# can be tricky.  Example: one.wav has useable segments (at least one sample above magnitude of 0.01) 
# 7 - 12 out of 16 segments of length 1000 samples.  In segment 9, there are 5 clear cycles with peak
# on the left. With FFT size 1024 we get arg_max for weak f_0 about 78 Hz, which gives target samples
# per cycle about 205.  However, looking at the 5 obvious cycles, indexed 0,3,6,8,11 in the pdf
# one-seg9-report.pdf we see that there is a very consistent cycle length: 188, 187, 188, 187, 187.
# Since the waveform is essentially equal to the concatenation of those cycles, we can say that the
# fundamental frequency is about 16000/187 = 85.5 Hz (and not 78). Of the twelve cycles in that report,
# if we group them according to similarity, there are those 5, and some smaller groups:
# Group 1:  0,3,6,8,11
# Group 2:  5,7,10
# Group 3:  1,9
# Group 4:  2,4
# We should be able distinguish Group 1 by using dot product comparison of B-spline coefficients,
# and number of similar cycles, as well as coverage of the segment by the group.
# 

# ------- Old Comments from findCycles.py --------
# 
# In the function findCycles we find "cycles" in an audio segment given weak f_0 (fundamental frequency).
# The weak f_0 is found by the function getArgMax() and the cycles are then found with getCycles()
# By "cycle" we mean a time interval [a,b] (with a and b time values in float samples)
# where time is measured from 0 in the audio segment, and where b-a has length in samples
# predicted by f_0, so b-a is approximately sample_rate * 1/f_0 (samples/cycle). 
#
# Cycles are computed based on zero crossings between samples, where the audio graph is computed
# using linear interpolation between samples.  Zero crossings are assumed to have the pattern of
# positive slope at t, so crossing of type f(x_i) < 0 and f(x_{i+1}) > 0 and x_i < t < x_{i+1}.
# Cycles of this type are found at each such zero crossing of positive slope, by finding the next
# such zero crossing which is closest to the projected number of cycles per sample. Cycles may also overlap.
#
# We call f_0 "weak" since it can be applied in an audio segment which is not harmonic, meaning
# that it might not be deemed to have a fundamental frequency.  For example, a noisy segment may not
# have a clear f_0 but it can still be measured with STFT to produce weak f_0 which can simply be
# defined as an average arg_max of frequency bins.  Additionally, the values of the audio signal
# over the short interval of one cycle can still be used to represent the signal in the entire segment
# or an interval larger than one segment, using cycle interpolation.  Of course, the accuracy of such
# a representation between cycles is another matter.
#
# description of input parameters to function getCycles(waveform, rate, freq):
# 1. waveform -- is a torch tensor which should come from a line like: (for example)
# waveform, sample_rate = torchaudio.load("output.wav"), where output.wav is 2 sec at 16K sample rate
# shape of waveform is:  torch.Size([1, length]), where length is number of frames
# also np_waveform = waveform.numpy() and num_channels, num_frames = np_waveform.shape
# or could do segments = torch.tensor_split(waveform, 16, dim=1)
# then do waveform = segments[i], then for output.wav get segment size = 2000 samples
# 2. rate -- audio sample rate
# 3. freq -- predicted weak f_0
# it may also be interesting to have a measure of energy in the cycle, which can be computed
# as an average sum of squares of sample values, or spectral values
# output is written to pdf with title page first, then plot of waveform, and plots of all cycles found
#
# ----- ----- ----- ----- -----

import sys
import math
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from argMaxSpec import plotSpecArgMax, getArgMax
from cycleSpline import plotCycleSpline
from matplotlib.backends.backend_pdf import PdfPages
from getCycles import getCycles
from getBcoeffs import export_bcoeffs

# dot product of two vectors of floats:
def dotProd(v, w) :
    n = len(v)
    m = len(w)
    # print("input type:  ", type(v))
    if abs(n-m) > 0 :
        print("inputs do not have same length")
        return(0)
    output = 0.0
    for i in range(n) :
        output += v[i] * w[i]
    return float(output)

# vectorize cycle from waveform data:
def vectorize(cycle, data, n) :
    # n is length of vector, cycle is [a,b] interval with float sample endpoints inside sample data 
    v = torch.zeros(n)
    a = cycle[0]
    b = cycle[1]
    length = b - a  # float length of cycle in samples
    step = length / (n + 1)
    t = a + step
    for i in range(n) :
        # t is some float sample, between integer samples 
        t0 = math.floor(t)  # integer left sample
        t1 = t0 + 1         # integer right sample
        x = t - float(t0)  # fractional part of t in [0,1]
        y0 = data[t0]
        y1 = data[t1]
        m = y1 - y0
        # line between (0,y0) and (1,y1) is now y = y0 + m * x, where x is in [0,1]
        y = y0 + m * x
        v[i] = y
        t = t + step
    return v


# recalling how to work with tensors or numpy arrays here:
# v = torch.zeros(3)
# w = torch.zeros(3)
# for i in range(3) :
#     v[i] = 2 * i
#     w[i] = -i
# 
# x = dotProd(v,w)
# print("dot prod of v and w:  ", x)
# 
# print("\nNow do it with numpy:\n")
# 
# a = v.numpy()
# b = w.numpy()
# x = dotProd(a,b)
# print("dot prod of a and b:  ", x)

# test the function getSpeechCycles with waveform data and write output to pdf
# start with input of about 1 second length

# main part of script

print("Argument List:", str(sys.argv))
args = len(sys.argv)

audio_file = sys.argv[1]
path = "../audio/" + audio_file
print("path: ", path)
index = audio_file.find(".")
param = audio_file[:index]
audio_prefix = audio_file[:index]
print("audio_prefix:", audio_prefix)


path1 = audio_prefix 
pathlib.Path(path1).mkdir(parents=True, exist_ok=True)
path2 = audio_prefix + "/reports" 
pathlib.Path(path2).mkdir(parents=True, exist_ok=True)
path3 = audio_prefix + "/bcoeffs" 
pathlib.Path(path3).mkdir(parents=True, exist_ok=True)

waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)
print("with length ", float(num_frames / sample_rate), " seconds")

# first two args are: [0] speechCycles.py [1] audio_file

bcoeffs_num = 0
seg_num = 0
n = 20
f0_guess = 0.0

if args > 2 :
    print("args > 2")
    n = int(sys.argv[2])
    print("spline dimension: ", n)

if args > 3 :
    print("args > 3")
    seg_num = int(sys.argv[3])
    print("segment selected: ", seg_num)

if args > 4 :
    print("args > 4")
    f0_guess = float(sys.argv[4])
    print("f0 guess: ", f0_guess)

# split waveform into segments, with segment_size about 1/16 second 
segment_size = int(sample_rate / 16.0)
# or use hard-coded segment_size:
# segment_size = 2048
# split waveform into segments
num_segments = int(num_frames / segment_size)  # this will be 1K samples for speech

if args < 3 :
    print("segment size:  ", segment_size)
    print("number of segments:  ", num_segments)
    print("rerun with spline dimension <n> and segment number <seg_num>")
    print("<speechCycles.py> <audiofilename.wav> <n> <seg_num>")
    print("to write report with graphs of cycles to pdf, and bcoeffs files")
    print("for each cycle to directory:  ", audio_prefix)
    print("can also add arg <f0_guess> to end of list to force f_0")
    print("<findCycles.py> <audiofilename.wav> <n> <seg_num> <f0_guess>")
    sys.exit(0)

segments = torch.split(waveform, segment_size, dim=1)
# we changed from using torch.tensor_split which splits based on number of segments 
# we can manage this by running loops on segments 0 to num_segments - 1 (rather than len(segments) - 1) 

print("splitting into ", num_segments, " segments")

# segment loop:
for seg_num in range(num_segments) :
# for seg_index in range(1) :
# print("size of segment ", i, " : ", segments[i].size())

    # below we set hop_size = 1024 = N which is FFT size, so that we are basically calling STFT to do only one FFT
    RATE = sample_rate
    N = 1024
    hop_size = 1024
    energy = 0.0
    
    print("processing segment number ", seg_num)
    
    segment_start = segment_size * seg_num
    print("segment start: ", segment_start)
    segment_end = segment_start + segment_size
    print("segment end: ", segment_end)
    waveform = segments[seg_num]  # waveform is now current segment
    np_waveform = waveform.numpy() 
    data = torch.squeeze(waveform).numpy()  # data is not current segment samples
    # i^th sample value is now data[i]
    num_channels, num_frames = np_waveform.shape

    # check for max value in segment, if < 0.01 then do not process segment 
    skip = 1
    for t in range(segment_size) :
        if abs(data[t]) > 0.01 :
            skip = 0
            continue
    if skip == 1 :
        print("skipping segment ", seg_num)
        continue

    # get the weak f_0 (approx fundamental frequency) with getArgMax
    max_f0 = 300
    arg_max = getArgMax(waveform, RATE, N, hop_size, max_f0)
    # here we shift by octaves to put arg_max Hz value between 80 and 320 
    while arg_max < 80 :
        arg_max *= 2
    while arg_max / 2 > 320 :
        arg_max /= 2
    # use f0_guess if > 0 from command line arg
    if f0_guess > 0 :
        arg_max = f0_guess
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
    
    # get cycles according to predicted f_0 passed in as arg_max
    # each cycle is a pair a,b giving an interval on the time axis is [a,b]
    # a and b are floats, sub-sample, and cycle length is b-a
    # to compute number of samples in one cycle use samples per cycle = int(b) - int(a)
    # note 1: cycles are computed with zero crossings, which are forced to be between samples
    # (if a zero occurs exactly at a sample, then we move it by 0.01 samples offset)
    # note 2: for graphing of cycles, we often want to pass two sample values as input,
    # so in order to graph entire cycle it is best to pass in int(a) as first, and int(b)+1 as last
    # although one should be aware that the graph will extend beyond the actual cycle endpoints
    
    cycles = getCycles(waveform, RATE, arg_max)
    num_cycles = len(cycles)
    new_cycles = []  # for removing any cycles that are too small

    # now go through cycles and remove any that have max size < 0.01 
    for i in range(num_cycles) :
        a = cycles[i][0]
        b = cycles[i][1]
        cycle = [a,b]
        skip = 1
        for t in range(int(a)+1,int(b)+1) :
            if abs(data[t]) > 0.01 :
                skip = 0
        if skip == 0 :
            new_cycles.append(cycle)
    cycles = new_cycles
    num_cycles = len(cycles)

    
    # print pdf with title page first, then plot of waveform, and plots of all cycles found:
    report_name = audio_prefix + "-seg" + str(seg_num) + "-report.pdf"
    report_path = audio_prefix + "/" + "reports/" + report_name
    pp = PdfPages(report_path)
    
    firstPage = plt.figure(figsize=(15,8))
    firstPage.clf()
    txt1 = "Audio File read: " + path 
    txt1 += "      Length in seconds: " + str(length) 
    txt1 += "      Sample Rate:  " + str(sample_rate)
    txt2 = "Number of Segments:  " + str(num_segments)
    txt2 += "      Segment Size:  " + str(segment_size)
    txt2 += "      FFT Size:  " + str(N)
    txt2 += "      Hop Size:  " + str(hop_size)
    txt3 = "Data for Segment " + str(seg_num) + ":"
    txt3 += "     Weak f_0:  " + str(arg_max) + " Hz"
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
            samples = int(b) - int(a)
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
            samples = int(b) - int(a)
            txt6 = str(j).rjust(8, ' ')
            txt7 = str(samples).rjust(8, ' ')
            firstPage.text(start + i * 0.05, 0.55 - lines * 0.1, txt6, transform=firstPage.transFigure, size=14)
            firstPage.text(start + i * 0.05, 0.5  - lines * 0.1, txt7, transform=firstPage.transFigure, size=14)
    
    for i in range(lines) :
        printOneLine(i)
    
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
    segment_title = "segment " + str(seg_num) + "  : "
    segment_title += str(segment_size) + " samples: (" + str(int(segment_start)) + " to " + str(int(segment_end)) + ")"
    plt.title(segment_title)
    plt.ylabel("sample float values")
    plt.xlabel("time in samples")
    plt.savefig(pp, format='pdf')
    
    # waveform segment graph page is finished
    # next do one page per cycle graph
    # now also adding in cycle bcoeffs comparison with dot product
    # so we put all bcoeffs into one tensor, matrix with one cycle's bcoeffs per row 
    cycle_bcoeffs = torch.zeros(num_cycles, n) 
    cycle_vectors = torch.zeros(num_cycles, n) 
    cycle_bcoeff_prods = torch.zeros(num_cycles, num_cycles) 
    cycle_vector_prods = torch.zeros(num_cycles, num_cycles) 
    
    for i in range(num_cycles) :
        a = cycles[i][0]
        b = cycles[i][1]
        cycle = [a,b]
        # a_str = f'{a:.2f}'
        # b_str = f'{b:.2f}'
        # print("cycle ", i, " a: ", a_str, " b: ", b_str)
        # n = 20  # n is now command line parameter
        fig, bcoeffs = plotCycleSpline(waveform, sample_rate, i, a, b, n)
        # print("index i = ", i)
        # print("bcoeffs =")
        # print(bcoeffs.numpy())
        cycle_bcoeffs[i] = bcoeffs
        v = vectorize(cycle, data, n)
        cycle_vectors[i] = v
        # print("cycle_bcoeffs:")
        # print(cycle_bcoeffs)
        file = audio_prefix + "/" + "bcoeffs/" + "bcoeffs-n" + str(n) + "-seg" + str(seg_num) + "-cyc" + str(i) + ".txt"
        export_bcoeffs(file, bcoeffs.numpy())
        plt.savefig(pp, format='pdf')
        plt.close()
    
    pp.close()

    # do dot products for each pair of cycles:
    for i in range(num_cycles) :
        v1 = cycle_bcoeffs[i]
        v2 = cycle_vectors[i]
        for j in range(i, num_cycles) :
            w1 = cycle_bcoeffs[j]
            w2 = cycle_vectors[j]
            x1 = dotProd(v1, w1)
            x2 = dotProd(v2, w2)
            cycle_bcoeff_prods[i, j] = x1
            cycle_vector_prods[i, j] = x2
            print("dot product of cycle bcoeffs ", i, " with cycle bcoeffs ", j, ":  ", x1)
            print("dot product of cycle vectors ", i, " with cycle vectors ", j, ":  ", x2)

    # now bias the cycle shape towards the first third of the cycle being positive
    bias = torch.zeros(num_cycles)
    first_third = int(float(n) / 3)

    for i in range(num_cycles) :
        v = cycle_bcoeffs[i]
        b = 0.0
        for j in range(first_third) :
            b += v[j]
        bias[i] = b

    print("bias toward first third of cycle:")
    print(bias)

    # now we need to group cycles with similarity by having large dot product




    
    # print("resolution of FFT in Hz up to bin 15:")
    # print("bin    Hz")
    # for i in range(15) :
    #    print(i, " :   ", i/512.0 * 22050)
    
    
