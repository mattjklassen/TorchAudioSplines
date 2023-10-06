These python scripts deal with the modeling of audio segments with cubic splines.
A segment is referred to as a cycle if it occurs inside a larger segment which has
approximate fundamental frequency f_0, and the cycle has approximate length 1/f_0.

Suggested tests to run:

> python torchSpline.py

... plots cubic spline on interval [0,1] interpolating 20 points with values in [-1,1]
equal to zero at the ends and random values in between. Edit code to change n.

> python wavspline.py ../audio/input.wav 200 500 20

... computes and plots spline for audio segment from input.wav for samples 200 to 500
with n = 20 interpolation points.

> python getf0.py 

... computes estimate of f0 for audio file ../audio/A445.wav using getCycles() first
then uses average of cycle lengths for refined f0, which may be a new method.

> python findCycles.py 

... finds cycles for audio file ../audio/A445.wav using getCycles() 
then prints pdf report to ../doc/out.pdf.


Summary of python files (in alphabetical order) with brief description of each:


1  argMaxSpec.py

# ----- Brief Description -----
# 
# getArgMax(waveform, rate, N, hop_size) computes weak f_0 using STFT 
# where waveform is a torch tensor, returns argMax as weak f_0.
# only use freq bins < (1/scale) * Nyquist = 1000 Hz, so scale = Nyquist / 1000.
# this is to get a weak f_0 less than 1000 Hz.
# plotSpecArgMax does the same as getArgMax but also plots the magnitude spectrum
# which is being used to get argMax.
#


2  computeBsplineVal.py

# ----- Brief Description -----
# 
# inputs: t is float assumed from 0 to 1, and c is an array of n=k+d B-spline coefficients
# compute value of spline f(t) for input t, with bcoeffs c, and the usual knot sequence t_i:
# 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1  (so t_i goes from i=0 to N, with N=n+d+1=N+4 if d=3)
# where f(t) = sum of c_i B^3_i(t) for i = 0,...,N-d-1=N-4
#
# computeSplineVal(d, k, c, t) computes f(t)
# newBsplineVal(d, k, j, t) computes one B-spline B^d_j(t)
#


3  cycleSpline.py

# ----- Brief Description -----
# 
# Here we convert the script wavspline.py into a function with parameters:
# waveform, end_points a and b, and n (spline dim)
# 
# The function reads in a chunk of audio samples (say 2048 samples) and some float endpoints,
# a and b, with 0 <= a < b <= 2047, and graphs a cubic spline that approximates the audio
# graph over the interval [a,b]. The graph is returned as a plt.figure.  
# The cubic spline interpolates n target values which are data points on the piecewise linear 
# graph which connects sample points. (The function is way too long, need to break it up ...)
#


4  findCycles.py

# ----- Brief Description -----
# 
# In this script we do three main things:
# 1. find weak f_0 with getArgMax() (for waveform about 2048 samples)
# 2. find cycles (in waveform) with getCycles()
# 3. produce output summary and graphs of cycles to pdf
#


5  genCycle.py

# ----- Brief Description -----
# 
# Generate one cycle as waveform, and return waveform.
# inputs: cycle = [a,b], B-spline coefficients vector = bcoeffs.
# assume a and b are time values in samples between integer points, so that the
# spline is computed on interval [a,b] and evaluated at M = floor(b)-floor(a) integer 
# points or samples to produce waveform sample values. The waveform will have
# values indexed 0 to M-1 at given sample_rate. 
#


6  genWavTone.py

# ----- Brief Description -----
# 
# Generate waveform given fundamental frequency f0 and key cycles
# using cycle interpolation, and return waveform.
# inputs: 
# f0 = fundamental frequency
# sample_rate = sample rate
# key_bcoeffs = B-spline coefficients vectors of each key cycle 
# keys = indices of key cycle
# gains = scalar multipliers for each key cycle, for envelope
# assume a and b are time values in samples between integer points, so that the
# spline is computed on interval [a,b] and evaluated at M = floor(b)-floor(a) integer 
# points or samples to produce waveform sample values. The waveform will have
# values indexed 0 to M-1 at given sample_rate. 
#


7  getBcoeffs.py

# ----- Brief Description -----
# 
# get bcoeffs (B-spline coefficients) from a cycle (or segment) [a,b] in a waveform
# inputs: waveform (tensor of audio data), cycle = [a,b], n = dimension of splines
# return: bcoeffs vector 
#


8  getCycles.py

# ----- Brief Description -----
# 
# function:  getCycles()
# In the function getCycles we find "cycles" in an audio segment given weak f_0 (fundamental frequency).
# The weak f_0 is found by the function getArgMax() and the cycles are then found with getCycles()
# By "cycle" we mean a time interval [a,b] (with a and b time values in float samples)
# where time is measured from 0 in the audio segment, and where b-a has length in samples
# predicted by f_0, so b-a is approximately sample_rate * 1/f_0 (samples/cycle). 
# function:  getf0withCycles()
# This function uses the above and then simply averages cycle lengths to get refined f0.
#


9  getf0.py

# ----- Brief Description -----
# 
# In this script we use getf0withCycles() applied to an audio file input.
# (change audio file with path variable below ...)
# Briefly, this constructs an estimate of f0 by first doing STFT and argMax
# then refining this estimate using zero crossings to form cycles with getCycles()
#


10  plotSegmentSpline.py

# ----- Brief Description -----
# 
# Here we convert the script wavspline.py into a function with parameters:
# waveform, end_points a and b (float time values measured in samples), and n (spline dim).
# Output to console gives audio file and segment info,
# output with matplot has audio graph as piecewise linear in grey
# overlayed with spline curve in green and n interpolation points highlighted in red.
#


11  plotSpec.py

# ----- Brief Description -----
# 
# This first draft of script to read in audio file and plot spectrum.
# reads in audio file left.wav and computes magnitude spectrum with stft and plots it.
#


12  rec2Spec.py

# ----- Brief Description -----
# 
# Record two seconds of audio and save as output.wav, then reopen this file 
# and do several spectrograms and predict a sequence of f_0 for each segment.
# Output this info to console, or also to spectrogram graphs with matplot.
#


13  record.py

# ----- Brief Description -----
# 
# Record mono from mic at given sample RATE in chunks of size CHUNK samples.
# Duration is in SECONDS, output is output.wav
#


14  recSpec.py

# ----- Brief Description -----
# 
# this program is superceded by rec2spec.py with options to print pdf of spec or not
# records one second of audio and saves as output.wav and then reopens
# this file and does spectrogram and graph.
# sample rate is 16K
#


15  testBcoeffs.py

# ----- Brief Description -----
# 
# get bcoeffs (B-spline coefficients) from a cycle (or segment) [a,b] in a waveform
# inputs: waveform (tensor of audio data), cycle = [a,b], n = dimension of splines
# return: bcoeffs vector 
#


16  testSegmentSpline.py

# ----- Brief Description -----
# 
# Here we convert the script wavspline.py into a function with parameters:
# waveform, end_points a and b (float time values measured in samples), and n (spline dim).
# Output to console gives audio file and segment info,
# output with matplot has audio graph as piecewise linear in grey
# overlayed with spline curve in green and n interpolation points highlighted in red.
#


17  torchSpline.py

# ----- Brief Description -----
#
# Plot cubic spline f(t) through n points (x,y) with x in [0,1], y in [-1,1]
# with f(0)=0=f(1) and other y-values randomly generated.
#


18  wavplot.py

# ----- Brief Description -----
# 
# This program takes a wav file and begin and end sample numbers as command line input
# and draws a plot of the wav file sample values with matplot as piecewise linear graph.
# The sample values are also printed out as text on the command line.
#


19  wavspline.py

# ----- Brief Description -----
# 
# This program takes the following input on the command line:
# [1] wav file 
# [2] start sample number
# [3] end sample number
# [4] n (number of spline interpolation points = dimension of cubic spline vector space)
# output graph with matplot is the original audio graph as piecewise linear function
# overlayed with spline curve in green and interpolation points highlighted in red.
#


20  writewav.py

# ----- Brief Description -----
# 
# read a wav file at sample_rate (like 16K), do linear interpolation bewteen samples
# and write output at 3 * sample_rate (like 48K), wav file.  
# Assume both files' data is 16-bit, short ints.
#


21  yinapp.py

# ----- Brief Description -----
# 
# import audio file and apply yin to chunks of samples
#


22  yinPyTorch.py

# ----- Brief Description -----
# 
# This script tests torchyin as in the example code, by creating a tensor of the
# piano fundamental frequencies f_0 and corresponding sinusoids with those f_0
# and then running YIN f_0 prediction on those audio samples.  YIN predicts a
# period length tau by using correlation function comparisons for various values
# of tau to detect maximum values (see https://brentspell.com/2022/pytorch-yin/).
# We add to this example some output which shows that the predictions are less
# accurate as frequency increases and especially when tau is small and is at
# the midpoint between two integers. (see comments at the bottom of output)
#
