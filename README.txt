These python scripts deal with the modeling of audio segments with cubic splines.
A segment is referred to as a cycle if it occurs inside a larger segment which has
approximate fundamental frequency f_0, and the cycle has approximate length 1/f_0.

Suggested tests to run:

> python torchSpline.py

... plots cubic spline on interval [0,1] interpolating n=10 points with values in [-1,1]
equal to zero at the ends and random values in between. Edit code to change n.

> python wavspline.py ../audio/input.wav 200 500 20

... computes and plots spline for audio segment from input.wav for samples 200 to 500
with n = 20 interpolation points.

> python getf0.py 

... computes estimate of f0 for audio file ../audio/guitarA445.wav getArgMax() as the first
approximation which uses torch.stft and takes the ArgMax bin below a set threshold, 
then uses average of cycle lengths for refined f0, which may be a new method.

> python findCycles.py ../audio/audio_file.wav

... finds cycles for audio_file.wav using getCycles() then prints pdf report to ../doc/out.pdf.


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
# inputs: t is float assumed from 0 to 1, c (or bcoeffs) is an array (or tensor) of n B-spline coefficients,
# k is number of subintervals, d is degree (default 3), knotVals is the knot sequence, with usual default:
# 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1  (so t_i goes from i=0 to N, with N=n+d+1=N+4 if d=3)
# compute value of spline f(t) for input t in interval [0,1]
# where f(t) = sum of c_i B^3_i(t) for i = 0,...,N-d-1=N-4
#
# computeSplineVal(d, k, c, t) computes f(t)
# computeSplineVal2(d, bcoeffs, knotVals, t) computes f(t)
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
# This script is broken up into separate functions in getCycleInfo.py
#


5  genCycle.py

# ----- Brief Description -----
# 
# Generate one cycle as waveform, and return waveform.
# inputs: cycle = [a,b], B-spline coefficients vector (tensor) = bcoeffs.
# assume a and b are time values in samples between integer points, so that the
# spline is computed on interval [a,b] and evaluated at M = floor(b)-floor(a) integer 
# points or samples to produce waveform sample values. The waveform will have
# values indexed 0 to M-1 at given sample_rate. 
#


6  genWavTone.py

# ----- Brief Description -----
# 
# Generate waveform given fundamental frequency f0 and key cycles using cycle interpolation. 
# genWavTone() returns waveform as tensor, insertWavTone writes into larger waveform tensor.
# inputs:  f0 = fundamental frequency, sample_rate = sample rate
# time = waveform duration in seconds,
# key_bcoeffs = B-spline coefficients vectors of each key cycle,
# keys = indices of key cycles
# gains = scalar multipliers for each key cycle, for envelope
#


7  getBcoeffs.py

# ----- Brief Description -----
# 
# get bcoeffs (B-spline coefficients) from a cycle (or segment) [a,b] in a waveform
# inputs: waveform (tensor of audio data), cycle = [a,b], n = dimension of splines
# return: bcoeffs vector 
#


8  getCycleInfo.py

# ----- Brief Description -----
# 
# breaking up findCycles.py into separate functions which we can call in material.py
# get_segments(waveform, sample_rate, segment_size)
# process_segment(segment, index, segment_size, sample_rate, n, N, hop_size, txt1, txt2)
# 
#


9  getCycles.py

# ----- Brief Description -----
# 
# functions:  getCycles() and getf0withCycles()
# In the function getCycles we find "cycles" in an audio segment given weak f_0 (fundamental frequency).
# The weak f_0 is found by the function getArgMax() and the cycles are then found with getCycles()
# By "cycle" we mean a time interval [a,b] (with a and b time values in float samples)
# where time is measured from 0 in the audio segment, and where b-a has length in samples
# predicted by f_0, so b-a is approximately sample_rate * 1/f_0 (samples/cycle). 
# The function getf0withCycles() uses the above and then simply averages cycle lengths to get refined f0.
#


10  getf0.py

# ----- Brief Description -----
# 
# In this script we use getf0withCycles() applied to an audio file input.
# (change audio file with path variable below ...)
# Briefly, this constructs an estimate of f0 by first doing STFT and argMax
# then refining this estimate using zero crossings to form cycles with getCycles()
#


11  getKeyCycles.py

# ----- Brief Description -----
# 
# In this script we have modified findCycles.py to do three main things:
# 1. find weak f_0 with getArgMax() in segments of 2048 samples
# 2. find average of weak f_0 values to use when finding cycles
# 2. find cycles (as endpoints only) in entire audio file input with getCycles()
# 3. write selected bcoeffs of key cycles to files
# (do not produce output summary and graphs of cycles to pdf)
#


12  getKnots.py

# ----- Brief Description -----
# 
# import and export knot sequences to text files knots-[description].txt
# also generate knotVals as standard sequence given n = dim of cubic splines
#


13  getStatVals.py

# ----- Brief Description -----
# 
# get stationary points from bcoeffs on interval [0,1]
#


14  material.py

# ----- Brief Description -----
# 
# create directory material-<name>-dim<n> where name = audio file name without .wav,
# and n = dimension of splines.  Then put melodic segments with transformed versions
# and report into directory.  Report should include plot of splines used also.
# command line args: 
# 1: audio file prefix (name)
# 2: dimension of splines (n)
#


15  melody.py

# ----- Brief Description -----
# 
# create melody based on spline curve
#


16  melody2.py

# ----- Brief Description -----
# 
# create melody based on spline curve, also with varying note durations
#


17  melody3.py

# ----- Brief Description -----
# 
# create melody based on spline curve, also with varying note durations
# but now use the stationary points to determine pitch and note duration.
#


18  melody4.py

# ----- Brief Description -----
# 
# Create melody based on spline curve using spline values y to determine pitch
# and x values to determine time durations.  If notes=0 then we use stationary points
# and if notes>0 we use that many equal divisions of the interval [0,1].
# Durations are scaled so that first note lasts for time0 seconds.
# possible command line: (see below for details)
# python melody4.py bcoeffs0.txt f0=234 scale=3 notes=4 shift=5 time0=0.123 r i
# or to use stationary points and do retrograde inversion (and other defaults):
# python melody4.py bcoeffs0.txt r i
#


19  melody5.py

# ----- Brief Description -----
# 
# melody5.py is derived from melody4.py but now expanded in various ways:
#
# 1. we now use config file mel5config.txt which contains the parameters used
#    to construct the melody or melodic fragment from bcoeffs files etc.
# 2. now use a sequence of key bcoeffs for note timbre, contained in files listed in config
# 3. use one designated cycle for the melodic contour
# 4. allow for regularly spaced notes in melodic countour sampling with "notes=12" etc.
# 5. allow for stationary point melodic contour sampling with "stat=1"
#


20  melody6.py

# ----- Brief Description -----
# 
# melody6.py is derived from melody5.py but now adding polyphony or voicing.
#
# 1. use config file mel6config.txt which contains the parameters used
#    to construct the melody or melodic fragment from bcoeffs files etc.
#
# 2. add polyphony, or voicing:
#    in addition to those configs in mel5config.txt we now control the duration
#    of notes in various ways.  For instance, note durations can all be set to
#    last for twice as long as the designated duration in the melody, so they overlap.
#    The waveform for each note will then be stored in a buffer chosen from a list of 
#    buffers, each representing a voice, or channel, and then these buffers will all 
#    be mixed before writing the final output.  
#


21  melodySplinusoid.py

# ----- Brief Description -----
# 
# create melody based on spline curve approx of sin(2Pi*x) with splinusoid.
#


22  plotBcoeffs.py

# ----- Brief Description -----
#
# Plot cubic spline f(t) with bcoeffs coming from file as arg1 on command line, and
# optional knot sequence from file as arg2, and optional inputs from file as arg3.
# Default knot sequence is 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1 and if inputs are
# given then they are also plotted as points in red. (Inputs are not needed for plot)
#


23  plotSegmentSpline.py

# ----- Brief Description -----
# 
# Here we convert the script wavspline.py into a function with parameters:
# waveform, end_points a and b (float time values measured in samples), and n (spline dim).
# Output to console gives audio file and segment info,
# output with matplot has audio graph as piecewise linear in grey
# overlayed with spline curve in green and n interpolation points highlighted in red.
#


24  plotSpec.py

# ----- Brief Description -----
# 
# This first draft of script to read in audio file and plot spectrum.
# reads in audio file left.wav and computes magnitude spectrum with stft and plots it.
#


25  rec2Spec.py

# ----- Brief Description -----
# 
# Record two seconds of audio and save as output.wav, then reopen this file 
# and do several spectrograms and predict a sequence of f_0 for each segment.
# Output this info to console, or also to spectrogram graphs with matplot.
#


26  record.py

# ----- Brief Description -----
# 
# Record mono from mic at given sample RATE in chunks of size CHUNK samples.
# Duration is in SECONDS, output is output.wav
#


27  recSpec.py

# ----- Brief Description -----
# 
# this program is superceded by rec2spec.py with options to print pdf of spec or not
# records one second of audio and saves as output.wav and then reopens
# this file and does spectrogram and graph.
# sample rate is 16K
#


28  scale.py

# ----- Brief Description -----
# 
# write chromatic scale to wav file
#


29  scale2.py

# ----- Brief Description -----
# 
# write two octave chromatic scale to wav file
#


30  t2Spline.py

# ----- Brief Description -----
#
# Plot cubic spline f(t) through n points (x,y) with x in [0,1], y in [-1,1]
# with f(0)=0=f(1), and f'(0)=0=f'(1) and other y-values randomly generated.
#


31  t3Spline.py

# ----- Brief Description -----
#
# Plot cubic spline f(t) through n points (x,y) with x in [0,1], y in [-1,1]
# with f(0)=0=f(1), and f'(0)=Pi, f'(1)=-Pi, f(1/2)=1, f(1/4)=2^(-1/2)=f(3/4).
# These seven conditions match the function y = sin(pi*x) on [-1,1].
# So n=7, k=4, d=3.
#


32  testBcoeffs.py

# ----- Brief Description -----
# 
# get bcoeffs (B-spline coefficients) from a cycle (or segment) [a,b] in a waveform
# inputs: waveform (tensor of audio data), cycle = [a,b], n = dimension of splines
# return: bcoeffs vector 
#


33  testGenWav.py

# ----- Brief Description -----
# 
# construct one tone of 1 sec long with genWavTone()
#


34  testing.py



35  testSegmentSpline.py

# ----- Brief Description -----
# 
# Here we convert the script wavspline.py into a function with parameters:
# waveform, end_points a and b (float time values measured in samples), and n (spline dim).
# Output to console gives audio file and segment info,
# output with matplot has audio graph as piecewise linear in grey
# overlayed with spline curve in green and n interpolation points highlighted in red.
#


36  testSplinusoid.py

# ----- Brief Description -----
# 
# construct one tone of 1 sec long with genWavTone2()
# uses bcoeffs and knotVals to allow for new knot sequence like splinusoid
#


37  tone5.py

# ----- Brief Description -----
# 
# (based on testGenWav.py which was to construct one tone of 1 sec long with genWavTone() )
# This one is for testing a dulcimer tone using 32 key cycles, with various cases.
# The key cycles are chose from a 3 second long sample called dulcimerA3-f.wav with
# fundamental frequency f_0 = 220 Hz, so approximately 3*220 = 660 cycles, from which we chose 32.
# The bcoeffs files are given below.
#


38  tone6.py

# ----- Brief Description -----
# 
# Continuing with testing a dulcimer tone using 32 key cycles, now we break those up into
# 17 subsequences of 16 consecutive key cycles, and construct a waveform in each case.
# 
# The key cycles are chosen from a 3 second long sample called dulcimerA3-f.wav with
# fundamental frequency f_0 = 220 Hz, so approximately 3*220 = 660 cycles, from which we chose 32.
# The bcoeffs files are given below.
#


39  tone7.py

# ----- Brief Description -----
# 
# Continuing with testing a dulcimer tone using 32 key cycles, now we break those up into
# 17 subsequences of 16 consecutive key cycles, and construct a waveform in each case.
# 
# The key cycles are chosen from a 3 second long sample called dulcimerA3-f.wav with
# fundamental frequency f_0 = 220 Hz, so approximately 3*220 = 660 cycles, from which we chose 32.
# The bcoeffs files are given below.
#


40  torchSpline.py

# ----- Brief Description -----
#
# Plot cubic spline f(t) through n points (x,y) with x in [0,1], y in [-1,1]
# with f(0)=0=f(1) and other y-values randomly generated.
#


41  wavplot.py

# ----- Brief Description -----
# 
# This program takes a wav file and begin and end sample numbers as command line input
# and draws a plot of the wav file sample values with matplot as piecewise linear graph.
# The sample values are also printed out as text on the command line.
#


42  wavspline.py

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


43  writewav.py

# ----- Brief Description -----
# 
# read a wav file at sample_rate (like 16K), do linear interpolation bewteen samples
# and write output at 3 * sample_rate (like 48K), wav file.  
# Assume both files' data is 16-bit, short ints.
#


44  yinapp.py

# ----- Brief Description -----
# 
# import audio file and apply yin to chunks of samples
#


45  yinPyTorch.py

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
