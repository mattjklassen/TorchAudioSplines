# ----- Brief Description -----
# 
# Here we convert the script wavspline.py into a function with parameters:
# waveform, end_points a and b (float time values measured in samples), and n (spline dim).
# Output to console gives audio file and segment info,
# output with matplot has audio graph as piecewise linear in grey
# overlayed with spline curve in green and n interpolation points highlighted in red.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# Another important difference from wavspline.py is that we allow endpoints of interval [a,b]
# to be float time values measured in samples, instead of integer sample values only.
# Instead of reading wav file, we start with the array of sample data as tensor or numpy array.
# The graph of signal is piecewise linear, connecting sample values by linear interpolation
# starting with math.floor(a) on the left, and ending with math.ceil(b) on the right.
# The graph of spline covers only the smaller interval [a,b].
#
# (this paragraph repeated from wavspline.py)
# Important note: the spline interpolation points are not uniform. We did this when
# forming splines to model cycles by choosing first a uniform sequence of k subintervals
# for the cubic spline and a knot sequence for the basis of B-splines for the vector 
# space V of C^2 cubic splines on the intervals [0,1/k,2/k,...,(k-1)/k,1].  We also chose
# the knot sequence 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1 since this gives a B-spline basis
# with the simple properties that for any f(t) given as a sum of the basis B-splines
# f(t) = SUM_{i=0}^{n-1} c_i B^3_i(t), where n=k+3 is the dimension of V, we have
# f(0) = c_0, f(1) = c_{n-1}, f'(0) = c_1, and f'(1) = c_{n-2}.  Further, the interpolation
# points (s_i,y_i) i=0,...,n-1 should be chosen on the interval [0,1] according to the
# Schoenberg-Whitney Theorem so that B^3_i(s_i) > 0, or equivalently such that 
# t_i < s_i < t_{i+4} for each i.  A uniform sequence s_i = i/(n-1) works for i=0,...,n-1
# but we chose to use the sequence of k+1 endpoints of subintervals and then to insert
# two more values to total n=k+3 at the midpoints of the first and last subintervals,
# at 1/(2k) and 1-1/(2k).  These values have the function of giving a slightly "tighter"
# fit of the spline at the endpoints.  Since we are fitting to audio data of one cycle
# at a time, this makes it more likely that there will be good agreement of derivatives 
# where cycle splines join at the endpoints.
#
# ----- ----- ----- ----- -----


import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import math

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 
from plotSegmentSpline import plotSegmentSpline

# testing plotSegmentSpline()
    
path = "../audio/"
file = "left.wav"
file = "dulcimerA3-f.wav"
path += file
waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
print("audio file loaded: ", file)
print("sample rate: ", sample_rate)
print("length in samples: ", num_frames)
num_segments = 16
print("splitting file into ", num_segments, " segments")
segments = torch.tensor_split(waveform, num_segments, dim=1)
for j in range(num_segments) :
    print("segment: ", j, " length in samples: ", segments[j].shape)
current_segment = 0
print("processing segment: ", current_segment)
waveform = segments[current_segment]

a = 35.5
b = 229.2
n = 80
plotSegmentSpline(waveform, sample_rate, a, b, n)



