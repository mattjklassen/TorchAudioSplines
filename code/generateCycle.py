
# ----- Brief Description -----
# 
# Generate one cycle as waveform, and return waveform.
# inputs: cycle = [a,b], B-spline coefficients vector = bcoeffs.
# assume a and b are time values in samples between integer points, so that the
# spline is computed on interval [a,b] and evaluated at M = floor(b)-floor(a) integer 
# points or samples to produce waveform sample values. The waveform will have
# values indexed 0 to M-1 at given sample_rate. 
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import torch
import torchaudio
import numpy as np
import math

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 



def genCycle(cycle, bcoeffs) :

    a = cycle[0]
    b = cycle[1]
    first_sample = math.ceil(a)
    last_sample = math.floor(b)
    num_samples = last_sample - first_sample + 1
    data = []
    for i in range(num_samples)
        

    return data

