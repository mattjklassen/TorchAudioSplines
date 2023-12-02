
# ----- Brief Description -----
# 
# import and export knot sequences to text files knots-[description].txt
# also generate knotVals as standard sequence given n = dim of cubic splines
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# standard knot sequence has k = n - 3 subintervals and knot sequence:
# 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1
#
# ----- ----- ----- ----- -----

import torch
import numpy as np

# this function imports bcoeffs from text file of strings (printed floats)
# and returns a pytorch tensor of those floats.
def import_knots(file) :
    knots_str = []
    with open(file, 'r') as f:
        knots_str = f.readlines()
        f.close()
    N = len(knots_str) - 1
    knots = torch.zeros(N+1)
    for i in range(N+1) :
        knots[i] = float(knots_str[i])
    return knots


def export_knots(file, knots) :
    knots_str = []
    for i in range(len(knots)) :
        knots_str.append(str(float(knots[i])))
        knots_str.append('\n')
    with open(file, 'w') as f:
        f.writelines(knots_str)
        f.close()

def getKnots(n) :
    d = 3
    k = n - d
    N = n + d
    knotVals = np.zeros(N+1)
    incr = 1 / k
    for j in range(d+1, N-d) :
        knotVals[j] = knotVals[j-1] +  incr
    for j in range(N-d, N+1) :
        knotVals[j] = 1.0
    return knotVals






