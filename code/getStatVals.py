
# ----- Brief Description -----
# 
# get stationary points from bcoeffs on interval [0,1]
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# return a list of stationary points where the spline curve has approximate numerical derivative zero
#
# ----- ----- ----- ----- -----

import torch
import numpy as np
from computeBsplineVal import computeSplineVal2 

def getSplineVals(bcoeffs, knots, numVals) :

    n = len(bcoeffs) 
    d = 3
    N = len(knots) - 1
    k = n - d

    incr = 1 / numVals
    xvals = np.arange(start=0.0, stop=1 + incr, step=incr)
    # print(xvals)
    # print("size of xvals:  ", xvals.size)
    yvals = np.zeros(numVals + 1)
    for i in range(numVals + 1) :
        t = xvals[i]
        yvals[i] = computeSplineVal2(d, bcoeffs, knots, t)

    splineVals = [xvals, yvals]
    
    return splineVals


def getStatPts(splineVals) :

    stat_pts = []
    xvals = splineVals[0]
    yvals = splineVals[1]
    current_slope = 1
    previous_slope = 1
    stat_pts.append([xvals[0],yvals[0]])
    for i in range(1,len(xvals)) :
        # compute numerical derivative:
        current_slope = yvals[i] - yvals[i-1]
        if previous_slope * current_slope < 0 :
            # print("critical value: ", xvals[i-1])
            stat_pts.append([xvals[i-1],yvals[i-1]])
        previous_slope = current_slope
    stat_pts.append([xvals[-1],yvals[-1]])
    
#    stat_num = len(stat_pts)
#    stat_xvals = torch.zeros(stat_num + 2)
#    stat_yvals = torch.zeros(stat_num + 2)
#    stat_xvals[-1] = 1
#    for i in range(1, stat_num + 1) :
#        stat_xvals[i] = stat_pts[i-1][0]
#        stat_yvals[i] = stat_pts[i-1][1]

    return stat_pts


