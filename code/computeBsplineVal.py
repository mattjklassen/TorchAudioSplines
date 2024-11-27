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
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# Note: most arrays are tensors, or numpy arrays, but not python lists.
#
# ----- ----- ----- ----- -----


import numpy as np
import torch

# computeSplineVal(d, k, bcoeffs, t01)

# computes f(t)
def computeSplineVal2(d, bcoeffs, knotVals, t) :
    N = len(knotVals) - 1
    n = len(bcoeffs)

    # deBoor points, which are set for j=0 to control points bcoeffs[i]
    controlCoeffs = torch.zeros(n, d+1)
    for i in range(n) :
        controlCoeffs[i, 0] = bcoeffs[i]

    fval = bcoeffs[0]

    J = 0
    denom = 1.0
    fac1 = 1.0
    fac2 = 1.0
    for i in range(1, N) :
        if (t < knotVals[i]) :
            J = i - 1
            break
    for p in range(1, d+1) :
        for i in range(J-d+p, J+1) :
            denom = (knotVals[i+d-(p-1)]-knotVals[i])
            if denom == 0 :
                break
            fac1 = (t-knotVals[i]) / denom
            fac2 = (knotVals[i+d-(p-1)]-t) / denom
            controlCoeffs[i, p] = fac1 * controlCoeffs[i, p-1] + fac2 * controlCoeffs[i-1, p-1]

    fval = controlCoeffs[J, d]

    return float(fval)


# computes f(t)
def computeSplineVal(d, k, c, t) :
    n = k + d  # dimension of splines
    N = n + d  # last index of knot sequence
    # c = array of n B-spline coefficients 

    # these are the deBoor points, which are set for j=0 to control points c[i]
    controlCoeffs = torch.zeros(n, d+1)
    for i in range(n) :
        controlCoeffs[i, 0] = c[i]

    fval = c[0]

    if (t > 0) :
        incr = 1 / k
        knotVals = torch.zeros(N+1)
        for i in range(N+1) :
            knotVals[i] = (i-d) * incr
            if (i < d) :
                knotVals[i] = 0
            if (i > N-d-1) :
                knotVals[i] = 1
        J = 0
        denom = 1.0
        fac1 = 1.0
        fac2 = 1.0
        for i in range(1, N) :
            if (t < knotVals[i]) :
                J = i - 1
                break
        for p in range(1, d+1) :
            for i in range(J-d+p, J+1) :
                denom = (knotVals[i+d-(p-1)]-knotVals[i])
                fac1 = (t-knotVals[i]) / denom
                fac2 = (knotVals[i+d-(p-1)]-t) / denom
                controlCoeffs[i, p] = fac1 * controlCoeffs[i, p-1] + fac2 * controlCoeffs[i-1, p-1]
    
        fval = controlCoeffs[J, d]

    if (t > 0.9999999) :
        fval = c[n-1]

    return float(fval)

# d = degree (default 3)
# k = number of subintervals
# j = index of B-spline B^3_j(t) to compute
# t = real value in [0,1]

# computes one B-spline B^d_j(t)
def newBsplineVal(d, k, j, t) :
    n = k + d  # dimension of splines
    N = n + d  # last index of knot sequence

    controlCoeffs = torch.zeros(n, n)
    controlCoeffs[j, 0] = 1
    incr = 1 / k
    knotVals = torch.zeros(N+1)
    for i in range(N+1) :
        knotVals[i] = (i-d) * incr
        if (i < d) :
            knotVals[i] = 0
        if (i > N-d-1) :
            knotVals[i] = 1
    J = 0
    denom = 1.0
    fac1 = 1.0
    fac2 = 1.0
    fval = 0.0
    for i in range(1, N) :
        if (t < knotVals[i]) :
            J = i - 1
            if (J > n-1) :
                J = n-1
            break
        else :
            J = n - 1
    for p in range(1, d+1) :
        for i in range(J-d+1, J+1) :
            denom = (knotVals[i+d-(p-1)]-knotVals[i])
            fac1 = (t-knotVals[i]) / denom
            fac2 = (knotVals[i+d-(p-1)]-t) / denom
            controlCoeffs[i, p] = fac1 * controlCoeffs[i, p-1] + fac2 * controlCoeffs[i-1, p-1]

    fval = controlCoeffs[J, d]
    return fval



