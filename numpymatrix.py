
"""
Support library for extracting numpy submatrices.

Requires numpy.
"""

def submatrix(X,r0,r1,c0,c1):
    
    if (r1-r0 == 1) and (c1-c0 == 1):
        return X[r0:r1,c0:c1].reshape(1,1)
    elif (r1-r0 == 1):
        return X[r0:r1,c0:c1].reshape(1,c1-c0)
    elif (c1-c0 == 1):
        return X[r0:r1,c0:c1].reshape(r1-r0,1)
    else:
        return X[r0:r1,c0:c1]
    