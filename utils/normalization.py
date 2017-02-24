
"""
Support library for computing L2 normalization.

Based on the Python port (by Jan Hendrik Metzen) of the sparse filtering matlab code (by Jiquan Ngiam). See:
https://github.com/jngiam/sparseFiltering
https://github.com/jmetzen/sparse-filtering

Requires numpy.
"""

import numpy as np

def l2row(X):    
    N = np.sqrt((X**2).sum(axis=1)+1e-8)
    Y = (X.T/N).T
    return Y,N
     
def l2rowg(X,Y,N,D):
    return (D.T/N - Y.T * (D*X).sum(axis=1) / N**2).T
