
"""
Periodic Sparse Filtering.
Details on the code and the theory behind it available at: 

It builds on the work on Sparse Filtering (by Jiquan Ngiam):
https://papers.nips.cc/paper/4334-sparse-filtering.pdf
https://github.com/jngiam/sparseFiltering (Matlab code)
https://github.com/jmetzen/sparse-filtering (Python port)

Requires numpy, scipy.
"""


import numpy as np
from scipy.optimize import minimize
from utils import normalization, nonlinearities, numpymatrix

class PSFilter(object):    
    
    ### INITIALIZERS ###
    def __init__(self, iterations=500, lambdaA=1.0, lambdaD=1.0, mu=1.0, 
                 nonlinearity=nonlinearities.softabsolute, deriv_nonlinearity=nonlinearities.deriv_softabsolute):
        self.name = 'PSF'
        self.iterations = iterations
        self.lambdaA = lambdaA
        self.lambdaD = lambdaD
        self.mu = mu
        self.nonlinearity = nonlinearity
        self.deriv_nonlinearity = deriv_nonlinearity
            
    def setFeatures(self,F0,F1,FU):
        self.F0 = F0
        self.F1 = F1
        self.FU = FU
        self.learned_features = F0+F1+FU
        
    def setTrainData(self,X_tr,y_tr,X_te):
        # This function is an external-internal interface
        # It receives data in the external shape [samples x features] and it sets parameters in the
        # internal shape [features x samples]
        self.data_nsamples = X_tr.shape[0] + X_te.shape[0]
        self.original_features = X_tr.shape[1]
        
        self.data_Y_0 = X_tr[y_tr==0].T
        self.data_Y_1 = X_tr[y_tr==1].T
        self.data_Y_U = X_te.T
        self.data = np.concatenate((self.data_Y_0,self.data_Y_1,self.data_Y_U),axis=1)
                
        self.data_nsamples_Y_0 = self.data_Y_0.shape[1]
        self.data_nsamples_Y_1 = self.data_Y_1.shape[1]
        self.data_nsamples_Y_U = self.data_Y_U.shape[1]
                        
        
    ### INITIALIZING WEIGHTS ###   
    def initializeWeights(self):
        self.W = np.random.randn(self.learned_features,self.original_features)
         
            
    ### TRAINING ###
    def train(self):
        optW = self.W
    
        def objFun(W):
            ## Feed forward
            W = W.reshape((self.learned_features,self.original_features))
            WA = numpymatrix.submatrix(W, 0,self.F0, 0,W.shape[1])
            WD = numpymatrix.submatrix(W, self.F0,self.F0+self.F1, 0,W.shape[1])
            XA = numpymatrix.submatrix(self.data, 0,self.data.shape[0], 0,self.data_nsamples_Y_0)
            XD = numpymatrix.submatrix(self.data, 0,self.data.shape[0], self.data_nsamples_Y_0,self.data_nsamples_Y_0+self.data_nsamples_Y_1)
            
            # Compute Z
            linWX = np.dot(W,self.data)
            F = self.nonlinearity(linWX)
            Fsquish, L2Fs = normalization.l2row(F)
            Fhat, L2Fn = normalization.l2row(Fsquish.T)
            
            # Compute A
            linA = np.dot(WA,XA)
            A = self.nonlinearity(linA)
            Asquish, L2As = normalization.l2row(A)
            Ahat, L2An = normalization.l2row(Asquish.T)
            
            # Compute D
            linD = np.dot(WD,XD)
            D = self.nonlinearity(linD)
            Dsquish, L2Ds = normalization.l2row(D)
            Dhat, L2Dn = normalization.l2row(Dsquish.T)
            
            # Record iteration
            self.callbackIteration()
                        
            ## Derivative of Sparse Filtering Loss Function
            ## Backprop through each feedforward step
            DeltaW = normalization.l2rowg(Fsquish.T, Fhat, L2Fn, np.ones(Fhat.shape))
            DeltaW = normalization.l2rowg(F, Fsquish, L2Fs, DeltaW.T)
            DeltaW = self.mu * ((DeltaW*(self.deriv_nonlinearity(linWX))).dot(self.data.T))
            
            DeltaA = normalization.l2rowg(Asquish.T, Ahat, L2An, np.ones(Ahat.shape))
            DeltaA = normalization.l2rowg(A, Asquish, L2As, DeltaA.T)
            DeltaA = self.lambdaA * ((DeltaA*(self.deriv_nonlinearity(linA))).dot(XA.T))
            
            DeltaD = normalization.l2rowg(Dsquish.T, Dhat, L2Dn, np.ones(Dhat.shape))
            DeltaD = normalization.l2rowg(D, Dsquish, L2Ds, DeltaD.T)
            DeltaD = self.lambdaD * ((DeltaD*(self.deriv_nonlinearity(linD))).dot(XD.T))
            
            DeltaW[0:self.F0,:] =  DeltaW[0:self.F0,:] - DeltaA
            DeltaW[self.F0:self.F0+self.F1,:] = DeltaW[self.F0:self.F0+self.F1,:] - DeltaD
             
            return (self.mu*Fhat.sum() - self.lambdaD* Dhat.sum() - self.lambdaA*Ahat.sum()), DeltaW.flatten()
                
        # Optimization
        self.current_iteration = 0
        _,_ = objFun(optW)
        res = minimize(objFun, optW, method='L-BFGS-B', jac = True, options = {'maxiter':self.iterations, 'disp':False})
        self.W = res.x.reshape(self.learned_features,self.original_features)



    ### CALLBACK MODULE ###    
    def callbackIteration(self):
        # Record your stats   
        return None 
         
                             
    ### FEEDFORWARD MODULE ###           
    def feedForward(self,data):
        # This function is an external function
        # It receives data in the external shape [samples x features] and it returns results in the same
        # external shape [samples x features]
        WX = np.dot(self.W, data.T)
        F = self.nonlinearity(WX)
        
        Fsquish = normalization.l2row(F)[0]
        return normalization.l2row(Fsquish.T)[0]