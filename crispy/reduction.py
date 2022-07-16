# from tkinter import Y
import numpy as np
import numba
# from numba import njit

def sqrtm(A):
    "Calcualte ethe sqrt of the matrix"
    # Computing diagonalization
    evalues, evectors = np.linalg.eig(a)
    
    # Ensuring square root matrix exists
    # assert (evalues >= 0).all()

    sqrt_matrix = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)
    return sqrt_matrix

def lstsq(y, X, n):
    # beta =  inv(X.T @ W_inv @ X) @ X.T @ W_inv @ y
    pass #


def lsf(X):
    C_inv = X.T @ X
    Q = sqrtm(C_inv)
    Q /= Q.sum(axis=1)
    return Q




def lstsq_fit(y, X, niter=1):
    pixnoise=0
    nlam = X.shape[0]
    # inital guess
    
    coef = np.sum(y).reshape(-1,1,1) / nlam
    
    for i in range(niter):
        var = np.sum(X * coef, axis=0).reshape(-1) + pixnoise + 1e-10
        Ninv = np.diag(1./ var)
        tmp = np.dot(X.T, Ninv)
        Cinv = np.dot(tmp, X)
        C = np.linalg.inv(Cinv)
        right =  np.dot(tmp, y)
        f = np.dot(C, right)
        coef = f # np.dot(R, f)
        coef = np.linalg.inv(tmp @ X) @ tmp @ y
    
    return coef
        
        