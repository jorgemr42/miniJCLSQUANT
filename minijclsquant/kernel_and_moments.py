import scipy as sci
import numpy as np
from time import perf_counter
import copy
import tqdm
from math import pi,sqrt
import warnings
from jclsquant.cython_modules.Extra_funs import table_G_delta


def moments_by_matrix(moments_vec,matrix):
    result_sum=np.zeros(np.shape(matrix)[1],dtype=complex)
    for i in range(len(moments_vec)):
        result_sum+=moments_vec[i]*matrix[i,:]
    return result_sum


#Kernel
def JacksonKernel(N):  #It gives all the element of the Kernel for a given Number of moments (N)
    n=np.arange(0,N)
    g=(N-n+1)*np.cos(np.pi*n/(N+1))+np.sin(np.pi*n/(N+1))*(1/np.tan(np.pi/(N+1)))
    g=g/(N+1)
    #g[0]=g[0]/2  #not needed <- LSQUANT includes this in the moments already
    return g

############# Moments
def moments_FD_T0(Ef,N):
    n=np.arange(1,N)    
    n=-2*np.sin(n*(np.arccos(Ef)))/(n*pi)
    n=np.concatenate((np.array([1-np.arccos(Ef)/pi]),n))
    return n

def moments_Gmas_2(Ef,broadening,N):

    z=Ef+1j*broadening
    m=np.arange(1,N)
    
    n=2*(z-1j*np.sqrt(1-z**2))**m/(1j*np.sqrt(1-z**2))
    
    n=np.concatenate((np.array([-1j/np.sqrt(1-z**2)]),n))
    return (n)

def moments_delta_2(Ef,broadening,M):
    return -np.imag(moments_Gmas_2(Ef,broadening,M))/np.pi


def moments_Gmas(Ef,N):

    n=np.arange(1,N)
    
    n=-2j*np.exp(-1j*n*np.arccos(Ef))/sqrt(1-Ef**2)
    
    n=np.concatenate((np.array([-1j/sqrt(1-Ef**2)]),n))
    return (n)

def moments_Gmin(Ef,N):

    n=np.arange(1,N)    
    n=2j*np.exp(1j*n*np.arccos(Ef))/sqrt(1-Ef**2)
    
    n=np.concatenate((np.array([1j/sqrt(1-Ef**2)]),n))
    return (n)

def moments_Gsum(Ef,N):
    n=np.arange(1,N)    
    n=-4*np.sin(n*np.arccos(Ef))/sqrt(1-Ef**2)
    
    n=np.concatenate((np.array([0]),n))
    return (n)


def moments_delta(Ef,N):
    #Ef must be renormalised
    n=np.arange(1,N)    
    n=2*np.cos(n*np.arccos(Ef))
    
    n=np.concatenate((np.array([1]),n))
    return n/(pi*sqrt(1-Ef**2))



def moments_U(t,N):
    n=np.arange(1,N)    
    n=2*(-1j)**n*sci.special.jv(n,t)
    return np.concatenate((np.array([sci.special.jv(0,t)]),n))

def moments_FD_T(Ef,T_mod,N):
    if type(Ef)==np.float64:
        Ef2=[Ef]
    else:
        Ef2=Ef
    mu_vec=np.zeros(N)
    E_vec=np.linspace(-1+1e-3,1-1e-3,2*N)
    mu_vec_E=np.zeros(len(E_vec))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for m in range(N):
            mu_vec_E=(1/(1+np.exp((E_vec-Ef)/T_mod)))*np.cos(m*np.arccos(E_vec))/(pi*np.sqrt(1-E_vec**2))
            if m==0:
                mu_vec[m]=sci.integrate.simpson(mu_vec_E,x=E_vec,dx=np.abs(E_vec[1]-E_vec[0]))
            else:
                mu_vec[m]=2*sci.integrate.simpson(mu_vec_E,x=E_vec,dx=np.abs(E_vec[1]-E_vec[0]))
    return mu_vec



def moments_FD(N,FD):
    mu_vec=np.zeros(N)
    E_vec=np.linspace(-1+1e-3,1-1e-3,len(FD))
    mu_vec_E=np.zeros(len(E_vec))
    for m in range(N):
        mu_vec_E=FD*np.cos(m*np.arccos(E_vec))/(pi*np.sqrt(1-E_vec**2))
        mu_vec[m]=2*sci.integrate.simpson(mu_vec_E,x=E_vec,dx=np.abs(E_vec[1]-E_vec[0]))
    mu_vec[0]=mu_vec[0]/2
    return mu_vec