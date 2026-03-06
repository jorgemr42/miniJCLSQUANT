import scipy as sci
import numpy as np
from minijclsquant.cython_modules.blas_funs import vdot,axpy2 # type: ignore
from minijclsquant.cython_modules.blas_funs import aAxby # type: ignore




def rec_A_tab(M,A,v = None):

    if v is None:
        print('You must introduce a vector.')
        return 
    len_v=len(v)
    ## Set things to zero
    A_vec=np.zeros(M,dtype=np.complex128)
    
    T_0=np.copy(v,order='C')
    T_1=np.zeros(len(v),dtype=np.complex128)
    
    ## Set T_1 to A*v
    A.dot(1+0j,0+0j,v,T_1)
    
    # Set the first elements of the array
    
    A_vec[0]=vdot(1+0j,v,v,len_v,A.n_threads)
    A_vec[1]=vdot(1+0j,v,T_1,len_v,A.n_threads)
        
    
    ## Recurrence loop
    for m in range(2,M):
        A.dot(2.0+0j,-1+0j,T_1,T_0)

        temp = T_0
        T_0 = T_1
        T_1 = temp    

        A_vec[m] = vdot(1+0j,v,T_1,len_v,A.n_threads)
    
    return A_vec



def rec_A_tab2v(M,A,v_l=None,v_r=None):

    if v_r is None or v_l is None:
        print('You must introduce a right and left vector.')
        return 
    len_v=len(v_r)
    ## Set things to zero
    A_vec=np.zeros(M,dtype=np.complex128)
    
    T_0=np.copy(v_r,order='C')
    T_1=np.zeros(len(v_r),dtype=np.complex128)
    
    ## Set T_1 to A*v
    A.dot(1+0j,0+0j,v_r,T_1)
    
    # Set the first elements of the array
    
    A_vec[0]=vdot(1+0j,v_l,v_r,len_v,A.n_threads)
    A_vec[1]=vdot(1+0j,v_l,T_1,len_v,A.n_threads)
    
    ## Recurrence loop
    for m in range(2,M):
        A.dot(2.0+0j,-1+0j,T_1,T_0)

        temp = T_0
        T_0 = T_1
        T_1 = temp    

        A_vec[m] = vdot(1+0j,v_l,T_1,len_v,A.n_threads)
    
    return A_vec

def rec_A_tab2v_2(M,A,v_l=None, v_r=None):

    if v_r is None or v_l is None:
        print('You must introduce a right and left vector.')
        return 
    len_v=len(v_r)
    ## Set things to zero
    A_vec_1=np.zeros(M,dtype=np.complex128)
    A_vec_2=np.zeros(M,dtype=np.complex128)
    
    T_0=np.copy(v_r,order='C')
    T_1=np.zeros(len(v_r),dtype=np.complex128)
    
    ## Set T_1 to A*v
    A.dot(1+0j,0+0j,v_r,T_1)
    
    # Set the first elements of the array
    
    A_vec_1[0]=vdot(1+0j,v_l,v_r,len_v,A.n_threads)
    A_vec_1[1]=vdot(1+0j,v_l,T_1,len_v,A.n_threads)    
    
    A_vec_2[0]=vdot(1+0j,v_r,v_r,len_v,A.n_threads)
    A_vec_2[1]=vdot(1+0j,v_r,T_1,len_v,A.n_threads)
    
    ## Recurrence loop
    for m in range(2,M):
        A.dot(2.0+0j,-1+0j,T_1,T_0)

        temp = T_0
        T_0 = T_1
        T_1 = temp    

        A_vec_1[m] = vdot(1+0j,v_l,T_1,len_v,A.n_threads)
        A_vec_2[m] = vdot(1+0j,v_r,T_1,len_v,A.n_threads)
    
    return A_vec_1,A_vec_2


def rec_A_vec(M,A,moments_kernel=None,v=None):
    if moments_kernel is None:
        print('You must introduce the analytical_moments*Kernel which is an array of size M')
        return 
    if v is None:
        print('You must introduce a vector.')
        return 
    len_v=len(v)
    
    ## Set things to zero
    T_0=np.copy(v,order='C')
    T_1=np.zeros(len(v),dtype=np.complex128)
    
    ## Set T_1 to A*v
    A.dot(1+0j,0+0j,v,T_1)
    
    # Set the first elements of the array
    A_kpm=moments_kernel[0]*v
    A_kpm+=moments_kernel[1]*T_1
    
    ## Recurrence loop
    for m in range(2,M):
        A.dot(2.0+0j,-1+0j,T_1,T_0)
        temp = T_0
        T_0 = T_1
        T_1 = temp 

        axpy2(moments_kernel[m],1+0j,T_1,A_kpm,len_v,A.n_threads)
    
    return A_kpm

## [r,A]
def rec_com_A_vec(M,A,V = None,moments_kernel = None,v = None):
    if V is None:
        print('You must introduce kpm renormalized velocity .')
        return 
    if moments_kernel is None:
        print('You must introduce the analytical_moments*Kernel which is an array of size M .')
        return 
    if v is None:
        print('You must introduce a vector.')
        return 
    len_v=len(v)
    ## Set the two recursive polinomials
    # First polynomial
    T_0=np.copy(v,order='C')
    # Set T1=H*v
    T_1=np.zeros(len(v),dtype=np.complex128)
    A.dot(1+0j,0+0j,v,T_1)
    
    ## Set the commutators

    # The First commutator is zero
    C_0=np.zeros(len(v),dtype=np.complex128)
    # The second commutator is C_1 = V*v
    C_1=np.zeros(len(v),dtype=np.complex128)
    V.dot(1+0j,0+0j,v,C_1)


    # Set the first elements of the final commutator
    
    C_kpm = moments_kernel[1]*C_1 # The firts one is zero
    
    ## Recurrence loop
    for m in range(2,M):
        # The recursive relation of the commutator
        
        A.dot(2.0+0j,-1+0j,C_1,C_0)
        V.dot(2.0+0j,1+0j,T_1,C_0)
        temp = C_0
        C_0 = C_1
        C_1 = temp 
        # The recursive relation of the Tm
        
        A.dot(2.0+0j,-1+0j,T_1,T_0)
        temp = T_0
        T_0 = T_1
        T_1 = temp 

        # C_kpm += moments_kernel[m]*C_1
        axpy2(moments_kernel[m],1+0j,C_1,C_kpm,len_v,A.n_threads)

    return C_kpm




## [r,A]
def rec_com_A_vec_tab(M,A,V = None,moments_kernel=None,v = None):
    if V is None:
        print('You must introduce kpm renormalized velocity .')
        return 
    if v is None:
        print('You must introduce a vector.')
        return 
    len_v=len(v)
    ## Set the two recursive polinomials
    # First polynomial
    T_0=np.copy(v,order='C')
    # Set T1=H*v
    T_1=np.zeros(len(v),dtype=np.complex128)
    A.dot(1+0j,0+0j,v,T_1)
    
    ## Set the commutators

    # The First commutator is zero
    C_0=np.zeros(len(v),dtype=np.complex128)
    # The second commutator is C_1 = V*v
    C_1=np.zeros(len(v),dtype=np.complex128)
    V.dot(1+0j,0+0j,v,C_1)

    C_tab=np.zeros((M,len(v)),dtype=np.complex128)
    C_tab[0,:]=moments_kernel[0]*C_0
    C_tab[1,:]=moments_kernel[1]*C_1
    
    
    
    ## Recurrence loop
    for m in range(2,M):
        # The recursive relation of the commutator
        
        A.dot(2.0+0j,-1+0j,C_1,C_0)
        V.dot(2.0+0j,1+0j,T_1,C_0)
        temp = C_0
        C_0 = C_1
        C_1 = temp 
        # The recursive relation of the Tm
        
        A.dot(2.0+0j,-1+0j,T_1,T_0)
        temp = T_0
        T_0 = T_1
        T_1 = temp 

        # C_kpm += moments_kernel[m]*C_1
        C_tab[m,:]=moments_kernel[m]*C_1

    return C_tab
