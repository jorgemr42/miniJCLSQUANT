import scipy as sci
import numba as numba
import numpy as np
from time import perf_counter
import copy
import tqdm
from math import pi,sqrt

@numba.njit(parallel=True)
def diagonal(data,diagonal,len_row,indices,diagonal_array):
    """
    Def:
        Modifies elements of the matrix that are in the diagonal
    Inputs:
        diagonal_array: array of len=len(A.data)
    Outputs:
        A: Ell matrix with diagonal elements modified.
    """
    if len(diagonal)==1 :    
        diagonal_new=diagonal_array
    else:
        diagonal_new=np.zeros(len(diagonal),dtype=diagonal.dtype)
        for i in numba.prange(len(diagonal)):
            diagonal_new[i]=diagonal[i]+diagonal_array[i]     
    return data,diagonal_new
# @numba.njit(parallel=True)
def Ham_reescalation(data,diagonal,len_row,indices,bounds_array):
    """
    Def:
        Modifies the matrix element of a matrix by dividing them by, ((bounds_array[1]-bounds_array[0])/2). 
        Only suitable for symmetric bonds.
    Inputs:
        bounds_array: array of bounds such that bounds_array[0]=-bounds_array[1] 
    Outputs:
        A: Ell matrix with elements modified.
    """
    data_new=np.zeros(len(data),dtype=data.dtype)

    if len(diagonal)==1 :
        for i in numba.prange(len(data)):
            data_new[i]=data[i]/((bounds_array[1]-bounds_array[0])/2)
            
        diagonal_new=np.zeros(1,dtype=diagonal.dtype)
    else:
        diagonal_new=np.zeros(len(diagonal),dtype=diagonal.dtype)
        for i in numba.prange(len(data)):
            data_new[i]=data[i]/((bounds_array[1]-bounds_array[0])/2)
            if i <len(diagonal):
                diagonal_new[i]=diagonal[i]/((bounds_array[1]-bounds_array[0])/2)        
    return data_new,diagonal_new

##Velocity computation
@numba.njit(parallel=False)
def velocities(data,diagonal,len_row,indices,D_data):
    """
    Def:
        Modifies the matrix element of a matrix by multiplyng them with (-1j)*D_data. 
    Inputs:
        D_data: Array of the difference in position of each amtrix element. 
    Outputs:
        A: Ell matrix with elements modified.
    """
    data_new=np.zeros(len(data),dtype=data.dtype)
    for i in numba.prange(len(data)):
        data_new[i]=data[i]*(-1j)*((D_data[i]))
    diagonal_new=np.zeros(1,dtype=np.int64)
    return data_new,diagonal_new

##Velocity computation plus reescalation of velocities
@numba.njit(parallel=True)
def velocities_reescalation(data,diagonal,len_row,indices,bounds_array,D_data):
    """
    Def:
        Modifies the matrix element of a matrix by multiplyng them with (1j)*(-1j)*D_data/((bounds_array[1]-bounds_array[0])/2). 
    Inputs:
        bounds_array: array of bounds such that bounds_array[0]=-bounds_array[1] 
        D_data: Array of the difference in position of each amtrix element. 
    Outputs:
        A: Ell matrix with elements modified.
    """
    data_new=np.zeros(len(data),dtype=data.dtype)
    for i in numba.prange(len(data)):
        data_new[i]=data[i]*1j*(((-1j)*(D_data[i])))/((bounds_array[1]-bounds_array[0])/2)
    diagonal_new=np.zeros(1,dtype=np.int64)
    return data_new,diagonal_new

## Circle polarized light with Peierls substitution
@numba.jit(parallel=True)
def circle_light(data,diagonal,len_row,indices,mod_params,t,D1_data,D2_data,conjugates):
    """
    Def:
        Modifies the matrix element of a matrix by multiplyng them with (np.exp(1j*A0*(np.cos(w*t)*(D1_data)+((-1)**conjugates)*1j*np.sin(w*t)*(D2_data)))) 
        for right polarization and (np.exp(1j*A0*(np.cos(w*t)*(D1_data)-((-1)**conjugates)*1j*np.sin(w*t)*(D2_data)))) for left polarization. 
    Inputs:
        mod_params: array of type [A0,w,pol], where A0 is the amplitude,w is the frecuency and pol=0.0,1.0 for right,left polarization
        t: time
        D1_data: Array of the difference in X position of each amtrix element. 
        D2_data: Array of the difference in Y position of each amtrix element. 
        conjugates: Array of 0's and 1's in such way that (i,j) element has 0 and (j,i) element has 1.
    Outputs:
        A: Ell matrix with elements modified.
    """
    data_new=np.zeros(len(data),dtype=np.complex128)

    [A0,w,pol]=mod_params
    if pol==0.0: #Right polarization
        if len(diagonal)==1  :
            for i in numba.prange(len(data)):
                data_new[i]=data[i]*(np.exp(1j*A0*(np.cos(w*t)*(D1_data[i])+((-1)**conjugates[i])*1j*np.sin(w*t)*(D2_data[i]))))
     
            diagonal_new=np.zeros(1,dtype=np.complex128)

        else:
            diagonal_new=np.zeros(len(diagonal),dtype=np.complex128)

            for i in numba.prange(len(data)):
                data_new[i]=data[i]*(np.exp(1j*A0*(np.cos(w*t)*(D1_data[i])+((-1)**conjugates[i])*1j*np.sin(w*t)*(D2_data[i]))))
            diagonal_new=diagonal.astype(np.complex128)

    else:#Left polarization
        if len(diagonal)==1 :
            for i in numba.prange(len(data)):
                data_new[i]=data[i]*(np.exp(1j*A0*(np.cos(w*t)*(D1_data[i])-((-1)**conjugates[i])*1j*np.sin(w*t)*(D2_data[i]))))

            diagonal_new=np.zeros(1,dtype=np.complex128)
        else:
            diagonal_new=np.zeros(len(diagonal),dtype=np.complex128)
            for i in numba.prange(len(data)):
                data_new[i]=data[i]*(np.exp(1j*A0*(np.cos(w*t)*(D1_data[i])-((-1)**conjugates[i])*1j*np.sin(w*t)*(D2_data[i]))))
 
            diagonal_new=diagonal.astype(np.complex128)

    return data_new,diagonal_new


@numba.jit(parallel=True)
def circle_light_packed(data,diagonal,len_row,indices,mod_params,t,D1_data,D2_data,conjugates):
    """
    Def:
        Modifies the matrix element of a matrix by multiplyng them with (np.exp(1j*A0*(np.cos(w*t)*(D1_data)+((-1)**conjugates)*1j*np.sin(w*t)*(D2_data)))) 
        for right polarization and (np.exp(1j*A0*(np.cos(w*t)*(D1_data)-((-1)**conjugates)*1j*np.sin(w*t)*(D2_data)))) for left polarization. 
    Inputs:
        mod_params: array of type [A0,w,pol], where A0 is the amplitude,w is the frecuency and pol=0.0,1.0 for right,left polarization
        t: time
        D1_data: Array of the difference in X position of each amtrix element. 
        D2_data: Array of the difference in Y position of each amtrix element. 
        conjugates: Array of 0's and 1's in such way that (i,j) element has 0 and (j,i) element has 1.
    Outputs:
        A: Ell matrix with elements modified.
    """
    data_new=np.zeros(len(data),dtype=np.complex128)

    [A0,w,pol,Tp]=mod_params
    if pol==0.0: #Right polarization
        if len(diagonal)==1  :
            for i in numba.prange(len(data)):
                data_new[i]=data[i]*(np.exp(1j*A0*(np.cos(w*t)*(D1_data[i])+((-1)**conjugates[i])*1j*np.sin(w*t)*(D2_data[i]))/np.cosh((t-2*Tp)/(0.5673*Tp))))
     
            diagonal_new=np.zeros(1,dtype=np.complex128)

        else:
            diagonal_new=np.zeros(len(diagonal),dtype=np.complex128)

            for i in numba.prange(len(data)):
                data_new[i]=data[i]*(np.exp(1j*A0*(np.cos(w*t)*(D1_data[i])+((-1)**conjugates[i])*1j*np.sin(w*t)*(D2_data[i]))/np.cosh((t-2*Tp)/(0.5673*Tp))))
            diagonal_new=diagonal.astype(np.complex128)

    else:#Left polarization
        if len(diagonal)==1 :
            for i in numba.prange(len(data)):
                data_new[i]=data[i]*(np.exp(1j*A0*(np.cos(w*t)*(D1_data[i])-((-1)**conjugates[i])*1j*np.sin(w*t)*(D2_data[i]))/np.cosh((t-2*Tp)/(0.5673*Tp))))

            diagonal_new=np.zeros(1,dtype=np.complex128)
        else:
            diagonal_new=np.zeros(len(diagonal),dtype=np.complex128)
            for i in numba.prange(len(data)):
                data_new[i]=data[i]*(np.exp(1j*A0*(np.cos(w*t)*(D1_data[i])-((-1)**conjugates[i])*1j*np.sin(w*t)*(D2_data[i]))/np.cosh((t-2*Tp)/(0.5673*Tp))))
 
            diagonal_new=diagonal.astype(np.complex128)

    return data_new,diagonal_new






@numba.jit(parallel=True)
def linear_light(data,diagonal,len_row,indices,mod_params,t,D2_data):
    """
    THIS DATA IS NOT UPDATED
    Def:

        Modifies the matrix element of a matrix by multiplyng them with (np.exp(1j*A0*(np.cos(w*t)*(D1_data)+((-1)**conjugates)*1j*np.sin(w*t)*(D2_data)))) 
        for right polarization and (np.exp(1j*A0*(np.cos(w*t)*(D1_data)-((-1)**conjugates)*1j*np.sin(w*t)*(D2_data)))) for left polarization. 
    Inputs:
        mod_params: array of type [A0,w,pol], where A0 is the amplitude,w is the frecuency and pol=0.0,1.0 for right,left polarization
        t: time
        D1_data: Array of the difference in X position of each amtrix element. 
        D2_data: Array of the difference in Y position of each amtrix element. 
        conjugates: Array of 0's and 1's in such way that (i,j) element has 0 and (j,i) element has 1.
    Outputs:
        A: Ell matrix with elements modified.
    """

    data_new=np.zeros(len(data),dtype=np.complex128)

    [A0,w,Tp]=mod_params
    if len(diagonal)==1  :
        for i in numba.prange(len(data)):
            data_new[i]=data[i]*np.exp(1j*A0*np.sin(w*t)*(D2_data[i])/np.cosh((t-2*Tp)/(0.5673*Tp)))
    
        diagonal_new=np.zeros(1,dtype=np.complex128)

    else:
        diagonal_new=np.zeros(len(diagonal),dtype=np.complex128)

        for i in numba.prange(len(data)):
            data_new[i]=data[i]*np.exp(1j*A0*np.sin(w*t)*(D2_data[i])/np.cosh((t-2*Tp)/(0.5673*Tp)))
        diagonal_new=diagonal.astype(np.complex128)


    return data_new,diagonal_new