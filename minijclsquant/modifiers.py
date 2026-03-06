from hmac import digest
from tracemalloc import start
import scipy as sci
import numpy as np
from time import perf_counter
import copy
from math import pi,sqrt
import random
from jclsquant.cython_modules.blas_funs import axpy,scal,vec_mul,vec_mul3,vec_mul2,axpy2
from jclsquant.cython_modules.Extra_funs import light_modifier_cython,diagonal_modifier,electron_hole_pud,electron_hole_pud3d,hopping_pos_phonons_modifier
from jclsquant.ell_matrix import *
import matplotlib.pyplot as plt

def modifier_bounds(A):
    """
    Def:
        Modifies the matrix element of a matrix by dividing them by, ((bounds_array[1]-bounds_array[0])/2). 
        Only suitable for symmetric bonds.
    Inputs:
        bounds_array: array of bounds such that bounds_array[0]=-bounds_array[1] 
    Outputs:
        A: Ell matrix with elements modified.
    """
    if np.abs(A.bounds[1])==np.abs(A.bounds[0]):
        data_new=np.copy(A.data)
        scal(1/((A.bounds[1]-A.bounds[0])/2)+0j,data_new,len(A.data),A.n_threads)
    else:
        data_new=np.zeros(len(A.data),dtype=np.complex128)

        diagonal_array=(-0.5*(A.bounds[1]+A.bounds[0]))*np.ones(A.shape[0],dtype=np.complex128)

        diagonal_modifier(A.data,A.indices,A.valid_row_values, data_new,  diagonal_array , A.len_row, len(A.data),A.n_threads)

        scal(1/(0.5*(A.bounds[1]-A.bounds[0]))+0j,data_new,len(A.data),A.n_threads)
    
    return data_new


def modifier_bounds_inverse(A):
    """
    Def:
        Modifies the matrix element of a matrix by dividing them by, ((bounds_array[1]-bounds_array[0])/2). 
        Only suitable for symmetric bonds.
    Inputs:
        bounds_array: array of bounds such that bounds_array[0]=-bounds_array[1] 
    Outputs:
        A: Ell matrix with elements modified.
    """
    data_new=np.copy(A.data)
    scal(((A.bounds[1]-A.bounds[0])/2)+0j,data_new,len(A.data),A.n_threads)
    return data_new




def modifier_velocity(A,direction = None):
    """
    Def:
        Modifies the matrix element of a matrix by dividing them by, ((bounds_array[1]-bounds_array[0])/2). 
        Only suitable for symmetric bonds.
    Inputs:
        bounds_array: array of bounds such that bounds_array[0]=-bounds_array[1] 
    Outputs:
        A: Ell matrix with elements modified.
    """
    data_new=np.copy(A.data)
    if direction is None:
        print(' You must introduce a desired direction: 0,1,2 or x,y,z ')
        return 
    elif direction == 0 or direction == 'x': 
        vec_mul(-1j, A.data,A.dx_vec,data_new,len(A.data),A.n_threads)
    elif direction == 1 or direction == 'y':
        vec_mul(-1j, A.data,A.dy_vec,data_new,len(A.data),A.n_threads)
    elif direction == 2 or direction == 'z':
        vec_mul(-1j, A.data,A.dz_vec,data_new,len(A.data),A.n_threads)
    else:
        print(' You must introduce a valid direction: 0,1,2 or x,y,z ')
        return
    return data_new


def modifier_velocity_bounds(A,direction = None):
    """
    Def:
        Modifies the matrix element of a matrix by multiplyng them with (1j)*(-1j)*D_data/((bounds_array[1]-bounds_array[0])/2) 
        Only suitable for symmetric bonds.
    Inputs:
        bounds_array: array of bounds such that bounds_array[0]=-bounds_array[1] 
    Outputs:
        A: Ell matrix with elements modified.
    """
    data_new=np.copy(A.data)
    if direction is None:
        print(' You must introduce a desired direction: 0,1,2 or x,y,z ')
        return 
    elif direction == 0 or direction == 'x': 
        vec_mul(1/((A.bounds[1]-A.bounds[0])/2)+0j, A.data,A.dx_vec,data_new,len(A.data),A.n_threads)
    elif direction == 1 or direction == 'y':
        vec_mul(1/((A.bounds[1]-A.bounds[0])/2)+0j, A.data,A.dy_vec,data_new,len(A.data),A.n_threads)
    elif direction == 2 or direction == 'z':
        vec_mul(1/((A.bounds[1]-A.bounds[0])/2)+0j, A.data,A.dz_vec,data_new,len(A.data),A.n_threads)
    else:
        print(' You must introduce a valid direction: 0,1,2 or x,y,z ')
        return  
    return data_new

def modifier_velocity_bounds_m(A,direction = None):
    """
    Def:
        Modifies the matrix element of a matrix by multiplyng them with (1j)*(-1j)*D_data/((bounds_array[1]-bounds_array[0])/2) 
        Only suitable for symmetric bonds.
    Inputs:
        bounds_array: array of bounds such that bounds_array[0]=-bounds_array[1] 
    Outputs:
        A: Ell matrix with elements modified.
    """
    data_new=np.copy(A.data)
    if direction is None:
        print(' You must introduce a desired direction: 0,1,2 or x,y,z ')
        return 
    elif direction == 0 or direction == 'x': 
        vec_mul(-1/((A.bounds[1]-A.bounds[0])/2)+0j, A.data,A.dx_vec,data_new,len(A.data),A.n_threads)
    elif direction == 1 or direction == 'y':
        vec_mul(-1/((A.bounds[1]-A.bounds[0])/2)+0j, A.data,A.dy_vec,data_new,len(A.data),A.n_threads)
    elif direction == 2 or direction == 'z':
        vec_mul(-1/((A.bounds[1]-A.bounds[0])/2)+0j, A.data,A.dz_vec,data_new,len(A.data),A.n_threads)
    else:
        print(' You must introduce a valid direction: 0,1,2 or x,y,z ')
        return  
    return data_new


def modifier_light(A,data_new,modifier_id=None,modifier_params=None,t=None):
    """
    Def :
        Modifies the matrix elements with the Peierls substitution
    Inputs :
        modifier_id : Different light types : linear circle , light_packed or circle packed
        modifier_params : Light parameters from the different light types = [A0 (all),w (all),t (all), pol (circle),Tp (packed)]
    Outputs :
        data_new : New data of the ELL matrix . 
    Notes :
        A0 is usually defined as A0=np.pi*Gamma/a_cc wher a_cc=0.246 [nm] is graphene .

    """
    if modifier_id is None or modifier_params is None:
        print(' You must introduce a valid id : light, circle,light_packed or circle packed with irs valid parameters .')
        return 
    
    elif modifier_id == 'linear':
        A0,w,=modifier_params
        vec_mul(1+0j, A.data,np.exp(1j*A0*np.sin(w*t)*A.dy_vec) ,data_new,len(A.data),A.n_threads)

    elif modifier_id == 'circle':
        A0,w,pol=modifier_params
        if pol == 'r': 
            vec_mul(1+0j, A.data,np.exp(1j*A0*(np.cos(w*t)*A.dx_vec+np.sin(w*t)*A.dy_vec)) , data_new,len(A.data),A.n_threads)
        elif pol == 'l': 
            vec_mul(1+0j, A.data,np.exp(1j*A0*(np.cos(w*t)*A.dx_vec-np.sin(w*t)*A.dy_vec)) , data_new,len(A.data),A.n_threads)
        else:
            print('You must introduce a valid polarization')
            return
    
    elif modifier_id == 'linear_packed':
        A0,w,Tp=modifier_params
        vec_mul(1+0j, A.data,np.exp(1j*A0*np.sin(w*t)*A.dy_vec/np.cosh((t-2*Tp)/(0.5673*Tp))) ,data_new,len(A.data),A.n_threads)

    elif modifier_id == 'circle_packed': 
        A0,w,pol,Tp=modifier_params
        if pol == 'r': 
            vec_mul(1+0j, A.data,np.exp(1j*A0*(np.cos(w*t)*A.dx_vec+np.sin(w*t)*A.dy_vec)/np.cosh((t-2*Tp)/(0.5673*Tp))) , data_new,len(A.data),A.n_threads)
        elif pol == 'l': 
            vec_mul(1+0j, A.data,np.exp(1j*A0*(np.cos(w*t)*A.dx_vec-np.sin(w*t)*A.dy_vec)/np.cosh((t-2*Tp)/(0.5673*Tp))) , data_new,len(A.data),A.n_threads)    
        else:
            print('You must introduce a valid polarization')
            return
    else:
        print(' You must introduce a valid id : light, circle,light_packed or circle packed with irs valid parameters .')
        return
    
    return data_new


def modifier_velocity_r_comm(A,direction_1 = None, direction_2 = None):
    """
    Def:
        Modifies the matrix element of a matrix by dividing them by, ((bounds_array[1]-bounds_array[0])/2). 
        Only suitable for symmetric bonds.
    Inputs:
        bounds_array: array of bounds such that bounds_array[0]=-bounds_array[1] 
    Outputs:
        A: Ell matrix with elements modified.
    """
    data_new=np.copy(A.data)
    if direction_1 is None or direction_2 is None:
        print(' You must introduce a desired direction: 0,1,2 or x,y,z ')
        return 
    
    elif direction_1 == 0 or direction_1 == 'x':

        if direction_2 == 0 or direction_2 == 'x':
            vec_mul3(1j, A.data,A.dx_vec,A.dx_vec,data_new,len(A.data),A.n_threads)
        elif direction_2 == 1 or direction_2 == 'y':
            vec_mul3(1j, A.data,A.dx_vec,A.dy_vec,data_new,len(A.data),A.n_threads)

    elif direction_1 == 1 or direction_1 == 'y':

        if direction_2 == 0 or direction_2 == 'x':
            vec_mul3(1j, A.data,A.dy_vec,A.dx_vec,data_new,len(A.data),A.n_threads)
        elif direction_2 == 1 or direction_2 == 'y':
            vec_mul3(1j, A.data,A.dy_vec,A.dy_vec,data_new,len(A.data),A.n_threads)
    else:
        print(' You must introduce a valid direction: 0,1,2 or x,y,z ')
        return
    return data_new


def modifier_diagonal(A,id=None,diagonal_array=None):
    """
    Def:
        Modifies the diagonal elements of a matrix. Important they must be already non-zero.
    Inputs:
        diagonal_array : Array of the same shape as A. 
    Outputs:
        A: Ell matrix with elements modified.
    """
    if diagonal_array is None :
        print('You must introduce valid modification of the diagonal .')
        return 
    if len(diagonal_array)!=A.shape[0]:
        print('You must introduce an array that is of the same shape as the matrix.')

    data_new=np.copy(A.data)
    

    diagonal_modifier(A.data,A.indices,A.valid_row_values, data_new,  diagonal_array ,A.len_row, len(A.data),A.n_threads)

    
    return data_new


def modifier_diagonal_mAB(A,m=None):
    """
    Def:
        Modifies the diagonal elements of a matrix with m and -m repteadly.
    Inputs:
        m : Value of change.
    Outputs:
        A: Ell matrix with elements modified.
    """

    if m is None:
        print('You must introduce a mass term to modify the system. ')
        return 
    

    data_new=np.copy(A.data)
    
    diagonal_array=np.array([m, -m] * (A.shape[0]//2))
    diagonal_modifier(A.data,A.indices,A.valid_row_values, data_new,  diagonal_array , A.len_row, len(A.data),A.n_threads)

    return data_new

def modifier_diagonal_mlayer(A,E=None):
    """
    Def:
        Modifies the diagonal elements of a matrix with -E the first hald and E the second.
    Inputs:
        E : strenght of the modification. 
    Outputs:
        A: Ell matrix with elements modified.
    """

    if E is None:
        print('You must introduce a mass term to modify the system. ')
        return 
    

    data_new=np.copy(A.data)
    
    diagonal_array=np.concatenate((-E*np.ones(A.shape[0]//2),E*np.ones(A.shape[0]//2)))
    diagonal_modifier(A.data,A.indices,A.valid_row_values, data_new,  diagonal_array , A.len_row, len(A.data),A.n_threads)

    return data_new


def modifier_diagonal_anderson(A,W=None,seed=None):
    """
    Def:
        Modifies the diagonal elements of a matrix randomly from -W to W reproducing Anderson disorder .
    Inputs:
        W : Strenght of the disorder
        seed : Seed for the random number generation.
    Outputs:
        A: Ell matrix with elements modified.
    """

    if W is None:
        print('You must introduce a disorder strength to modify the system. ')

    

    data_new=np.copy(A.data)
    rng = np.random.default_rng(seed)  # seed is an integer

    diagonal_array=(W)*(2*rng.random(A.shape[0])-1)
    diagonal_modifier(A.data,A.indices,A.valid_row_values, data_new,  diagonal_array , A.len_row, len(A.data),A.n_threads)

    return data_new

def modifier_diagonal_e_h_puddles(A,S=None,n=None,W=None,g=None,seed=None):
    """
    Def:
        Modifies the diagonal elements of a matrix randomly from -W to W reproducing Anderson disorder .
    Inputs:
        S : Positions as S[i,:] is the (x,y,z) of the i atom
        W : Percentage with sites with disorder
        W : Strenght of the disorder
        W : Eponential decay of the disorder with the distance
        seed : Seed for the random number generation.

    Outputs:
        A: Ell matrix with elements modified.
    """

    if S is None:
        print('You must introduce the position of the atoms to modify the diagonal. ')
    if W is None:
        print('You must introduce a disorder strength to modify the system. ')
    if n is None:
        print('You must introduce the percentage of the number of sites you want the disorder to affect. ')
    if g is None:
        print('You must introduce a lenght scale for the decay of the disorder. ')
    N_d=int(A.shape[0]*n/100)
    rng = np.random.default_rng(seed)  # Create a random generator with optional seed
    random_sites = rng.choice(A.shape[0], size=N_d, replace=False)
    random_sites = np.array(random_sites, order='C', dtype=np.int32)
    if N_d==0:
        print('Very small number of impurities for this system size the average is that the material is clean')
        return
    
    W=np.float64(W)
    g=np.float64(g)

    r = rng.integers(0, 2, size=N_d, dtype=np.int32)

    diagonal_array=np.zeros(A.shape[0],np.float64)    

    if np.abs(np.sum(S[:,2]))!=0:
        electron_hole_pud3d(diagonal_array,A.shape[0],r,np.ascontiguousarray(S[:,0]),np.ascontiguousarray(S[:,1]) ,np.ascontiguousarray(S[:,2]),random_sites , len(random_sites),W, g,A.n_threads)
    else:
        electron_hole_pud(diagonal_array,A.shape[0],r,np.ascontiguousarray(S[:,0]),np.ascontiguousarray(S[:,1]) ,random_sites , len(random_sites),W, g,A.n_threads)

    diagonal_array=diagonal_array.astype(np.complex128)
    data_new=np.copy(A.data)
    diagonal_modifier(A.data,A.indices,A.valid_row_values, data_new,  diagonal_array , A.len_row, len(A.data),A.n_threads)

    return data_new



def modifier_hoppings_c(A,data_new,dx_vec_new,dy_vec_new,modifier_id=None,modifier_params=None,t=None):
    if A.space=='r':
        return modifier_hoppings_c_r(A,data_new,dx_vec_new,dy_vec_new,modifier_id,modifier_params,t)
    elif A.space=='k':
        return modifier_hoppings_c_k(A,data_new,dx_vec_new,dy_vec_new,modifier_id,modifier_params,t)


def modifier_hoppings_c_r(A,data_new,dx_vec_new,dy_vec_new,modifier_id=None,modifier_params=None,t=None):
    """
    Def :
        Modifies the matrix elements with the Peierls substitution
    Inputs :
        modifier_id : Different light types : linear circle , light_packed or circle packed
        modifier_params : Light parameters from the different light types = [A0 (all),w (all),t (all), pol (circle),Tp (packed)]
    Outputs :
        data_new : New data of the ELL matrix . 
    Notes :
        A0 is usually defined as A0=np.pi*Gamma/a_cc wher a_cc=0.246 [nm] is graphene .

    """
    if modifier_id is None or modifier_params is None:
        print(' You must introduce a valid id : light, circle,light_packed or circle packed with irs valid parameters .')
        return 
    
    elif modifier_id == 'linear':
        A0,w,=modifier_params
        
        light_modifier_cython(1+0j,  A0,  t, w, 0+0j,0+0j, A.data, A.dx_vec,A.dy_vec,data_new, len(A.data),A.n_threads)
    
    elif modifier_id == 'circle':
        A0,w,pol=modifier_params
        if pol == 'r': 
            light_modifier_cython(1+0j,  A0,  t, w, 0+0j,1+0j, A.data, A.dx_vec,A.dy_vec,data_new, len(A.data),A.n_threads)
        
        elif pol == 'l': 
            light_modifier_cython(1+0j,  A0,  t, w, 0+0j,-1+0j, A.data, A.dx_vec,A.dy_vec,data_new, len(A.data),A.n_threads)
        else:
            print('You must introduce a valid polarization')
            return
    elif modifier_id == 'linear_packed':
        A0,w,Tp=modifier_params
        light_modifier_cython(1+0j,  A0,  t, w, Tp,0+0j, A.data, A.dx_vec,A.dy_vec,data_new, len(A.data),A.n_threads)

    elif modifier_id == 'circle_packed': 
        A0,w,pol,Tp=modifier_params
        if pol == 'r': 
            light_modifier_cython(1+0j,  A0,  t, w, Tp,1+0j, A.data, A.dx_vec,A.dy_vec,data_new, len(A.data),A.n_threads)
        elif pol == 'l': 
            light_modifier_cython(1+0j,  A0,  t, w, Tp,-1+0j, A.data, A.dx_vec,A.dy_vec,data_new, len(A.data),A.n_threads)
        else:
            print('You must introduce a valid polarization')
            return
    elif modifier_id == 'phonons_packed':
        
        a0,b,Aq,wq,pol,S=modifier_params
        
        e_pol=np.zeros((A.shape[0],2),dtype=complex,order='C')

        if pol=='chiral_+':
            q=np.array([4*np.pi/(3*np.sqrt(3)*a0),0],dtype=complex)
            e_pol[:,0]=np.array([1/np.sqrt(2), 1/np.sqrt(2)] * (A.shape[0]//2))
            e_pol[:,1]=np.array([1j/np.sqrt(2), -1j/np.sqrt(2)] * (A.shape[0]//2))
        elif pol=='chiral_-':
            q=np.array([-4*np.pi/(3*np.sqrt(3)*a0),0],dtype=complex)
            e_pol[:,0]=np.array([1/np.sqrt(2), 1/np.sqrt(2)] * (A.shape[0]//2))
            e_pol[:,1]=np.array([-1j/np.sqrt(2), 1j/np.sqrt(2)] * (A.shape[0]//2))
        elif pol=='optical':
            q=np.array([0,0],dtype=complex)
            e_pol[:,0]=np.array([0, 0] * (A.shape[0]//2))
            e_pol[:,1]=np.array([1, -1] * (A.shape[0]//2))
        else:
            print('You must introduce a valid polarization')
            return 

        hopping_pos_phonons_modifier(A.data,A.indices, A.dx_vec, A.dy_vec,data_new,dx_vec_new,dy_vec_new,a0,b,Aq,np.ascontiguousarray(e_pol[:,0]),np.ascontiguousarray(e_pol[:,1]),wq,t,np.ascontiguousarray(S[:,0]),np.ascontiguousarray(S[:,1]),q,A.len_row,len(A.data),A.n_threads)
        return data_new,dx_vec_new,dy_vec_new

    else:
        print(' You must introduce a valid id : light, circle,light_packed or circle packed with irs valid parameters .')
        return
    
    return data_new



def modifier_hoppings_c_k(A,data_new,dx_vec_new,dy_vec_new,modifier_id=None,modifier_params=None,t=None):
    """
    Def :
        Modifies the matrix elements with the Peierls substitution
    Inputs :
        modifier_id : Different light types : linear circle , light_packed or circle packed
        modifier_params : Light parameters from the different light types = [A0 (all),w (all),t (all), pol (circle),Tp (packed)]
    Outputs :
        data_new : New data of the ELL matrix . 
    Notes :
        A0 is usually defined as A0=np.pi*Gamma/a_cc wher a_cc=0.246 [nm] is graphene .

    """

    if modifier_id is None or modifier_params is None:
        print(' You must introduce a valid id : light, circle,light_packed or circle packed with irs valid parameters .')
        return 
    
    elif modifier_id == 'linear':
        A0,w,=modifier_params
        
        data_new=np.copy(A.data)
        
        axpy2(-A0*np.cos(w*t),1+0j, A.dx_vec,data_new, len(A.data),A.n_threads)
        ### HAY QUE CAMBIAR TAMBIEN DX Y DY
    elif modifier_id == 'circle':
        data_new=np.copy(A.data)
        A0,w,pol=modifier_params

        if pol == 'r': 

            axpy2((A0*np.cos(w*t))/(0.5*(A.bounds[1]-A.bounds[0])),1+0j, A.dx_vec,data_new, len(A.data),A.n_threads)
            axpy2((A0*np.sin(w*t))/(0.5*(A.bounds[1]-A.bounds[0])),1+0j, A.dy_vec,data_new, len(A.data),A.n_threads)
        
        elif pol == 'l': 

            axpy2((A0*np.cos(w*t))/(0.5*(A.bounds[1]-A.bounds[0])),1+0j, A.dx_vec,data_new, len(A.data),A.n_threads)
            axpy2(-(A0*np.sin(w*t))/(0.5*(A.bounds[1]-A.bounds[0])),1+0j, A.dy_vec,data_new, len(A.data),A.n_threads)
        
        else:
            print('You must introduce a valid polarization')
            return
    
    elif modifier_id == 'linear_packed':
        A0,w,Tp=modifier_params

        data_new=np.copy(A.data)
        
        axpy2((A0*np.cos(w*t)/ np.cosh((t - 2*Tp) / (0.5673*Tp)))/(0.5*(A.bounds[1]-A.bounds[0])),1+0j, A.dx_vec,data_new, len(A.data),A.n_threads)
        # axpy2((A0*np.sin(w*t)/ np.cosh((t - 2*Tp) / (0.5673*Tp)))/(0.5*(A.bounds[1]-A.bounds[0])),1+0j, A.dy_vec,data_new, len(A.data),A.n_threads)

    elif modifier_id == 'circle_packed': 
        A0,w,pol,Tp=modifier_params
        data_new=np.copy(A.data)
        if pol == 'r': 
            
            axpy2((A0*np.cos(w*t)/ np.cosh((t - 2*Tp) / (0.5673*Tp)))/(0.5*(A.bounds[1]-A.bounds[0])),1+0j, A.dx_vec,data_new, len(A.data),A.n_threads)
            axpy2((A0*np.sin(w*t)/ np.cosh((t - 2*Tp) / (0.5673*Tp)))/(0.5*(A.bounds[1]-A.bounds[0])),1+0j, A.dy_vec,data_new, len(A.data),A.n_threads)
        
        elif pol == 'l': 

            axpy2((A0*np.cos(w*t)/ np.cosh((t - 2*Tp) / (0.5673*Tp)))/(0.5*(A.bounds[1]-A.bounds[0])),1+0j, A.dx_vec,data_new, len(A.data),A.n_threads)
            axpy2(-(A0*np.sin(w*t)/ np.cosh((t - 2*Tp) / (0.5673*Tp)))/(0.5*(A.bounds[1]-A.bounds[0])),1+0j, A.dy_vec,data_new, len(A.data),A.n_threads)
        
        else:
            print('You must introduce a valid polarization')
            return
    elif modifier_id == 'phonons_packed':
        print('NOT IMPLEMENTED FOR THE MOMENT')
        data_new=np.copy(A.data)
        
        a0,b,Aq,wq,pol,S=modifier_params
        
        
        e_pol=np.zeros((A.shape[0],2),dtype=complex,order='C')

        if pol=='chiral_+':
            q=np.array([4*np.pi/(3*np.sqrt(3)*a0),0],dtype=complex)
            e_pol[:,0]=np.array([1/np.sqrt(2), 1/np.sqrt(2)] * (A.shape[0]//2))
            e_pol[:,1]=np.array([1j/np.sqrt(2), -1j/np.sqrt(2)] * (A.shape[0]//2))
        elif pol=='chiral_-':
            q=np.array([-4*np.pi/(3*np.sqrt(3)*a0),0],dtype=complex)
            e_pol[:,0]=np.array([1/np.sqrt(2), 1/np.sqrt(2)] * (A.shape[0]//2))
            e_pol[:,1]=np.array([-1j/np.sqrt(2), 1j/np.sqrt(2)] * (A.shape[0]//2))
        elif pol=='optical':
            q=np.array([0,0],dtype=complex)
            e_pol[:,0]=np.array([0, 0] * (A.shape[0]//2))
            e_pol[:,1]=np.array([1, -1] * (A.shape[0]//2))
        else:
            print('You must introduce a valid polarization')
            return 

        hopping_pos_phonons_modifier(A.data,A.indices, A.dx_vec, A.dy_vec,data_new,dx_vec_new,dy_vec_new,a0,b,Aq,np.ascontiguousarray(e_pol[:,0]),np.ascontiguousarray(e_pol[:,1]),wq,t,np.ascontiguousarray(S[:,0]),np.ascontiguousarray(S[:,1]),q,A.len_row,len(A.data),A.n_threads)
        return data_new,dx_vec_new,dy_vec_new

    else:
        print(' You must introduce a valid id : light, circle,light_packed or circle packed with irs valid parameters .')
        return
    
    return data_new




def modifier_random_hoppings(A,alpha=None,seed= None,velocities=None):
    """
    Def:
        Modifies the matrix element of a matrix by multiplyng each element thought a random value in the interval [alpha,1].
    Inputs:
        alpha : higher than zero and less than one, for example if alpha=0.8 the hoppings will be modified in the interval [0.8,1].
    Outputs:
        A: Ell matrix with elements modified.
    """

    if velocities is not None:
        print('Be carefull if the dx of the ELL matrix are the velocities and no tthe distance they need ot be modified.')

    data_new=np.copy(A.data)

    rng = np.random.default_rng(seed)  # Create a random generator with optional seed
    
    r_vec = 1-(1-alpha)*rng.random(len(A.data))+0j


    A_csr=ell_to_csr(A)
    
    vec_mul(0.5+0j, A_csr.data,r_vec,data_new,len(A.data),A.n_threads)
    
    A_csr.data=data_new

    A_csr=(((A_csr.T).conjugate())+A_csr)
    

    return A_csr.data

def modifier_random_hoppings_2(A,alpha=None,seed= None,velocities=None):
    """
    Def:
        Modifies the matrix element of a matrix by multiplyng each element thought a random value in the interval [1+alpha,1-alpha].
    Inputs:
        alpha : higher than zero and less than one, for example if alpha=0.8 the hoppings will be modified in the interval [1.8,0.2].
    Outputs:
        A: Ell matrix with elements modified.
    """

    if velocities is not None:
        print('Be carefull if the dx of the ELL matrix are the velocities and no tthe distance they need ot be modified.')

    data_new = np.copy(A.data)

    rng = np.random.default_rng(seed)

    r_vec = (1 - alpha) + 2 * alpha * rng.random(len(A.data)) +0j

    A_csr=ell_to_csr(A)
    
    vec_mul(0.5+0j, A_csr.data,r_vec,data_new,len(A.data),A.n_threads)
    
    A_csr.data=data_new

    A_csr=(((A_csr.T).conjugate())+A_csr)
    

    return A_csr.data