#include <complex>

from libc.stdlib cimport malloc, free
import numpy as np
import json as json
import scipy as sci
cimport numpy as np
cimport openmp
cimport cython
from cython.parallel import prange
from libc.math cimport exp,sqrt,M_PI,sin,acos,cos  # Importing the C exp function
from openmp cimport omp_get_thread_num
from tqdm import tqdm

## Function to initialize the ELL matrices 
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.cdivision(True)
# def data_indices_c(
#     int len_row, int A_shape, 
#     int[::1] A_indptr, 
#     double complex[::1] A_data, int[::1] A_indices, 
#     double complex[::1] dx_data, double complex[::1] dy_data, 
#     int[::1] conjugates_data, 
#     int[::1] indices, double complex[::1] data, 
#     double complex[::1] data_dx, double complex[::1] data_dy, 
#     int[::1] data_conjugates, 
#     int n_threads
# ):
#     cdef int i, j, k, start, end
#     cdef int remaining
#     cdef double complex *row_i_data
#     cdef double complex *row_i_data_dx
#     cdef double complex *row_i_data_dy
#     cdef int *row_i_indices
#     cdef int *row_i_data_conjugates

#     # Parallel loop
#     for i in prange(A_shape, nogil=True, schedule='static', num_threads=n_threads):
#         start = A_indptr[i]
#         end = A_indptr[i+1]

#         row_i_data = &A_data[start]  # Pointer to slice start
#         row_i_data_dx = &dx_data[start]
#         row_i_data_dy = &dy_data[start]
#         row_i_data_conjugates = &conjugates_data[start]
#         row_i_indices = &A_indices[start]

#         remaining = end - start  # Length of the current row slice

#         # Fill output arrays
#         for k in range(remaining):
#             indices[i * len_row + k] = row_i_indices[k]
#             data[i * len_row + k] = row_i_data[k]
#             data_dx[i * len_row + k] = row_i_data_dx[k]
#             data_dy[i * len_row + k] = row_i_data_dy[k]
#             data_conjugates[i * len_row + k] = row_i_data_conjugates[k]

#         # Fill remaining part with zeros
#         for k in range(remaining, len_row):
#             indices[i * len_row + k] = row_i_indices[remaining - 1]  # Repeat last value
#             data[i * len_row + k] = 0
#             data_dx[i * len_row + k] = 0
#             data_dy[i * len_row + k] = 0
#             data_conjugates[i * len_row + k] = 0

#     return np.asarray(data), np.asarray(indices), np.asarray(data_dx), np.asarray(data_dy), np.asarray(data_conjugates)



#ifndef ELL_UTILS_HPP
#define ELL_UTILS_HPP

#include <limits>
#include <cstdint>
from libc.stdint cimport SIZE_MAX

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def data_indices_c(
    int len_row, int A_shape, 
    int[::1] A_indptr, 
    double complex[::1] A_data, 
    int[::1] A_indices, 
    double complex[::1] dx_data,
    double complex[::1] dy_data, 
    int[::1] conjugates_data, 
    int[::1] indices,
    double complex[::1] data, 
    double complex[::1] data_dx,
    double complex[::1] data_dy, 
    int[::1] data_conjugates, 
    int[::1] valid_row_values, 
    int n_threads,
):
    cdef int i, j, k, start, end
    cdef int remaining
    cdef double complex *row_i_data
    cdef double complex *row_i_data_dx
    cdef double complex *row_i_data_dy
    cdef int *row_i_indices
    cdef int *row_i_data_conjugates
    cdef size_t INVALID_INDEX = SIZE_MAX

    # Parallel loop
    for i in prange(A_shape, nogil=True, schedule='static', num_threads=n_threads):
        start = A_indptr[i]
        end = A_indptr[i+1]

        row_i_data = &A_data[start]  # Pointer to slice start
        row_i_data_dx = &dx_data[start]
        row_i_data_dy = &dy_data[start]
        row_i_data_conjugates = &conjugates_data[start]
        row_i_indices = &A_indices[start]

        remaining = end - start  # Length of the current row slice

        # Fill output arrays
        for k in range(remaining):
            indices[i * len_row + k] = row_i_indices[k]
            data[i * len_row + k] = row_i_data[k]
            data_dx[i * len_row + k] = row_i_data_dx[k]
            data_dy[i * len_row + k] = row_i_data_dy[k]
            data_conjugates[i * len_row + k] = row_i_data_conjugates[k]
            valid_row_values[i * len_row + k] = 1

        # Fill remaining part with zeros
        for k in range(remaining, len_row):
            indices[i * len_row + k] = row_i_indices[remaining - 1]  # Repeat last value
            data[i * len_row + k] = 0
            data_dx[i * len_row + k] = 0
            data_dy[i * len_row + k] = 0
            data_conjugates[i * len_row + k] = 0
            valid_row_values[i * len_row + k] = 0

    return np.asarray(data), np.asarray(indices), np.asarray(data_dx), np.asarray(data_dy), np.asarray(data_conjugates),np.asarray(valid_row_values)


### Function to initialize the ELL matrices 
# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.cdivision(True)
# def data_indices_c_0(
#     int len_row, int A_shape, 
#     int[::1] A_indptr, 
#     double complex[::1] A_data, int[::1] A_indices, 
#     int[::1] indices, double complex[::1] data, 
#     int n_threads
# ):
#     cdef int i, j, k, start, end
#     cdef int remaining
#     cdef double complex *row_i_data
#     cdef int *row_i_indices

#     # Parallel loop
#     for i in prange(A_shape, nogil=True, schedule='static', num_threads=n_threads):
#         start = A_indptr[i]
#         end = A_indptr[i+1]

#         row_i_data = &A_data[start]  # Pointer to slice start
#         row_i_indices = &A_indices[start]

#         remaining = end - start  # Length of the current row slice

#         # Fill output arrays
#         for k in range(remaining):
#             indices[i * len_row + k] = row_i_indices[k]
#             data[i * len_row + k] = row_i_data[k]

#         # Fill remaining part with zeros
#         for k in range(remaining, len_row):
#             indices[i * len_row + k] = row_i_indices[remaining - 1]  # Repeat last value
#             data[i * len_row + k] = 0

#     return np.asarray(data), np.asarray(indices)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def data_indices_c_0(
    int len_row, int A_shape, 
    int[::1] A_indptr, 
    double complex[::1] A_data, 
    int[::1] A_indices, 
    int[::1] indices, 
    double complex[::1] data, 
    int[::1] valid_row_values, 
    int n_threads
):
    cdef int i, j, k, start, end
    cdef int remaining
    cdef double complex *row_i_data
    cdef int *row_i_indices

    # Parallel loop
    for i in prange(A_shape, nogil=True, schedule='static', num_threads=n_threads):
        start = A_indptr[i]
        end = A_indptr[i+1]

        row_i_data = &A_data[start]  # Pointer to slice start
        row_i_indices = &A_indices[start]

        remaining = end - start  # Length of the current row slice

        # Fill output arrays
        for k in range(remaining):
            indices[i * len_row + k] = row_i_indices[k]
            data[i * len_row + k] = row_i_data[k]
            valid_row_values[i * len_row + k] = 1

        # Fill remaining part with zeros
        for k in range(remaining, len_row):
            indices[i * len_row + k] = row_i_indices[remaining - 1]  # Repeat last value
            data[i * len_row + k] = 0
            valid_row_values[i * len_row + k] = 0

    return np.asarray(data), np.asarray(indices),np.asarray(valid_row_values)


cdef extern from "complex.h":
    double complex cexp(double complex z) nogil
    double complex ccos(double complex z) nogil
    double complex csin(double complex z) nogil
    double complex ccosh(double complex z) nogil
    double complex csqrt(double complex z) nogil
    double complex I 



### Light modifier (non-packed)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void light_modifier_circle_c(double complex a,
                                      double complex A0,
                                      double complex t,
                                      double complex w,
                                      double complex Tp,
                                      double complex eta,
                                      double complex* data,
                                      double complex* dx_vec,
                                      double complex* dy_vec,
                                      double complex* data_new,
                                      int size,
                                      int n_threads) :
    cdef int i
    cdef double complex temp1, temp2, temp3, temp4, scale
    for i in prange(size, nogil=True, schedule='static', num_threads=n_threads):
        data_new[i] = a * data[i] * cexp(I * A0 * (ccos(w*t) * dx_vec[i] + eta * csin(w*t) * dy_vec[i]))
    return
# Linear
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void light_modifier_linear_c(double complex a,
                                      double complex A0,
                                      double complex t,
                                      double complex w,
                                      double complex Tp,
                                      double complex eta,
                                      double complex* data,
                                      double complex* dx_vec,
                                      double complex* dy_vec,
                                      double complex* data_new,
                                      int size,
                                      int n_threads) :
    cdef int i
    cdef double complex temp1, temp2, temp3, temp4, scale
    for i in prange(size, nogil=True, schedule='static', num_threads=n_threads):
        data_new[i] = a * data[i] * cexp(I * A0 * (csin(w*t) * dy_vec[i]))
    return

# Linear
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void light_modifier_linear_packed_c(double complex a,
                                      double complex A0,
                                      double complex t,
                                      double complex w,
                                      double complex Tp,
                                      double complex eta,
                                      double complex* data,
                                      double complex* dx_vec,
                                      double complex* dy_vec,
                                      double complex* data_new,
                                      int size,
                                      int n_threads) :
    cdef int i
    cdef double complex temp1, temp2, temp3, temp4, scale
    for i in prange(size, nogil=True, schedule='static', num_threads=n_threads):
        data_new[i] = a * data[i] * cexp(I * A0 * (csin(w*t) * dy_vec[i])/ ccosh((t - 2*Tp) / (0.5673*Tp)))
    return

### Light modifier (packed version)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void light_modifier_circle_packed_c(double complex a,
                                      double complex A0,
                                      double complex t,
                                      double complex w,
                                      double complex Tp,
                                      double complex eta,
                                      double complex* data,
                                      double complex* dx_vec,
                                      double complex* dy_vec,
                                      double complex* data_new,
                                      int size,
                                      int n_threads) :
    cdef int i
    cdef double complex temp1, temp2, temp3, temp4, scale
    for i in prange(size, nogil=True, schedule='static', num_threads=n_threads):
        data_new[i] = a * data[i] * cexp(I * A0 * (ccos(w*t) * dx_vec[i] + eta * csin(w*t) * dy_vec[i])/ ccosh((t - 2*Tp) / (0.5673*Tp)))
    return

def light_modifier_cython(double complex a,
                          double complex A0,
                          double complex t,
                          double complex w,
                          double complex Tp,
                          double complex eta,
                          double complex[::1] data,
                          double complex[::1] dx_vec,
                          double complex[::1] dy_vec,
                          double complex[::1] data_new,
                          int size,
                          int n_threads):
    if Tp == 0:
        if eta ==0:
            light_modifier_linear_c(a, A0, t, w, Tp, eta, &data[0], &dx_vec[0], &dy_vec[0], &data_new[0], size, n_threads)
        else:
            light_modifier_circle_c(a, A0, t, w,Tp, eta, &data[0], &dx_vec[0], &dy_vec[0], &data_new[0], size, n_threads)
            
    else:
        if eta ==0:
            light_modifier_linear_packed_c(a, A0, t, w, Tp, eta, &data[0], &dx_vec[0], &dy_vec[0], &data_new[0], size, n_threads)
        else:
            light_modifier_circle_packed_c(a, A0, t, w,Tp, eta, &data[0], &dx_vec[0], &dy_vec[0], &data_new[0], size, n_threads)
    return




#################################################
### Modifies the diagonal elements of a matrix
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void diagonal_modifier_c(double complex* data,int* indices,int* valid_row_values, double complex* data_new, double complex* diagonal_array , int len_row , int len_data, int n_threads):
    cdef int i,n_row

    for i in prange(len_data,nogil=True,schedule='static',num_threads=n_threads):
        n_row = i/len_row   
        if n_row == indices[i]  and valid_row_values[i]==1:
            data_new[i]=diagonal_array[n_row]+data[i]                                                                                                                                         
        else:
            data_new[i]=data[i]
        
    return 

def diagonal_modifier( double complex[::1] data,int[::1] indices,int[::1] valid_row_values, double complex[::1] data_new, double complex[::1] diagonal_array , int len_row,int len_data,int n_threads):
    diagonal_modifier_c( &data[0] , &indices[0] , &valid_row_values[0]  ,&data_new[0]  ,&diagonal_array[0],len_row,len_data,n_threads)
    return


############################################################3
### Modifies the diagonal elements of a matrix
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void electron_hole_pud_c(double* diagonal_array,int len_diagonal,int* r, double* Sx, double* Sy ,int* random_sites , int len_random_sites,double W, double g,int n_threads):
    cdef int i, j, tid
    cdef int num_threads = n_threads
    cdef double[:, :] local_arrays = np.zeros((num_threads, len_diagonal), dtype=np.float64)

    # Parallel loop — each thread writes to its own row
    for i in prange(len_random_sites, nogil=True, schedule='static', num_threads=n_threads):
        tid = omp_get_thread_num()
        for j in range(len_diagonal):
            local_arrays[tid, j] += (-1)**(r[i]) * W * exp(-((Sx[random_sites[i]] - Sx[j])**2 +
                                                             (Sy[random_sites[i]] - Sy[j])**2) / (2 * g**2))

    # Serial reduction step
    for i in range(num_threads):
        for j in range(len_diagonal):
            diagonal_array[j] += local_arrays[i, j]
    return 

def electron_hole_pud( double[::1] diagonal_array,int len_diagonal,int[::1] r, double[::1] Sx, double[::1] Sy ,int[::1] random_sites , int len_random_sites,double W, double g,int n_threads):
    electron_hole_pud_c(&diagonal_array[0],len_diagonal,&r[0],&Sx[0],&Sy[0] ,&random_sites[0],len_random_sites,W,g,n_threads)
    return

### Modifies the diagonal elements of a matrix
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void electron_hole_pud_3d_c(double* diagonal_array,int len_diagonal,int* r, double* Sx, double* Sy ,double* Sz,int* random_sites , int len_random_sites,double W, double g,int n_threads):
    cdef int i, j, tid
    cdef int num_threads = n_threads
    cdef double[:, :] local_arrays = np.zeros((num_threads, len_diagonal), dtype=np.float64)

    # Parallel loop — each thread writes to its own row
    for i in prange(len_random_sites, nogil=True, schedule='static', num_threads=n_threads):
        tid = omp_get_thread_num()
        for j in range(len_diagonal):
            local_arrays[tid, j] += (-1)**(r[i]) * W * exp(-((Sx[random_sites[i]] - Sx[j])**2 +
                                                             (Sy[random_sites[i]] - Sy[j])**2 +
                                                             (Sz[random_sites[i]] - Sz[j])**2) / (2 * g**2))

    # Serial reduction step
    for i in range(num_threads):
        for j in range(len_diagonal):
            diagonal_array[j] += local_arrays[i, j]

    return 

def electron_hole_pud3d( double[::1] diagonal_array,int len_diagonal,int[::1] r, double[::1] Sx, double[::1] Sy, double[::1] Sz ,int[::1] random_sites , int len_random_sites,double W, double g,int n_threads):
    electron_hole_pud_3d_c(&diagonal_array[0],len_diagonal,&r[0],&Sx[0],&Sy[0],&Sz[0] ,&random_sites[0],len_random_sites,W,g,n_threads)

    return


######################Construction of the double table for r
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void moments_Gmin_real_c(double Ef, int N, double* out) noexcept nogil:
    cdef int i
    cdef double factor = sqrt(1.0 - Ef*Ef)
    if factor == 0:
        factor = 1e-16  # avoid division by zero
    
    out[0] = 0.0
    for i in range(1, N):
        out[i] = -2.0 * sin(i * acos(Ef)) / factor

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void moments_Gmas_real_c(double Ef, int N, double* out) noexcept nogil:
    cdef int i
    cdef double factor = sqrt(1.0 - Ef*Ef)
    if factor == 0:
        factor = 1e-16  # avoid division by zero
    
    out[0] = 0.0
    for i in range(1, N):
        out[i] = 2.0 * sin(-i * acos(Ef)) / factor


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def JacksonKernel_c(N):  #It gives all the element of the Kernel for a given Number of moments (N)
    n=np.arange(0,N)
    g=(N-n+1)*np.cos(np.pi*n/(N+1))+np.sin(np.pi*n/(N+1))*(1/np.tan(np.pi/(N+1)))
    g=g/(N+1)
    #g[0]=g[0]/2  #not needed <- LSQUANT includes this in the moments already
    return g
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void moments_delta_c(double Ef, int N, double* out) noexcept nogil:
    cdef int i
    cdef double factor = M_PI * sqrt(1.0 - Ef*Ef)
    if factor == 0:
        factor = 1e-16  # avoid division by zero

    out[0] = 1.0 / factor
    for i in range(1, N):
        out[i] = 2.0 * cos(i * acos(Ef)) / factor

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef table_G_delta_c(int M, double alpha_integral,double H_kpm_bounds, int n_threads):
    cdef double[:,:,:] moments_table_bigmas = np.ascontiguousarray(np.zeros((M, M, 2*M), dtype=np.float64))
    cdef double[:,:,:] moments_table_bigmin = np.ascontiguousarray(np.zeros((M, M, 2*M), dtype=np.float64))


    cdef double[::1] JacksonKernel_vec=JacksonKernel_c(M)
    cdef double[::1] E_vec=np.linspace(-alpha_integral,alpha_integral,2*M)


    cdef int i, m, n
    cdef int M2 = 2*M
    cdef double e

    # Define the pointers
    cdef double* Jackson_ptr=&JacksonKernel_vec[0]
    cdef double* E_vec_ptr=&E_vec[0]

    # Preallocate temporary vectors
    cdef double* moments_Gmin_vec
    cdef double* moments_Gmas_vec
    cdef double* moments_delta_vec

    for i in prange(M2, nogil=True, schedule='static', num_threads=n_threads):
        # allocate scratch arrays per thread/iteration
        moments_Gmin_vec = <double*> malloc(M * sizeof(double))
        moments_Gmas_vec = <double*> malloc(M * sizeof(double))
        moments_delta_vec = <double*> malloc(M * sizeof(double))

        if not moments_Gmin_vec or not moments_Gmas_vec or not moments_delta_vec:
            with gil:
                raise MemoryError()

        e = E_vec_ptr[i]

        # use C helper functions that take (double e, int M, double* out)
        moments_Gmin_real_c(e, M, moments_Gmin_vec)
        moments_Gmas_real_c(e, M, moments_Gmas_vec)
        moments_delta_c(e, M, moments_delta_vec)


        # Apply Jackson kernel and H_kpm_bounds scaling
        for n in range(M):
            
            moments_Gmin_vec[n] = moments_Gmin_vec[n] *Jackson_ptr[n] / H_kpm_bounds
            moments_Gmas_vec[n] = moments_Gmas_vec[n] * Jackson_ptr[n] / H_kpm_bounds
            moments_delta_vec[n] = moments_delta_vec[n]* Jackson_ptr[n]
   
        # Fill the 3D tables
        for n in range(M):
            for m in range(M):
                moments_table_bigmas[m, n, i] =  moments_Gmas_vec[m] * moments_delta_vec[n]
                moments_table_bigmin[m, n, i] = -moments_delta_vec[m] * moments_Gmin_vec[n]

    return  np.array(moments_table_bigmas),np.array(moments_table_bigmin)

def table_G_delta(int M,double alpha_integral,double H_kpm_bounds,int n_threads):
    return table_G_delta_c(M,alpha_integral,H_kpm_bounds,n_threads)


##################### Modifier of the hoppings according to their positions.

def hopping_pos_phonons_modifier(double complex[::1] data,int[::1] indices, double complex[::1] dx_vec, double complex[::1] dy_vec, double complex[::1] data_new, double complex[::1] dx_vec_new, double complex[::1] dy_vec_new,double complex a0,double complex b, double complex Aq, double complex[::1] e_pol_x, double complex[::1] e_pol_y, double complex wq, double complex t, double complex[::1] Sx,double complex[::1] Sy,double complex [::1] q,int len_row,int size,int n_threads):
    return hopping_pos_phonons_modifier_c(&data[0],&indices[0],&dx_vec[0],&dy_vec[0],&data_new[0],&dx_vec_new[0],&dy_vec_new[0],a0,b,Aq,&e_pol_x[0],&e_pol_y[0],wq,t,&Sx[0],&Sy[0],&q[0],len_row,size,n_threads)



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef hopping_pos_phonons_modifier_c(double complex* data,int* indices,double complex* dx_vec,double complex* dy_vec,double complex* data_new,double complex* dx_vec_new,double complex* dy_vec_new,double complex a0,double complex b,double complex Aq,double complex* e_pol_x,double complex* e_pol_y,double complex wq,double complex t,double complex* Sx,double complex* Sy,double complex* q,int len_row,int size,int n_threads):
    cdef int i
    cdef double complex Tp=2*M_PI/wq
    ## The only things you have to change here is the modified function
    for i in prange(size, nogil=True, schedule='static', num_threads=n_threads):
        if i//len_row!=indices[i]:
            dx_vec_new[i]=dx_vec[i]+Aq*(e_pol_x[i//len_row]*ccos(wq*t-q[0]*Sx[i//len_row])-e_pol_x[indices[i]]*ccos(wq*t-q[0]*Sx[indices[i]]))/ ccosh((t - 2*Tp) / (0.5673*Tp))
            dy_vec_new[i]=dy_vec[i]+Aq*(e_pol_y[i//len_row]*ccos(wq*t-q[1]*Sy[i//len_row])-e_pol_y[indices[i]]*ccos(wq*t-q[1]*Sy[indices[i]]))/ ccosh((t - 2*Tp) / (0.5673*Tp))
            data_new[i] = data[i] * cexp(-b*(csqrt(abs(dx_vec_new[i])**2+abs(dy_vec_new[i])**2)/a0-1))

    return


 ################# Construction of the k Hamiltonian #########################

def Hk_loop(hop,sub,lat,label_to_index,label_to_index_orbitals,int block_shape,total_nnz,double[::1] k_points_f,int nk,int n_threads,double complex[::1] data,int[::1] columns,int[::1] rows):
    Hk_loop_c(hop,sub,lat,label_to_index,label_to_index_orbitals,block_shape,total_nnz,&k_points_f[0],nk,n_threads,&data[0],&columns[0],&rows[0])
    return 

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void Hk_loop_c(hop,sub,lat,label_to_index,label_to_index_orbitals,int block_shape,total_nnz,double* k_points_f,int nk,int n_threads,double complex* data,int* columns,int* rows):

    cdef int i,i3,index,h,Ncols,len_hop,iter
    cdef double complex[:] value_view 
    cdef double[:] delta_view
    cdef np.ndarray[np.complex128_t, ndim=1] value_array

    cdef complex phase

    cdef long[:] label_to_index_1_view
    cdef long[:] label_to_index_2_view
    iter=0
    for h in tqdm(range(len(hop['Hoppings']))):
        label1=hop['Hoppings'][h]['From']
        label2=hop['Hoppings'][h]['To']
        rel=hop['Hoppings'][h]['Rel_Idx']
        len_hop=len(hop['Hoppings'][h]['ValsReal'])

        # Allocate a new array for this hopping
        value_array = np.empty(len_hop, dtype=np.complex128)
        value_view = value_array  # memory view
        
        for i in range(len_hop):
            value_view[i] = hop['Hoppings'][h]['ValsReal'][i] + 1j*hop['Hoppings'][h]['ValsImag'][i]
        

        delta_view = np.array(sub['Sublattices'][label_to_index[label1]]['Position'])-(np.array(sub['Sublattices'][label_to_index[label2]]['Position'])+(rel[0]*np.array(lat['Vec1'])+rel[1]*np.array(lat['Vec2'])+rel[2]*np.array(lat['Vec3'])))
        



        label_to_index_1_view=np.array(label_to_index_orbitals[label1])
        label_to_index_2_view=np.array(label_to_index_orbitals[label2])

        Ncols=hop['Hoppings'][h]['Ncols']

        for i in prange(nk, nogil=True, schedule='static', num_threads=n_threads):
            for i3 in range(len_hop):
                index=iter+i*len_hop+i3
                data[index]=value_view[i3]*cexp(-1.0*I*(k_points_f[3*i]*delta_view[0]+k_points_f[3*i+1]*delta_view[1]+k_points_f[3*i+2]*delta_view[2]))
                rows[index]=label_to_index_1_view[i3//Ncols]+block_shape*i
                columns[index]=label_to_index_2_view[i3%Ncols]+block_shape*i
        iter+=nk*len(hop)

    return 



def Hk_loop2(hop,sub,lat,label_to_index,label_to_index_orbitals,int block_shape,Ncols,double[::1] k_points_f,int nk,int n_threads,double complex[::1] data,int[::1] columns,int[::1] rows):
    Hk_loop_c2(hop,sub,lat,label_to_index,label_to_index_orbitals,block_shape,Ncols,&k_points_f[0],nk,n_threads,&data[0],&columns[0],&rows[0])
    return 

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void Hk_loop_c2(hop,sub,lat,label_to_index,label_to_index_orbitals,int block_shape,int Ncols,double* k_points_f,int nk,int n_threads,double complex* data,int* columns,int* rows):

    cdef int i,i3,index,h,len_hop,iter,Ncols_h,row,column
    cdef double complex[:] value_view 
    cdef double[:] delta_view
    cdef np.ndarray[np.complex128_t, ndim=1] value_array

    cdef complex phase

    cdef long[:] label_to_index_1_view
    cdef long[:] label_to_index_2_view
    iter=0
    for h in tqdm(range(len(hop['Hoppings']))):
        label1=hop['Hoppings'][h]['From']
        label2=hop['Hoppings'][h]['To']
        rel=hop['Hoppings'][h]['Rel_Idx']
        len_hop=len(hop['Hoppings'][h]['ValsReal'])

        # Allocate a new array for this hopping
        value_array = np.empty(len_hop, dtype=np.complex128)
        value_view = value_array  # memory view

        Ncols_h=hop['Hoppings'][h]['Ncols']

        for i in range(len_hop):
            value_view[i] = hop['Hoppings'][h]['ValsReal'][i] + 1j*hop['Hoppings'][h]['ValsImag'][i]
        

        delta_view = np.array(sub['Sublattices'][label_to_index[label1]]['Position'])-(np.array(sub['Sublattices'][label_to_index[label2]]['Position'])+(rel[0]*np.array(lat['Vec1'])+rel[1]*np.array(lat['Vec2'])+rel[2]*np.array(lat['Vec3'])))
        



        label_to_index_1_view=np.array(label_to_index_orbitals[label1])
        label_to_index_2_view=np.array(label_to_index_orbitals[label2])

        Ncols_h=hop['Hoppings'][h]['Ncols']

        for i in prange(nk, nogil=True, schedule='static', num_threads=n_threads):
            for i3 in range(len_hop):
                
                row=label_to_index_1_view[i3//Ncols_h]
                column=label_to_index_2_view[i3%Ncols_h]+block_shape*i

                index=row*Ncols+column

                rows[index]=row+block_shape*i
                columns[index]=column

                data[index]+=value_view[i3]*cexp(-1.0*I*(k_points_f[3*i]*delta_view[0]+k_points_f[3*i+1]*delta_view[1]+k_points_f[3*i+2]*delta_view[2]))

    return 




def Hk_loopv(hop,sub,lat,label_to_index,label_to_index_orbitals,int block_shape,Ncols,double[::1] k_points_f,int nk,int n_threads,double complex[::1] data,double complex[::1] data_x,double complex[::1] data_y,double complex[::1] data_z,int[::1] columns,int[::1] rows):
    Hk_loop_cv(hop,sub,lat,label_to_index,label_to_index_orbitals,block_shape,Ncols,&k_points_f[0],nk,n_threads,&data[0],&data_x[0],&data_y[0],&data_z[0],&columns[0],&rows[0])
    return 

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void Hk_loop_cv(hop,sub,lat,label_to_index,label_to_index_orbitals,int block_shape,int Ncols,double* k_points_f,int nk,int n_threads,double complex* data,double complex* data_x,double complex* data_y,double complex* data_z,int* columns,int* rows):

    cdef int i,i3,index,h,len_hop,iter,Ncols_h,row,column
    cdef double complex[:] value_view 
    cdef double[:] delta_view
    cdef np.ndarray[np.complex128_t, ndim=1] value_array

    cdef complex phase

    cdef long[:] label_to_index_1_view
    cdef long[:] label_to_index_2_view
    iter=0
    for h in tqdm(range(len(hop['Hoppings']))):
        label1=hop['Hoppings'][h]['From']
        label2=hop['Hoppings'][h]['To']
        rel=hop['Hoppings'][h]['Rel_Idx']
        len_hop=len(hop['Hoppings'][h]['ValsReal'])

        # Allocate a new array for this hopping
        value_array = np.empty(len_hop, dtype=np.complex128)
        value_view = value_array  # memory view

        Ncols_h=hop['Hoppings'][h]['Ncols']

        for i in range(len_hop):
            value_view[i] = hop['Hoppings'][h]['ValsReal'][i] + 1j*hop['Hoppings'][h]['ValsImag'][i]
        

        delta_view = np.array(sub['Sublattices'][label_to_index[label1]]['Position'])-(np.array(sub['Sublattices'][label_to_index[label2]]['Position'])+(rel[0]*np.array(lat['Vec1'])+rel[1]*np.array(lat['Vec2'])+rel[2]*np.array(lat['Vec3'])))
        



        label_to_index_1_view=np.array(label_to_index_orbitals[label1])
        label_to_index_2_view=np.array(label_to_index_orbitals[label2])

        Ncols_h=hop['Hoppings'][h]['Ncols']

        for i in prange(nk, nogil=True, schedule='static', num_threads=n_threads):
            for i3 in range(len_hop):
                
                row=label_to_index_1_view[i3//Ncols_h]
                column=label_to_index_2_view[i3%Ncols_h]+block_shape*i

                index=row*Ncols+column

                rows[index]=row+block_shape*i
                columns[index]=column

                data[index]+=value_view[i3]*cexp(-1.0*I*(k_points_f[3*i]*delta_view[0]+k_points_f[3*i+1]*delta_view[1]+k_points_f[3*i+2]*delta_view[2]))
                
                data_x[index]+=-1.0*I*delta_view[0]*value_view[i3]*cexp(-1.0*I*(k_points_f[3*i]*delta_view[0]+k_points_f[3*i+1]*delta_view[1]+k_points_f[3*i+2]*delta_view[2]))
                data_y[index]+=-1.0*I*delta_view[1]*value_view[i3]*cexp(-1.0*I*(k_points_f[3*i]*delta_view[0]+k_points_f[3*i+1]*delta_view[1]+k_points_f[3*i+2]*delta_view[2]))
                data_z[index]+=-1.0*I*delta_view[2]*value_view[i3]*cexp(-1.0*I*(k_points_f[3*i]*delta_view[0]+k_points_f[3*i+1]*delta_view[1]+k_points_f[3*i+2]*delta_view[2]))

    return 