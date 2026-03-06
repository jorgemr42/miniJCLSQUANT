from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport openmp

from cython.parallel import prange

# Function to multiply a scalar by each element in the array (using C-style array) better input a new array with OpenMP
# Splitting into cores

cdef extern from "omp.h":
    void omp_set_num_threads(int num_threads)
    int omp_get_thread_num()
    void omp_set_affinity(int cpu_id)


cdef void matrix_vector_c(Py_ssize_t size,complex[:] data,int[:] indices, Py_ssize_t len_row,complex[:] v, complex[:] v2,int[:] starts, int[:] ends,int[:] row_starts, Py_ssize_t num_threads_user):
    
    cdef int i
    cdef int j


    cdef Py_ssize_t size_core=size//num_threads_user
    cdef int idx 
    cdef int start 
    cdef int end 
    cdef int row_start
    cdef complex sum_val

    for i in prange(num_threads_user,nogil=True,schedule='static',num_threads=num_threads_user,chunksize=1):
        start=starts[i]
        end=ends[i]
        for idx in range(start,end):
            sum_val=0.0+0.0j
            for j in range(len_row):
                sum_val =sum_val+ data[row_starts[idx] + j] * v[indices[row_starts[idx] + j]]
            v2[idx]=sum_val
    return 


def matrix_vector(data, indices, len_row, v,v2, starts, ends,row_starts,num_threads):
    return matrix_vector_c(v.shape[0],data, indices, len_row, v,v2, starts, ends,row_starts,num_threads)



def matrix_vector_2(data, indices, len_row, v, starts, ends,row_starts,num_threads):
    v2=np.zeros(len(v),dtype=complex,order='C')
    matrix_vector_c(v.shape[0],data, indices, len_row, v,v2, starts, ends,row_starts,num_threads)
    return v2 


cdef void matrix_vector_c_diagonal(Py_ssize_t size,complex[:] data,complex[:] diagonal,int[:] indices, Py_ssize_t len_row,complex[:] v, complex[:] v2,int[:] starts, int[:] ends,int[:] row_starts, Py_ssize_t num_threads_user):
    cdef int i
    cdef int j


    cdef Py_ssize_t size_core=size//num_threads_user
    cdef int idx 
    cdef int start 
    cdef int end 
    cdef int row_start
    cdef complex sum_val

    for i in prange(num_threads_user,nogil=True,schedule='static',num_threads=num_threads_user,chunksize=1):
        start=starts[i]
        end=ends[i]
        for idx in range(start,end):
            v2[idx]=v2[idx]+v[idx]*diagonal[idx]


def matrix_vector_diagonal(data,diagonal, indices, len_row, v,v2, starts, ends,row_starts,num_threads):
    if diagonal is None:
        return matrix_vector_c(v.shape[0],data, indices, len_row, v,v2, starts, ends,row_starts,num_threads)
    else:
        matrix_vector_c(v.shape[0],data, indices, len_row, v,v2, starts, ends,row_starts,num_threads)
        return matrix_vector_c_diagonal(v.shape[0],data, diagonal,indices, len_row, v,v2, starts, ends,row_starts,num_threads)


## Optimization to the core architecture


cdef void matrix_vector_c_3(Py_ssize_t len_data,complex[:] data,int[:] indices, Py_ssize_t len_row,complex[:] v, complex[:] v2):
    
    cdef int i
    cdef int j
    cdef int idx 
    cdef Py_ssize_t len_columns=len_data//len_row    


    for i in prange(len_columns,nogil='True',schedule='static'):
        for j in range(len_row):
            idx=i*len_row+j    
            v2[i]+=data[idx]*v[indices[idx]]
    
    return 

def matrix_vector_3(data, indices, len_row, v,v2):
    matrix_vector_c_3(data.shape[0],data, indices, len_row, v,v2)
    return 

def matrix_vector_4(data, indices, len_row, v):
    v2=np.zeros(len(v),dtype=complex,order='C')
    matrix_vector_c_3(data.shape[0],data, indices, len_row, v,v2)
    return v2 
