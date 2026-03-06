from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport openmp
cimport cython

from cython.parallel import prange

cdef extern from "omp.h":
    void omp_set_num_threads(int num_threads)
    int omp_get_thread_num()
    void omp_set_affinity(int cpu_id)


## Optimization to the core architecture
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void matrix_vector_c(int len_data,double complex[:] data,long[:] indices, int len_row,double complex[:] v,double complex[:] v2,int n_threads):
    cdef Py_ssize_t i, j, idx
    cdef int num_columns=int(len_data/len_row)  
    
    for i in prange(num_columns,nogil='True',schedule='static',num_threads=n_threads):
        for j in range(len_row):
            idx=i*len_row+j    
            v2[i]+=data[idx]*v[indices[idx]]

    
    return 

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void matrix_vector_c_diagonal(int len_data,double complex[:] data,double complex[:] diagonal,long[:] indices, int len_row,double complex[:] v,double complex[:] v2,int n_threads):
    
    cdef Py_ssize_t i, j, idx
    cdef int num_columns=(len_data/len_row)    

    for i in prange(num_columns,nogil='True',schedule='static',num_threads=n_threads):
        for j in range(len_row):
            idx=i*len_row+j    
            v2[i]+=data[idx]*v[indices[idx]]
        v2[i]+=diagonal[i]*v[i]
    
    return 



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def matrix_vector(data,diagonal, indices, len_row, v,v2,n_threads):
    if diagonal is None:
        matrix_vector_c(data.shape[0],data, indices, len_row, v,v2,n_threads)
    else:
        matrix_vector_c_diagonal(data.shape[0],data, diagonal,indices, len_row, v,v2,n_threads)
    return v2 



@cython.wraparound(False)
@cython.boundscheck(False)
def matrix_vector_2(int len_data, double complex[:] data,double complex[:] diagonal,long[:] indices, int len_row,double complex[:] v,double complex[:] v2,int n_threads):
    cdef Py_ssize_t i, j,idx
    cdef int num_columns = int(len_data/len_row)  
    if diagonal is None:
        for i in prange(num_columns,nogil='True',schedule='static',num_threads=n_threads):
            for j in range(len_row):
                idx=i*len_row+j    
                v2[i]+=data[idx]*v[indices[idx]]
    else:
        for i in prange(num_columns,nogil='True',schedule='static',num_threads=n_threads):
            for j in range(len_row):
                idx=i*len_row+j    
                v2[i]+=data[idx]*v[indices[idx]]
            v2[i]+=diagonal[i]*v[i]
    return v2 


