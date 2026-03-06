from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport openmp

from cython.parallel import prange

cdef extern from "omp.h":
    void omp_set_num_threads(int num_threads)
    int omp_get_thread_num()
    void omp_set_affinity(int cpu_id)


## Optimization to the core architecture
cdef void matrix_vector_c(Py_ssize_t len_data,complex[:] data,int[:] indices, Py_ssize_t len_row,complex[:] v, complex[:] v2):
    
    cdef int i
    cdef int j
    cdef int idx 
    cdef Py_ssize_t len_columns=len_data//len_row    


    for i in prange(len_columns,nogil='True',schedule='static'):
        for j in range(len_row):
            idx=i*len_row+j    
            v2[i]+=data[idx]*v[indices[idx]]
    
    return 


def matrix_vector(data, indices, len_row, v):
    v2=np.zeros(len(v),dtype=complex,order='C')
    matrix_vector_c(data.shape[0],data, indices, len_row, v,v2)
    return v2 
