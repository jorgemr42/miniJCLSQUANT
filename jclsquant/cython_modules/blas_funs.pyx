#include <complex>

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport openmp
cimport cython
from cython.parallel import prange


### The scal function y=ay
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void scal_c(double complex a, double complex* x, int size,int n_threads):
    cdef int i
    for i in prange(size,nogil=True,schedule='static',num_threads=n_threads):
        x[i]=a*x[i]
    return 
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def scal(double complex a, double complex[::1] x, int size,int n_threads):
    scal_c(a,&x[0],size,n_threads)
    return


### The y=a*x+y function where a is a complex float and y and x are complex vectors
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void axpy_c(double complex a, double complex* x,double complex* y, int size,int n_threads):
    cdef int i
    for i in prange(size,nogil=True,schedule='static',num_threads=n_threads):
        y[i]=a*x[i]+y[i]
    return 
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def axpy(double complex a, double complex[::1] x,double complex[::1] y, int size,int n_threads):
    axpy_c(a,&x[0], &y[0],size,n_threads)
    return

### The y=a*x+b*y function where a is a complex float and y and x are complex vectors
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void  axpy2_c(double complex a,double complex b, double complex* x,double complex* y, int size,int n_threads):
    cdef int i
    for i in prange(size,nogil=True,schedule='static',num_threads=n_threads):
        y[i]=a*x[i]+b*y[i]
    return 
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def axpy2(double complex a,double complex b, double complex[::1] x,double complex[::1] y, int size,int n_threads):
    axpy2_c(a,b,&x[0], &y[0],size,n_threads)
    return

### The y=alpha*A*x+beta*y function where alpha and beta are complex floats, y and x are complex vectors and A is a matrix in ELL
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void aAxby_c(double complex a,double complex b, double complex* x,double complex* y, double complex* data,int* indices,int len_row,int num_columns,int n_threads):
    cdef int i,j,idx
    for i in prange(num_columns,nogil=True,schedule='static',num_threads=n_threads):
        y[i]*=b
        for j in range(len_row):
            idx=i*len_row+j    
            y[i]+=a*data[idx]*x[indices[idx]]                                                                                                                                                
    return 

def aAxby(double complex a,double complex b, double complex[::1] x,double complex[::1] y, double complex[::1] data, int[::1] indices,int len_row,int num_columns,int n_threads):
    aAxby_c(a , b , &x[0] , &y[0] , &data[0] , &indices[0] ,len_row,num_columns,n_threads)
    return



### The vdot product between two vectors,c=<x|a*y>
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double complex vdot_c(double complex a, double complex* x,double complex* y,int size,int n_threads):
    cdef int i
    cdef double complex sum =  0
    for i in prange(size,nogil=True,schedule='static',num_threads=n_threads):
        sum+=y[i]*a*x[i].conjugate()                                                                                                                                       
    return sum

def vdot(double complex a, double complex[::1] x,double complex[::1] y,int size,int n_threads):
    return vdot_c(a  , &x[0] , &y[0] , size,n_threads)

### The vdot product between two vectors,c=<x|a*y>
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double complex dot_c(double complex a, double complex* x,double complex* y,int size,int n_threads):
    cdef int i
    cdef double complex sum = 0
    for i in prange(size,nogil=True,schedule='static',num_threads=n_threads):
        sum+=y[i]*a*x[i]                                                                                                                                       
    return sum

def dot(double complex a, double complex[::1] x,double complex[::1] y,int size,int n_threads):
    return dot_c(a  , &x[0] , &y[0] , size,n_threads)



### Vector vector multiplication of the form z= a*y*x
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void vec_mul_c(double complex a, double complex* x,double complex* y,double complex* z,int size,int n_threads):
    cdef int i
    for i in prange(size,nogil=True,schedule='static',num_threads=n_threads):
        z[i]=a*y[i]*x[i]                                                                                                                                       
    return 

def vec_mul(double complex a, double complex[::1] x,double complex[::1] y,double complex[::1] z,int size,int n_threads):
    vec_mul_c(a  , &x[0] , &y[0] ,&z[0], size,n_threads)
    return


### Vector vector multiplication of the form y= a*x*y
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void vec_mul_c2(double complex a, double complex* x,double complex* y,int size,int n_threads):
    cdef int i
    for i in prange(size,nogil=True,schedule='static',num_threads=n_threads):
        y[i]=a*x[i]*y[i]                                                                                                                                       
    return 

def vec_mul2(double complex a, double complex[::1] x,double complex[::1] y,int size,int n_threads):
    vec_mul_c2(a  , &x[0] , &y[0] , size,n_threads)
    return




### Three vector multiplication of the form z= a*y*x*w
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void vec_mul3_c(double complex a, double complex* x,double complex* y,double complex* w,double complex* z,int size,int n_threads):
    cdef int i
    for i in prange(size,nogil=True,schedule='static',num_threads=n_threads):
        z[i]=a*y[i]*x[i]*w[i]                                                                                                                                       
    return 

def vec_mul3(double complex a, double complex[::1] x,double complex[::1] y,double complex[::1] w,double complex[::1] z,int size,int n_threads):
    vec_mul3_c(a  , &x[0] , &y[0] , &w[0] , &z[0] , size,n_threads)
    return

