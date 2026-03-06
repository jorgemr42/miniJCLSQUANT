import scipy as sci
import numpy as np
from time import perf_counter
import copy
from math import pi,sqrt
import joblib

from jclsquant.cython_modules.blas_funs import aAxby # type: ignore
from jclsquant.cython_modules.Extra_funs import data_indices_c,data_indices_c_0 # type: ignore
import os
from jclsquant.hams import *
    

### ELL to CSR matrix
def ell_to_csr(A):
    """
    Def:
        Transforms a matrix_ell to sci.sparse.csr_matrix 
    Inputs:
        A: matrix in ELL format.
    Outputs:
        A: matrix in sci.sparse.csr_matrix.
    """
    rows=np.zeros(len(A.data),dtype=int)
    cols=np.zeros(len(A.data),dtype=int)
    for i in range(len(A.data)):
        if A.indices[i]!=-1:
            rows[i]=i//A.len_row
            cols[i]=A.indices[i]
    B=sci.sparse.csr_matrix((A.data, (rows, cols)), shape=A.shape)
    return B




def data_indices_cython(len_row,A_shape,A_indptr,A_data,A_indices,A_dx,A_dy,A_conjugates,n_threads):
    indices=np.zeros(A_shape*len_row,dtype=np.int32)
    data=np.zeros(A_shape*len_row,dtype=np.complex128)
    data_dx=np.zeros(A_shape*len_row,dtype=np.complex128)
    data_dy=np.zeros(A_shape*len_row,dtype=np.complex128)
    data_conjugates=np.zeros(A_shape*len_row,dtype=np.int32)
    valid_row_values=np.zeros(A_shape*len_row,dtype=np.int32)
    
    A_dx_2=A_dx.astype(np.complex128)
    A_dy_2=A_dy.astype(np.complex128)
    A_conjugates_2=np.real(A_conjugates).astype(np.int32)
    return data_indices_c(len_row,A_shape,A_indptr,A_data,A_indices,A_dx_2,A_dy_2,A_conjugates_2,indices,data,data_dx,data_dy,data_conjugates,valid_row_values,n_threads)


def data_indices_cython_0(len_row,A_shape,A_indptr,A_data,A_indices,n_threads):
    indices=np.zeros(A_shape*len_row,dtype=np.int32)
    valid_row_values=np.zeros(A_shape*len_row,dtype=np.int32)
    data=np.zeros(A_shape*len_row,dtype=np.complex128)


    return data_indices_c_0(len_row,A_shape,A_indptr,A_data,A_indices,indices,data,valid_row_values,n_threads)







## Functions that are paraleize and that are used inside the class
def has_diagonal_elements(csr):
    # Get the row and column indices of non-zero elements
    rows, cols = csr.nonzero()
    
    # Check if any non-zero element is on the diagonal
    return any(rows[i] == cols[i] for i in range(len(rows)))

    
def sub_outside(data, other):
    return data - other

def add_outside(data_self,other):
    return data_self+other    
def mul_outside(data, other):
    return  data * other  
def div_outside(data, other):
    return data / other  

def Gershgorin(H):
    '''
    For hermitian H, extracts bounds for min and max eigenvalues using Gershgorin's Circle theorem.
    This function is not so well tested for sparse matrices. Use with care.
    '''
    centers=H.diagonal()  #extract the centers of the Gershgorin circles
    radii=np.sum(np.abs(H),axis=0)-np.abs(centers)  #compute their radii
    vmin=np.amin(centers-radii)
    vmax=np.amax(centers+radii)

    return np.array([vmin,vmax])


def bounds(H):
    """
    Def: 
        Gives symmetrical bounds for a Ell matrix
    Inputs:
        H: Ell matrix
    Outputs:
        bounds_array: arrray of [lower,upper] bound symmetric, so lower=-upper
    """
    if sci.sparse.isspmatrix_csr(H) is False:
        H=ell_to_csr(H)

    boundsGer=Gershgorin(H).real  #if H is hermitian, the bounds are real

    Bounds_safety_factor=0.025  #enlargment of the bounds by a small extra safety factor
    BoundsRange=boundsGer[1]-boundsGer[0]
    bounds_extra=BoundsRange*Bounds_safety_factor
    bounds_array=np.array([boundsGer[0]-bounds_extra, boundsGer[1]+bounds_extra])
    return np.array([-np.max(np.abs(bounds_array)),np.max(np.abs(bounds_array))])


# numba.set_num_threads(24)

class ell_matrix:
    """
    Def:
        Matrix ELL class with impemented matrix vector multiplication in Cython.
    Atributes:
        type: type of the matrix 
        diagonal: array of the diagonal element of the matrix
        data: elements of the matrix. Ordered in such way that from 0 to len_row there are all 
            the elements of 0 row and the different columns label by indices.
        len_row: Maximum number of non zero elements in any row.
        indices: column of each element in data
        shape: shape of the matrix, not the number of elements, in a tuple (N,N) 
    """
    
    def __init__(self, A , DX = None , DY = None , CJ = None,velocities='False',space='r'):
        self.A = A
        self.type = None
        self.len_row = None
        self.len_col = None

        self.ell_data = None      
        self.ell_indices = None
        self.shape=None
        self.n_threads=None
        # Hamiltonian params
        self.DX=DX
        self.DY=DY
        self.CJ=CJ

        self.dx_vec=None
        self.dy_vec=None
        self.dz_vec=None
        self.Omega=None
        self.conjugates=None
        self.valid_row_values=None
        if DX is not None and DY is not None:
            if len(DX.data) != len(A.data) or len(DY.data) != len(A.data):
                print('BE CAREFULL THE NNZ OF THE DX OR DY AND H ARE NOT THE SAME') 

        # Funcitons that automatically run when started
        self.set_n_threads()
        self.check_format()

        self.bounds=bounds(self.A)
        self.space=space
        self.velocities=velocities
        

        
    def check_format(self):
        if sci.sparse.isspmatrix_csr(self.A):
            self.csr_to_ell()
            self.type='ELLPACK(ELL) sparse matrix format'

        elif self.A.type=='ELLPACK(ELL) sparse matrix format':
            print('Matrix is already in ELL format')
        else:
            print("Matrix is not in CSR or in ELL")


    
    def set_n_threads(self):
        if self.n_threads is None:
            if os.getenv("OMP_NUM_THREADS") is None:
                self.n_threads=8
            else:
                self.n_threads=int(os.getenv("OMP_NUM_THREADS"))
            
    def csr_to_ell(self):
        self.len_row = max(self.A.indptr[i+1] - self.A.indptr[i] 
                                for i in range(self.A.shape[0]))
        self.len_row.astype(np.int32)
        self.shape = np.array([self.A.shape[0],self.A.shape[0]],dtype=np.int32)
        self.len_col = self.shape[0]
        if self.DX is None and self.DY is None:
            self.data,self.indices,self.valid_row_values=data_indices_cython_0(self.len_row,self.A.shape[0],self.A.indptr,self.A.data,self.A.indices,self.n_threads)
        else:
            if self.CJ is None:
                self.data,self.indices,self.dx_vec,self.dy_vec,self.conjugates,self.valid_row_values=data_indices_cython(self.len_row,self.A.shape[0],self.A.indptr,self.A.data,self.A.indices,self.DX.data,self.DY.data,self.DX.data,self.n_threads)
            else:
                self.data,self.indices,self.dx_vec,self.dy_vec,self.conjugates,self.valid_row_values=data_indices_cython(self.len_row,self.A.shape[0],self.A.indptr,self.A.data,self.A.indices,self.DX.data,self.DY.data,self.CJ.data,self.n_threads)
        return

    def save(self,path=None):
        if path is None:
            print('None path selected save as ./H.joblib')
            path='./H.joblib'
        joblib.dump(self,path)
        return

    def dot(self,a,b,x,y):
        aAxby(a,b,x,y,self.data,self.indices,self.len_row,self.len_col,self.n_threads)
        return
    
    def deep_copy(self):
        # Create a shallow copy of the current instance
        return copy.deepcopy(self)  # This avoids processing self.A again


    def modifier(self,func,*args,**kwargs):
        
        new_data = func(self, *args, **kwargs)  
        new_matrix = self.deep_copy() 
        new_matrix.data=new_data
        return new_matrix

    # def modifier_nocop(self, other, func, *args, **kwargs):
    #     """
    #     Modifies self based on func, without changing other.
    #     func can return one or multiple outputs.
    #     """
    #     result = func(other, self.data,self.dx_vec,self.dy_vec, *args, **kwargs)

    #     if isinstance(result, tuple):
    #         # Multiple outputs: assume first one modifies self
    #         self.data = result[0]
    #         self.dx_vec = result[1]
    #         self.dy_vec = result[2]
    #         return result  # return all results
    #     else:
    #         # Single output: just update self
    #         self.data = result
    #         return result

    def modifier_nocop(self, other, func, *args, **kwargs):
        """
        Modifies self based on func, without changing other.
        func signature: func(A, data_new, dx_new, dy_new, ...)
        func can return either data_new or (data_new, dx_new, dy_new)
        """
        # make independent copies / ensure C-contiguous layout for Cython
        data_new = np.ascontiguousarray(self.data.copy())
        dx_new   = np.ascontiguousarray(self.dx_vec.copy())
        dy_new   = np.ascontiguousarray(self.dy_vec.copy())

        result = func(other, data_new, dx_new, dy_new, *args, **kwargs)

        if isinstance(result, tuple):
            # expect (data_new, dx_new, dy_new)
            self.data, self.dx_vec, self.dy_vec = result[0], result[1], result[2]
            return #result
        else:
            # single output -> just update data
            self.data = result
            return #result



    def __copy__(self):
        """Create a shallow copy of the matrix."""
        new_matrix = ell_matrix(self.A)
        new_matrix.type = self.type
        new_matrix.data = self.data if self.ell_data is not None else None
        new_matrix.indices = self.indices if self.ell_indices is not None else None
        new_matrix.len_row = self.len_row
        new_matrix.len_col = self.len_col
    
        return new_matrix
    