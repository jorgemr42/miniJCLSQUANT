import scipy as sci
import numba as numba
import numpy as np
from time import perf_counter
import copy
import tqdm
from math import pi,sqrt
from .matrix_vector_module import matrix_vector
import os
import multiprocessing

    


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
        rows[i]=i//A.len_row
        cols[i]=A.indices[i]
    B=sci.sparse.csr_matrix((A.data, (rows, cols)), shape=A.shape)
    if A.diagonal is not None:
        B.setdiag(A.diagonal)
    return B






## Functions that are paraleize and that are used inside the class
def has_diagonal_elements(csr):
    # Get the row and column indices of non-zero elements
    rows, cols = csr.nonzero()
    
    # Check if any non-zero element is on the diagonal
    return any(rows[i] == cols[i] for i in range(len(rows)))

    
@numba.njit(parallel=True)
def sub_outside(data, other):
    return data - other

@numba.njit(parallel=True)    
def add_outside(data_self,other):
    return data_self+other    
@numba.njit(parallel=True)
def mul_outside(data, other):
    return  data * other  
@numba.njit(parallel=True)
def div_outside(data, other):
    return data / other  


# numba.set_num_threads(24)

class matrix_ell_c:
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
    def __init__(self, A):
        self.A = A
        self.type=None
        self.diagonal = None       
        self.ell_data = None      
        self.ell_indices = None
        self.len_row = None
        self.shape=None
        self.n_threads=None


        # Funcitons that automatically run when started
        self.check_format()
        self.set_n_threads()
        


        
    def check_format(self):
        if sci.sparse.isspmatrix_csr(self.A):
            self.csr_to_ell(self.A)
            self.type='ELLPACK(ELL) sparse matrix format'

        elif self.A.type=='ELLPACK(ELL) sparse matrix format':
            print('Matrix is already in ELL format')
        else:
            print("Matrix is not in CSR or in ELL")

    def csr_to_ell(self,csr_matrix):
        if has_diagonal_elements(csr_matrix)==False:
            self.diagonal=None
        else:
            self.diagonal=csr_matrix.diagonal()
            csr_matrix=csr_matrix-sci.sparse.diags(csr_matrix.diagonal(),offsets=0,format='csr')

        # Determine the maximum number of non-zero elements in any row
        self.len_row = max(csr_matrix.indptr[i+1] - csr_matrix.indptr[i] 
                                for i in range(csr_matrix.shape[0]))
        self.data = csr_matrix.data
        if csr_matrix.indices.dtype == np.int32:
            self.indices = csr_matrix.indices.astype(np.int64) 
        else:
            self.indices = csr_matrix.indices
        self.shape=csr_matrix.shape
        return
    def set_n_threads(self):
        if self.n_threads is None:
            self.n_threads=os.getenv("OMP_NUM_THREADS")
            if self.n_threads is None:
                self.n_threads=multiprocessing.cpu_count()            
        return
            


    def dot(self,v):
        return matrix_vector(self.data,self.diagonal, self.indices, self.len_row, v,self.n_threads)

    def deep_copy(self):
        # Create a shallow copy of the current instance
        return copy.deepcopy(self)  # This avoids processing self.A again

    def modifier(self,func,*args,**kwargs):
        if self.diagonal is not None:
            self.data = self.data.astype(np.complex128)
            self.diagonal = self.diagonal.astype(np.complex128)
            self.len_row = self.len_row.astype(np.int64)
            self.indices = self.indices.astype(np.int64)

            new_data ,new_diagonal =func(self.data,self.diagonal,self.len_row,self.indices, *args, **kwargs)
        else:
            self.data = self.data.astype(np.complex128)
            self.len_row = self.len_row.astype(np.int64)
            self.indices = self.indices.astype(np.int64)

            new_data ,new_diagonal=func(self.data,np.array([0],dtype=np.complex128),self.len_row,self.indices, *args, **kwargs)
        if len(new_diagonal)==1:
            new_diagonal=None

        
        new_matrix = self.deep_copy() 
        new_matrix.data=new_data
        new_matrix.diagonal=new_diagonal
        return new_matrix


    def __truediv__(self, other):
        """Define behavior for the division operator."""
        if isinstance(other, (int, float)):
            result_data = div_outside(self.data,other) 
            if self.diagonal is not None:
                result_diagonal = div_outside(self.diagonal,other)
            else:
                result_diagonal=None  
        elif isinstance(other, (np.ndarray, list)):
            result_data= div_outside(self.data,other)
            if self.diagonal is not None:
                result_diagonal = div_outside(self.diagonal,other)
            else:
                result_diagonal=None 
        else:
            result_data = div_outside(self.data,other.data) 
            if self.diagonal is not None:
                result_diagonal = div_outside(self.diagonal,other.diagonal) 
            else:
                result_diagonal=None

        
        # Return a new MatrixELL object or whatever is appropriate
        new_matrix = matrix_ell_c(self.A)  # Initialize with the original matrix
        new_matrix.data = result_data  # Set the modified data
        new_matrix.diagonal = result_diagonal  # Set the modified data

        return new_matrix
        
    def __add__(self, other):
        """Define behavior for the addition operator."""
        if isinstance(other, (int, float)):
            result_data = add_outside(self.data,other) 
            if self.diagonal is not None:
                result_diagonal = add_outside(self.diagonal,other)
            else:
                result_diagonal=add_outside(np.zeros((self.shape[0]),dtype=complex), other*np.ones((self.shape[0]),dtype=complex))  
        elif isinstance(other, (np.ndarray, list)):
            result_data = div_outside(self.data,other)
            if self.diagonal is not None:
                result_diagonal = add_outside(self.diagonal,other)
            else:
                result_diagonal=add_outside(np.zeros(len(other),dtype=complex), other)  
        else:
            result_data= add_outside(self.data,other.data) 
            if self.diagonal is not None:
                result_diagonal = add_outside(self.diagonal,other.diagonal) 
            elif other.diagonal is not None:
                result_diagonal=add_outside(np.zeros(len(other.diagonal),dtype=complex), other.diagonal)
            else:
                result_diagonal=None

        new_matrix = matrix_ell_c(self.A)  # Initialize with the original matrix
        new_matrix.data = result_data  # Set the modified data
        new_matrix.diagonal = result_diagonal  # Set the modified data

        return new_matrix
    def __sub__(self, other):
        """Define behavior for the addition operator."""
        if isinstance(other, (int, float)):
            result_data = sub_outside(self.data,other) 
            if self.diagonal is not None:
                result_diagonal = sub_outside(self.diagonal,other)
            else:
                result_diagonal=sub_outside(np.zeros((self.shape[0]),dtype=complex), other*np.ones((self.shape[0]),dtype=complex))  
        elif isinstance(other, (np.ndarray, list)):
            result_data = sub_outside(self.data,other)
            if self.diagonal is not None:
                result_diagonal = sub_outside(self.diagonal,other)
            else:
                result_diagonal=sub_outside(np.zeros(len(other),dtype=complex), other) 
        else:
            result_data= sub_outside(self.data,other.data) 
            if self.diagonal is not None:
                result_diagonal = sub_outside(self.diagonal,other.diagonal) 
            elif other.diagonal is not None:
                result_diagonal=sub_outside(np.zeros(len(other.diagonal),dtype=complex), other.diagonal)
            else:
                result_diagonal=None
            
        new_matrix = matrix_ell_c(self.A)  # Initialize with the original matrix
        new_matrix.data = result_data  # Set the modified data
        new_matrix.diagonal = result_diagonal  # Set the modified data

        return new_matrix
    def __mul__(self, other):
        """Define behavior for the addition operator."""
        if isinstance(other, (int, float)):
            result_data = mul_outside(self.data,other) 
            if self.diagonal is not None:
                result_diagonal = mul_outside(self.diagonal,other)
            else:
                result_diagonal = None 
        elif isinstance(other, (np.ndarray, list)):
            result_data= sub_outside(self.data,other)
            if self.diagonal is not None:
                result_diagonal = mul_outside(self.diagonal,other) 
            else:
                result_diagonal=None 
        else:
            result_data,result_diagonal = sub_outside(self.data,other.data) 
            if self.diagonal is not None:
                result_diagonal = mul_outside(self.diagonal,other.diagonal)
            else:
                result_diagonal=None 
        new_matrix = matrix_ell_c(self.A)  # Initialize with the original matrix
        new_matrix.data = result_data  # Set the modified data
        new_matrix.diagonal = result_diagonal  # Set the modified data
        return new_matrix
    def __rmul__(self, other):
        """Define behavior for the addition operator."""
        if isinstance(other, (int, float)):
            result_data = mul_outside(self.data,other) 
            if self.diagonal is not None:
                result_diagonal = mul_outside(self.diagonal,other) 
            else:
                result_diagonal=None 
        elif isinstance(other, (np.ndarray, list)):
            result_data= sub_outside(self.data,other)
            if self.diagonal is not None:
                result_diagonal = mul_outside(self.diagonal,other) 
            else:
                result_diagonal=None 
        else:
            result_data,result_diagonal = sub_outside(self.data,other.data) 
            if self.diagonal is not None:
                result_diagonal = mul_outside(self.diagonal,other.diagonal)
            else:
                result_diagonal=None 
        new_matrix = matrix_ell_c(self.A)  # Initialize with the original matrix
        new_matrix.data = result_data  # Set the modified data
        new_matrix.diagonal = result_diagonal  # Set the modified data
        return new_matrix

    def __copy__(self):
        """Create a shallow copy of the matrix."""
        new_matrix = matrix_ell_c(self.A)
        new_matrix.type = self.type
        new_matrix.diagonal = self.diagonal.copy() if self.diagonal is not None else None
        new_matrix.data = self.data if self.ell_data is not None else None
        new_matrix.indices = self.indices if self.ell_indices is not None else None
        new_matrix.len_row = self.len_row
    
        return new_matrix
    