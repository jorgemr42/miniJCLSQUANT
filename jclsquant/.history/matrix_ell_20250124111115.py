import scipy as sci
import numba as numba
import numpy as np
from time import perf_counter
import copy
import tqdm
from math import pi,sqrt

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




# @numba.njit
def precompute_thread_params(n_rows,len_row):

    n_threads = numba.get_num_threads()
    n_per_thread = (n_rows + n_threads - 1) // n_threads  # Ceiling division
    row_starts = np.zeros((n_rows), dtype=np.int32)       # For storing row_start indices

    starts = np.empty(n_threads, dtype=np.int32)
    ends = np.empty(n_threads, dtype=np.int32)

    for i in range(n_threads):
        start = i * n_per_thread
        end = min((i + 1) * n_per_thread, n_rows)
        starts[i] = start
        ends[i] = end
# Precompute row start index for each row
    for idx in range(n_rows):
        row_starts[idx] = idx * len_row
    return n_threads, starts, ends, row_starts


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

@numba.njit(parallel=True)
def dot_outside_v6(data, diagonal, indices, len_row, v,n_threads, starts, ends,row_starts):
    n_rows = len(v)  # Get the length of v
    
    combined_memory = np.empty(n_rows * 2, dtype=np.complex128)  # Allocate space for both v and v2
    v_copy = combined_memory[:n_rows]  # First half for v
    v2 = combined_memory[n_rows:]  # Second half for v2

    # Copy values from external v into the newly allocated v_copy
    # for i in range(n_rows):

    #     v_copy[i] = v[i]  # Copy values from v to v_copy
    v_copy=np.copy(v)
    

    # Precompute threading parameters

    if diagonal is None:
        for i in numba.prange(n_threads):
            start = starts[i]
            end = ends[i]
            for idx in range(start, end):
                sum_val = 0.0
                # row_start =   # Precompute row start index
                for j in range(len_row):
                    sum_val += data[row_starts[idx] + j] * v_copy[indices[row_starts[idx] + j]]
                v2[idx] = sum_val
    else:
        for i in numba.prange(n_threads):
            start = starts[i]
            end = ends[i]
            for idx in range(start, end):
                sum_val = 0.0
                row_start = row_starts[idx]  # Precompute row start index
                for j in range(len_row):
                    sum_val += data[row_start + j] * v_copy[indices[row_start + j]]
                v2[idx] = v_copy[idx] * diagonal[idx] + sum_val        
                # v2[idx] = sum_val
        # combined_memory[n_rows:]=combined_memory[n_rows:]*diagonal       

    return combined_memory[n_rows:]  # Return the combined memory containing both v and v2






# numba.set_num_threads(24)

class matrix_ell:
    """
    Def:
        Matrix ELL class.
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
        self.mul_n_threads = None
        self.mul_starts = None
        self.mul_ends = None
        self.mul_row_starts = None

        self.check_format()
        self.precompute_mul_params()
        


        
    def check_format(self):
        if sci.sparse.isspmatrix_csr(self.A):
            self.csr_to_ell(self.A)
            self.type='ELLPACK(ELL) sparse matrix format'

        elif self.A.type=='ELLPACK(ELL) sparse matrix format':
            print('Matrix is already in ELL format')
        else:
            print("Matrix is not in CSR or in ELL")
    
    def data_indices(self,A):
        self.indices=np.zeros(self.shape[0]*self.len_row,dtype=np.int64)
        self.data=np.zeros(self.shape[0]*self.len_row,dtype=np.complex128)
        for i in range(self.shape[0]):
            row_i_data=A[i,:].data
            row_i_indices=A[i,:].indices
            if len(row_i_data)<self.len_row:
                self.indices[i*self.len_row:i*self.len_row+len(row_i_data)]=row_i_indices
                self.data[i*self.len_row:i*self.len_row+len(row_i_data)]=row_i_data

                self.indices[i*self.len_row+len(row_i_data):(i+1)*self.len_row]=row_i_indices[-1]*np.ones(self.len_row-len(row_i_data),dtype=np.int64)
                self.data[i*self.len_row+len(row_i_data):(i+1)*self.len_row]=np.zeros(self.len_row-len(row_i_data),dtype=np.complex128)

            else:
                self.indices[i*self.len_row:(i+1)*self.len_row]=row_i_indices
                self.data[i*self.len_row:(i+1)*self.len_row]=row_i_data
            return



    def csr_to_ell(self,csr_matrix):
        if has_diagonal_elements(csr_matrix)==False:
            self.diagonal=None
        else:
            self.diagonal=csr_matrix.diagonal()
            csr_matrix=csr_matrix-sci.sparse.diags(csr_matrix.diagonal(),offsets=0,format='csr')

        # Determine the maximum number of non-zero elements in any row
        self.len_row = max(csr_matrix.indptr[i+1] - csr_matrix.indptr[i] 
                                for i in range(csr_matrix.shape[0]))
        self.shape=csr_matrix.shape
        self.data_indices(self,csr_matrix)
        return
    def precompute_mul_params(self):
        self.n_threads,self.mul_starts,self.mul_ends,self.mul_row_starts=precompute_thread_params(self.shape[0],self.len_row)

    def dot(self,v):
        # if self.n_threads!=numba.get_num_threads():
        #     self.precompute_mul_params()
        return dot_outside_v6(self.data, self.diagonal, self.indices, self.len_row, v,self.n_threads, self.mul_starts, self.mul_ends,self.mul_row_starts)

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
        new_matrix = matrix_ell(self.A)  # Initialize with the original matrix
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

        new_matrix = matrix_ell(self.A)  # Initialize with the original matrix
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
            
        new_matrix = matrix_ell(self.A)  # Initialize with the original matrix
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
        new_matrix = matrix_ell(self.A)  # Initialize with the original matrix
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
        new_matrix = matrix_ell(self.A)  # Initialize with the original matrix
        new_matrix.data = result_data  # Set the modified data
        new_matrix.diagonal = result_diagonal  # Set the modified data
        return new_matrix

    def __copy__(self):
        """Create a shallow copy of the matrix."""
        new_matrix = matrix_ell(self.A)
        new_matrix.type = self.type
        new_matrix.diagonal = self.diagonal.copy() if self.diagonal is not None else None
        new_matrix.data = self.data if self.ell_data is not None else None
        new_matrix.indices = self.indices if self.ell_indices is not None else None
        new_matrix.len_row = self.len_row
    
        return new_matrix
    