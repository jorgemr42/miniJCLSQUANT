import scipy as sci
import numpy as np
from time import perf_counter
import copy
import os
from minijclsquant.ell_matrix import *
import warnings
from scipy.sparse import SparseEfficiencyWarning  # Import the warning class
import matplotlib.pyplot as plt
import joblib
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

## For loading matrices with format joblib
def load_ell(path=None):
    if path is None:
        print('No path inserted')
        return 
    else:
        return joblib.load(path)


## For computing the bounds

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




## Graphene hamiltonian only 1st NN
def H_graphene(positions, t,m=0,W=0,periodic=True,type_H='CSR'):
    """
    Def:
        Tight binding hamiltonian of graphene to 1NN
    Inputs:
        positions: Positions of the lattices array it must be rectangular (you can take them from lattice_hexagonal)
        t : hopping amplitude in arbitrary units
        m : mass term which is -m in A and m in B
        W : Anderson disorder in arbitrary units between -W and W
        Periodic: Periodic boundary conditions, by default is True
    Outputs:
        H: hamiltonian in CSR format
        dx_vec: distance in X of the atoms
        dy_vec: distance in Y of the atoms
        Omega: Area of the system
        conjugates: Array of 0's and 1's in such way that (i,j) element has 0 and (j,i) element has 1.
    """

    a = 0.24595   # [nm] unit cell length
    a_cc = a/np.sqrt(3)  # [nm] carbon-carbon distance

    threshold_1NN=1.1*a_cc
    N = np.shape(positions)[0]
    row = []
    col = []
    data = []
    dx_vec=[]
    dy_vec=[]
    conjugate_list=[]

    tree = sci.spatial.cKDTree(positions)
    non_periodic_pairs = tree.query_pairs(threshold_1NN)
    for i, j in non_periodic_pairs:
        dx=positions[i,0] - positions[j,0]
        dy=positions[i,1] - positions[j,1]
        if np.sqrt(dx**2+dy**2)<=threshold_1NN:
            dx_vec.extend([-dx,dx])
            dy_vec.extend([-dy,dy])
            row.extend([i, j])
            col.extend([j, i])
            data.extend([t, t])
            conjugate_list.extend([0, 1])
            


    Lx, Ly = np.max(positions[:,0])- np.min(positions[:,0]), np.max(positions[:,1])- np.min(positions[:,1])

    if periodic==True:
        #Borders
        indices_min_x=np.where(positions[:,0]==np.min(positions[:,0]))
        indices_max_x=np.where(positions[:,0]==np.max(positions[:,0]))
        indices_min_y=np.where(positions[:,1]==np.min(positions[:,1]))
        indices_max_y=np.where(positions[:,1]==np.max(positions[:,1]))
        indices_borders=np.unique(np.concatenate((indices_min_x[0],indices_max_x[0],indices_min_y[0],indices_max_y[0])))
        S_borders=positions[indices_borders]
        # Lx, Ly = positions[indices_max_x[0][0],0]- positions[indices_min_x[0][0],0], positions[indices_max_y[0][0],1] - positions[indices_min_y[0][0],1]

        for i in (range(np.shape(S_borders)[0])):
            for j in range(i + 1, np.shape(S_borders)[0]):
                dx = (S_borders[i,0] - S_borders[j,0])
                dy = (S_borders[i,1] - S_borders[j,1])
                if np.abs(dx)>=Lx-a_cc or np.abs(dy)>=Ly-a_cc:
                    dx=dx-np.sign(dx)*(np.abs(dx)//Lx)*(Lx+np.sqrt(3)*a_cc/2)
                    dy=dy-np.sign(dy)*(np.abs(dy)//Ly)*(Ly+0.5*a_cc)
                    if np.sqrt(dx**2+dy**2) < threshold_1NN:
                        dx_vec.extend([-dx,dx])
                        dy_vec.extend([-dy,dy])
                        row.extend([indices_borders[i], indices_borders[j]])
                        col.extend([indices_borders[j], indices_borders[i]])
                        data.extend([t, t])
                        conjugate_list.extend([0, 1])


    H = sci.sparse.csr_matrix((data, (row, col)), shape=(N, N))
    Dx = sci.sparse.csr_matrix((dx_vec, (row, col)), shape=(N, N))
    Dy = sci.sparse.csr_matrix((dy_vec, (row, col)), shape=(N, N))
    conjugate_mat = sci.sparse.csr_matrix((conjugate_list, (row, col)), shape=(N, N))

    if np.abs(m)!= 0:


        diag = np.array([m if i % 2 == 0 else -m for i in range(N)])
        # Or using numpy methods:
        # diag = np.ones(N) * m
        # diag[1::2] = -m
        
        H += sci.sparse.diags(diag, offsets=0, shape=(N, N), format="csr")

        conjugate_mat += sci.sparse.diags(np.array([2] * N), offsets=0, shape=(N, N), format="csr")

    if np.abs(W)!= 0:
        print('b')
        H+=sci.sparse.diags((W)*(2*np.random.rand(H.shape[0])-1), offsets=0, format="csr")
        
        conjugate_mat.setdiag(np.array([2, 2] * (H.shape[0]//2)))

    if type_H=='ELL':
        H=H+1e-6*(Dx+Dy)
        Dx=Dx+1e-6*H
        Dy=Dy+1e-6*H

        H_ell=ell_matrix(H,Dx,Dy,conjugate_mat)
        H_ell.Omega=Lx*Ly
        return H_ell
    elif type_H=='CSR':
        return H,Dx.data,Dy.data,Lx*Ly,conjugate_mat.data
# Graphene hamiltonian only 1st NN

