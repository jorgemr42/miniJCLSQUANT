import scipy as sci
import numpy as np
from time import perf_counter
import copy
import os
from jclsquant.ell_matrix import *
import warnings
from scipy.sparse import SparseEfficiencyWarning  # Import the warning class
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


def write_sparse(filename, data):
    """
    Def: 
        Saves a CSR matrix into the format used by LSQUANT. Creates the folder 'operators' if it is not created.
    Inputs:
        A: CSR matrix
    Outputs:
        None
    """

    # Ensure the 'operators' directory exists
    if not os.path.exists('operators'):
        os.makedirs('operators')

    # Adjust filename to include the 'operators' directory
    filepath = os.path.join('operators', filename)

    # Write to the file in the 'operators' directory
    hr=open(filepath,'w')
    hr.write(str(data.shape[0]) + ' ' + str(data.size) + '\n')
    values = data.data
    valueswrite = [str(x) + ' ' + str(y) for x, y in zip(values.real, values.imag)]
    valueswrite = ' '.join(valueswrite)
    hr.write(valueswrite + '\n')
    hr.write(' '.join(data.indices.astype(str)) + '\n')
    hr.write(' '.join(data.indptr.astype(str)))


### LSQUANT -> CSR
def load_sparse(filename):
    '''Reads matrix data from address filename in LSQUANT input format and returns CSR matrix.'''
    
    with open(filename, 'r', encoding='latin-1') as file:
        # Read the first line containing dimensions
        dimensions = file.readline().strip().split()
        rows, size = map(int, dimensions)
        
        # Read the second line containing values
        values = file.readline().strip().split()
        values = [complex(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]

        # Read the third line containing indices
        indices = list(map(int, file.readline().strip().split()))

        # Read the fourth line containing indptr
        indptr = list(map(int, file.readline().strip().split()))

    # Reconstruct CSR matrix
    cols = len(indptr) - 1
    data = sci.sparse.csr_matrix((values, indices, indptr), shape=(rows, cols))

    return data



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
        H+=sci.sparse.diags(np.array([m, -m] * (N//2)), offsets=0, format="csr")
        Dx.setdiag(np.array([1e-9, 1e-9] * (N//2)))
        Dy.setdiag(np.array([1e-9, 1e-9] * (N//2)))
        conjugate_mat.setdiag(np.array([2, 2] * (N//2)))
    if np.abs(W)!= 0:
        H+=sci.sparse.diags((W)*(2*np.random.rand(H.shape[0])-1), offsets=0, format="csr")
        Dx.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        Dy.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        conjugate_mat.setdiag(np.array([2, 2] * (H.shape[0]//2)))

    if type_H=='ELL':
        H_ell=ell_matrix(H,Dx,Dy,conjugate_mat)
        H_ell.Omega=Lx*Ly
        return H_ell
    elif type_H=='CSR':
        return H,Dx.data,Dy.data,Lx*Ly,conjugate_mat.data


## Haldane hamiltonian
def sign_haldane(dx, dy,sublattice):
    a1=np.array([1,0])
    # Compute the 2D cross product (determinant)
    dot_product=np.dot(a1,np.array([dx,dy]))/(np.linalg.norm(a1)*np.linalg.norm(np.array([dx,dy])))
    angle=np.arccos(dot_product)*180/np.pi
    if (np.cross(a1,np.array([dx,dy]))/(np.linalg.norm(a1)*np.linalg.norm(np.array([dx,dy]))))<0:
        angle=-angle
        angle=angle+360
    return (-1)**(sublattice)*((-1)**(round(angle,0)//60))
 

def H_haldane(positions, t,t2,m=0,W=0,periodic=True,type_H='CSR'):
    """
    Def:
        Tight binding hamiltonian of the Haldane model
    Inputs:
        positions: Positions of the lattices array it must be rectangular (you can take them from lattice_hexagonal)
        t: hopping amplitude in arbitrary units to 1NN
        t2: hopping amplitude in aribitrary units (must be real the phase is automatic inserted)
        Periodic: Periodic boundary conditions, by default is True
    Outputs:
        H: hamiltonian in CSR format
        dx_vec: distance in X of the atoms
        dy_vec: distance in Y of the atoms
        Omega: Area of the system
    """
    a = 0.24595   # [nm] unit cell length
    a_cc = a/np.sqrt(3)  # [nm] carbon-carbon distance
    
    threshold_1NN=1.1*a_cc
    threshold_2NN=1.1*np.sqrt(3)*a_cc
    N = np.shape(positions)[0]
    row = []
    col = []
    data = []
    dx_vec=[]
    dy_vec=[]
    conjugate_list=[]



    tree = sci.spatial.cKDTree(positions)
    non_periodic_pairs = tree.query_pairs(threshold_2NN)
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

        if np.sqrt(dx**2+dy**2)>=threshold_1NN:
            dx_vec.extend([-dx,dx])
            dy_vec.extend([-dy,dy])
            row.extend([i, j])
            col.extend([j, i])
            data.extend([t2*1j*sign_haldane(dx,dy,i), np.conj(t2*1j*sign_haldane(dx,dy,i))])
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

        #2NN borders
        
        indices_min_x=np.where(positions[:,0]==np.unique(positions[:,0])[1])
        indices_max_x=np.where(positions[:,0]==np.unique(positions[:,0])[-2])
        indices_min_y=np.where(positions[:,1]==np.unique(positions[:,1])[1])
        indices_max_y=np.where(positions[:,1]==np.unique(positions[:,1])[-2])
        indices_borders_2NN=np.unique(np.concatenate((indices_min_x[0],indices_max_x[0],indices_min_y[0],indices_max_y[0])))
        
        indices_borders_end=np.unique(np.concatenate((indices_borders,indices_borders_2NN)))
        S_borders_2NN=positions[indices_borders_end]

        for i in (range(np.shape(S_borders_2NN)[0])):
            for j in range(i + 1, np.shape(S_borders_2NN)[0]):
                dx = (S_borders_2NN[i,0] - S_borders_2NN[j,0])
                dy = (S_borders_2NN[i,1] - S_borders_2NN[j,1])
                if np.abs(dx)>=Lx-threshold_2NN or np.abs(dy)>=Ly-threshold_2NN:
                    dx=dx-np.sign(dx)*(np.abs(dx)//(Lx-np.sqrt(3)*a_cc))*(Lx+np.sqrt(3)*a_cc/2)
                    dy=dy-np.sign(dy)*(np.abs(dy)//(Ly-np.sqrt(3)*a_cc))*(Ly+0.5*a_cc)
                    if np.sqrt(dx**2+dy**2) < threshold_2NN and np.sqrt(dx**2+dy**2)>threshold_1NN:
                        dx_vec.extend([-dx,dx])
                        dy_vec.extend([-dy,dy])
                        row.extend([indices_borders_end[i], indices_borders_end[j]])
                        col.extend([indices_borders_end[j], indices_borders_end[i]])
                        data.extend([t2*1j*sign_haldane(dx,dy,i), np.conj(t2*1j*sign_haldane(dx,dy,i))])
                        conjugate_list.extend([0, 1])

    H = sci.sparse.csr_matrix((data, (row, col)), shape=(N, N))
    Dx = sci.sparse.csr_matrix((dx_vec, (row, col)), shape=(N, N))
    Dy = sci.sparse.csr_matrix((dy_vec, (row, col)), shape=(N, N))
    conjugate_mat = sci.sparse.csr_matrix((conjugate_list, (row, col)), shape=(N, N))
    if np.abs(m)!= 0:
        H+=sci.sparse.diags(np.array([m, -m] * (N//2)), offsets=0, format="csr")
        Dx.setdiag(np.array([1e-9, 1e-9] * (N//2)))
        Dy.setdiag(np.array([1e-9, 1e-9] * (N//2)))
        conjugate_mat.setdiag(np.array([2, 2] * (N//2)))
    if np.abs(W)!= 0:
        H+=sci.sparse.diags((W)*(2*np.random.rand(H.shape[0])-1), offsets=0, format="csr")
        Dx.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        Dy.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        conjugate_mat.setdiag(np.array([2, 2] * (H.shape[0]//2)))

    if type_H=='ELL':
        H_ell=ell_matrix(H,Dx,Dy,conjugate_mat)
        H_ell.Omega=Lx*Ly
        return H_ell
    elif type_H=='CSR':
        return H,Dx.data,Dy.data,Lx*Ly,conjugate_mat.data

    


def H_AB_bilayer(positions, gamma0,gamma1=0.4,m=0,W=0,E=0,periodic=True,type_H='ELL'):
    """
    Def:
        Tight binding hamiltonian of the AB (Bernal) bilayer of graphene
    Inputs:
        positions: Positions of the lattices array it must be rectangular (you can take them from lattice_hexagonal)
        gamma0: In plane hopping amplitude in arbitrary units to 1NN.
        gamma1: Inter plane hopping amplitude of A up and B down sites, which are one on top of the other.
        Periodic: Periodic boundary conditions, by default is True
    Outputs:
        H: hamiltonian in CSR format
        dx_vec: distance in X of the atoms
        dy_vec: distance in Y of the atoms
        Omega: Area of the system
    """
    a = 0.24595   # [nm] unit cell length
    a_cc = a/np.sqrt(3)  # [nm] carbon-carbon distance
    d=0.335

    threshold_gamma0=1.1*a_cc
    threshold_gamma1=1.1*d

    N = np.shape(positions)[0]
    row = []
    col = []
    data = []
    dx_vec=[]
    dy_vec=[]
    dz_vec=[]
    conjugate_list=[]



    tree = sci.spatial.cKDTree(positions)
    non_periodic_pairs = tree.query_pairs(threshold_gamma1)
    for i, j in non_periodic_pairs:
        dx=positions[i,0] - positions[j,0]
        dy=positions[i,1] - positions[j,1]
        dz=positions[i,2] - positions[j,2]
        if np.sqrt(dx**2+dy**2+dz**2)<=threshold_gamma0 and np.abs(dz)<1e-6:
            dx_vec.extend([-dx,dx])
            dy_vec.extend([-dy,dy])
            dz_vec.extend([-dz,dz])
            row.extend([i, j])
            col.extend([j, i])
            data.extend([gamma0, gamma0])
            conjugate_list.extend([0, 1])

        if  np.abs(dy)<1e-6 and np.abs(dy)<1e-6 and np.abs(dz)>1e-6:
            dx_vec.extend([-dx,dx])
            dy_vec.extend([-dy,dy])
            dz_vec.extend([-dz,dz])
            row.extend([i, j])
            col.extend([j, i])
            data.extend([gamma1, gamma1])
            conjugate_list.extend([0, 1])


    if periodic==True:
        #Borders of the first monolayer
        positions_1=positions[:len(positions[:,0])//2]
        Lx, Ly, Lz = np.max(positions_1[:,0])- np.min(positions_1[:,0]), np.max(positions_1[:,1])- np.min(positions_1[:,1]), np.max(positions_1[:,2])- np.min(positions_1[:,2])

        indices_min_x=np.where(positions_1[:,0]==np.min(positions_1[:,0]))
        indices_max_x=np.where(positions_1[:,0]==np.max(positions_1[:,0]))
        indices_min_y=np.where(positions_1[:,1]==np.min(positions_1[:,1]))
        indices_max_y=np.where(positions_1[:,1]==np.max(positions_1[:,1]))
        indices_borders=np.unique(np.concatenate((indices_min_x[0],indices_max_x[0],indices_min_y[0],indices_max_y[0])))
        S_borders=positions_1[indices_borders]
        for i in (range(np.shape(S_borders)[0])):
            for j in range(i + 1, np.shape(S_borders)[0]):
                dx = (S_borders[i,0] - S_borders[j,0])
                dy = (S_borders[i,1] - S_borders[j,1])
                dz = (S_borders[i,2] - S_borders[j,2])
                if np.abs(dx)>=Lx-a_cc or np.abs(dy)>=Ly-a_cc and np.abs(dz)<1e-6:
                    dx=dx-np.sign(dx)*(np.abs(dx)//Lx)*(Lx+np.sqrt(3)*a_cc/2)
                    dy=dy-np.sign(dy)*(np.abs(dy)//Ly)*(Ly+0.5*a_cc)
                    if np.sqrt(dx**2+dy**2) < threshold_gamma0 :
                        dx_vec.extend([-dx,dx])
                        dy_vec.extend([-dy,dy])
                        dz_vec.extend([-dz,dz])
                        row.extend([indices_borders[i], indices_borders[j]])
                        col.extend([indices_borders[j], indices_borders[i]])
                        data.extend([gamma0, gamma0])
                        conjugate_list.extend([0, 1])
      
        #Borders of the second monolayer
        positions_1=positions[len(positions[:,0])//2:]

        Lx, Ly, Lz = np.max(positions_1[:,0])- np.min(positions_1[:,0]), np.max(positions_1[:,1])- np.min(positions_1[:,1]), np.max(positions_1[:,2])- np.min(positions_1[:,2])

        indices_min_x=np.where(positions_1[:,0]==np.min(positions_1[:,0]))
        indices_max_x=np.where(positions_1[:,0]==np.max(positions_1[:,0]))
        indices_min_y=np.where(positions_1[:,1]==np.min(positions_1[:,1]))
        indices_max_y=np.where(positions_1[:,1]==np.max(positions_1[:,1]))
        indices_borders=np.unique(np.concatenate((indices_min_x[0],indices_max_x[0],indices_min_y[0],indices_max_y[0])))
        S_borders=positions_1[indices_borders]
        for i in (range(np.shape(S_borders)[0])):
            for j in range(i + 1, np.shape(S_borders)[0]):
                dx = (S_borders[i,0] - S_borders[j,0])
                dy = (S_borders[i,1] - S_borders[j,1])
                dz = (S_borders[i,2] - S_borders[j,2])
                if np.abs(dx)>=Lx-a_cc or np.abs(dy)>=Ly-a_cc and np.abs(dz)<1e-6:
                    dx=dx-np.sign(dx)*(np.abs(dx)//Lx)*(Lx+np.sqrt(3)*a_cc/2)
                    dy=dy-np.sign(dy)*(np.abs(dy)//Ly)*(Ly+0.5*a_cc)
                    if np.sqrt(dx**2+dy**2) < threshold_gamma0 :
                        dx_vec.extend([-dx,dx])
                        dy_vec.extend([-dy,dy])
                        dz_vec.extend([-dz,dz])
                        row.extend([indices_borders[i]+len(positions[:,0])//2, indices_borders[j]+len(positions[:,0])//2])
                        col.extend([indices_borders[j]+len(positions[:,0])//2, indices_borders[i]+len(positions[:,0])//2])
                        data.extend([gamma0, gamma0])
                        conjugate_list.extend([0, 1])

    
    H = sci.sparse.csr_matrix((data, (row, col)), shape=(N, N))
    Dx = sci.sparse.csr_matrix((dx_vec, (row, col)), shape=(N, N))
    Dy = sci.sparse.csr_matrix((dy_vec, (row, col)), shape=(N, N))
    Dz = sci.sparse.csr_matrix((dz_vec, (row, col)), shape=(N, N))
    conjugate_mat = sci.sparse.csr_matrix((conjugate_list, (row, col)), shape=(N, N))

    if np.abs(m)!= 0:
        H+=sci.sparse.diags(np.array([m, -m] * (N//2)), offsets=0, format="csr")
        Dx.setdiag(np.array([1e-9, 1e-9] * (N//2)))
        Dy.setdiag(np.array([1e-9, 1e-9] * (N//2)))
        conjugate_mat.setdiag(np.array([2, 2] * (N//2)))
    if np.abs(W)!= 0:
        H+=sci.sparse.diags((W)*(2*np.random.rand(H.shape[0])-1), offsets=0, format="csr")
        Dx.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        Dy.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        conjugate_mat.setdiag(np.array([2, 2] * (H.shape[0]//2)))
    if np.abs(E)!= 0:
        H+=sci.sparse.diags(np.concatenate((-E*np.ones(N//2),E*np.ones(N//2))), offsets=0, format="csr")
        Dx.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        Dy.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        conjugate_mat.setdiag(np.array([2, 2] * (H.shape[0]//2)))

    ## Calculating volume of the system
    Lx, Ly, Lz = np.max(positions[:,0])- np.min(positions[:,0]), np.max(positions[:,1])- np.min(positions[:,1]), np.max(positions[:,2])- np.min(positions[:,2])
    if type_H=='ELL':
        H_ell=ell_matrix(H,Dx,Dy,conjugate_mat)
        H_ell.Omega=Lx*Ly
        return H_ell
    elif type_H=='CSR':
        return H,Dx.data,Dy.data,Lx*Ly,conjugate_mat.data


def H_AB_bilayer_2(positions, gamma0,gamma1=0.4,m=0,W=0,E=0,periodic=True,type_H='ELL'):
    """
    Def:
        Tight binding hamiltonian of the AB (Bernal) bilayer of graphene
    Inputs:
        positions: Positions of the lattices array it must be rectangular (you can take them from lattice_hexagonal)
        gamma0: In plane hopping amplitude in arbitrary units to 1NN.
        gamma1: Inter plane hopping amplitude of A up and B down sites, which are one on top of the other.
        Periodic: Periodic boundary conditions, by default is True
    Outputs:
        H: hamiltonian in CSR format
        dx_vec: distance in X of the atoms
        dy_vec: distance in Y of the atoms
        Omega: Area of the system
    """
    a = 0.24595   # [nm] unit cell length
    a_cc = a/np.sqrt(3)  # [nm] carbon-carbon distance
    d=0.335

    threshold_gamma0=1.1*a_cc
    threshold_gamma1=1.1*d

    N = np.shape(positions)[0]
    row = []
    col = []
    data = []
    dx_vec=[]
    dy_vec=[]
    dz_vec=[]
    conjugate_list=[]



    tree = sci.spatial.cKDTree(positions)
    non_periodic_pairs = tree.query_pairs(threshold_gamma1)
    for i, j in non_periodic_pairs:
        dx=positions[i,0] - positions[j,0]
        dy=positions[i,1] - positions[j,1]
        dz=positions[i,2] - positions[j,2]
        if np.sqrt(dx**2+dy**2+dz**2)<=threshold_gamma0 and np.abs(dz)<1e-6:
            dx_vec.extend([-dx,dx])
            dy_vec.extend([-dy,dy])
            dz_vec.extend([-dz,dz])
            row.extend([i, j])
            col.extend([j, i])
            data.extend([gamma0, gamma0])
            conjugate_list.extend([0, 1])

        if  np.abs(dy)<1e-6 and np.abs(dy)<1e-6 and np.abs(dz)>1e-6:
            dx_vec.extend([-dx,dx])
            dy_vec.extend([-dy,dy])
            dz_vec.extend([-dz,dz])
            row.extend([i, j])
            col.extend([j, i])
            data.extend([gamma1, gamma1])
            conjugate_list.extend([0, 1])


    if periodic==True:

        #Borders of the first monolayer
        positions_1=positions[0::2]
        Lx, Ly, Lz = np.max(positions_1[:,0])- np.min(positions_1[:,0]), np.max(positions_1[:,1])- np.min(positions_1[:,1]), np.max(positions_1[:,2])- np.min(positions_1[:,2])

        indices_min_x=np.where(positions_1[:,0]==np.min(positions_1[:,0]))
        indices_max_x=np.where(positions_1[:,0]==np.max(positions_1[:,0]))
        indices_min_y=np.where(positions_1[:,1]==np.min(positions_1[:,1]))
        indices_max_y=np.where(positions_1[:,1]==np.max(positions_1[:,1]))
        indices_borders=np.unique(np.concatenate((indices_min_x[0],indices_max_x[0],indices_min_y[0],indices_max_y[0])))
        S_borders=positions_1[indices_borders]
        for i in (range(np.shape(S_borders)[0])):
            for j in range(i + 1, np.shape(S_borders)[0]):
                dx = (S_borders[i,0] - S_borders[j,0])
                dy = (S_borders[i,1] - S_borders[j,1])
                dz = (S_borders[i,2] - S_borders[j,2])
                if np.abs(dx)>=Lx-a_cc or np.abs(dy)>=Ly-a_cc and np.abs(dz)<1e-6:
                    dx=dx-np.sign(dx)*(np.abs(dx)//Lx)*(Lx+np.sqrt(3)*a_cc/2)
                    dy=dy-np.sign(dy)*(np.abs(dy)//Ly)*(Ly+0.5*a_cc)
                    if np.sqrt(dx**2+dy**2) < threshold_gamma0 :
                        dx_vec.extend([-dx,dx])
                        dy_vec.extend([-dy,dy])
                        dz_vec.extend([-dz,dz])
                        row.extend([indices_borders[i]*2, indices_borders[j]*2])
                        col.extend([indices_borders[j]*2, indices_borders[i]*2])
                        data.extend([gamma0, gamma0])
                        conjugate_list.extend([0, 1])
        
        #Borders of the second monolayer
        positions_1=positions[1::2]

        Lx, Ly, Lz = np.max(positions_1[:,0])- np.min(positions_1[:,0]), np.max(positions_1[:,1])- np.min(positions_1[:,1]), np.max(positions_1[:,2])- np.min(positions_1[:,2])

        indices_min_x=np.where(positions_1[:,0]==np.min(positions_1[:,0]))
        indices_max_x=np.where(positions_1[:,0]==np.max(positions_1[:,0]))
        indices_min_y=np.where(positions_1[:,1]==np.min(positions_1[:,1]))
        indices_max_y=np.where(positions_1[:,1]==np.max(positions_1[:,1]))
        indices_borders=np.unique(np.concatenate((indices_min_x[0],indices_max_x[0],indices_min_y[0],indices_max_y[0])))
        S_borders=positions_1[indices_borders]
        for i in (range(np.shape(S_borders)[0])):
            for j in range(i + 1, np.shape(S_borders)[0]):
                dx = (S_borders[i,0] - S_borders[j,0])
                dy = (S_borders[i,1] - S_borders[j,1])
                dz = (S_borders[i,2] - S_borders[j,2])
                if np.abs(dx)>=Lx-a_cc or np.abs(dy)>=Ly-a_cc and np.abs(dz)<1e-6:
                    dx=dx-np.sign(dx)*(np.abs(dx)//Lx)*(Lx+np.sqrt(3)*a_cc/2)
                    dy=dy-np.sign(dy)*(np.abs(dy)//Ly)*(Ly+0.5*a_cc)
                    if np.sqrt(dx**2+dy**2) < threshold_gamma0 :
                        dx_vec.extend([-dx,dx])
                        dy_vec.extend([-dy,dy])
                        dz_vec.extend([-dz,dz])
                        row.extend([indices_borders[i]*2+1, indices_borders[j]*2+1])
                        col.extend([indices_borders[j]*2+1, indices_borders[i]*2+1])
                        data.extend([gamma0, gamma0])
                        conjugate_list.extend([0, 1])

    
    H = sci.sparse.csr_matrix((data, (row, col)), shape=(N, N))
    Dx = sci.sparse.csr_matrix((dx_vec, (row, col)), shape=(N, N))
    Dy = sci.sparse.csr_matrix((dy_vec, (row, col)), shape=(N, N))
    Dz = sci.sparse.csr_matrix((dz_vec, (row, col)), shape=(N, N))
    conjugate_mat = sci.sparse.csr_matrix((conjugate_list, (row, col)), shape=(N, N))

    if np.abs(m)!= 0:
        print('Not well implemented')
        H+=sci.sparse.diags(np.array([m, -m] * (N//2)), offsets=0, format="csr")
        Dx.setdiag(np.array([1e-9, 1e-9] * (N//2)))
        Dy.setdiag(np.array([1e-9, 1e-9] * (N//2)))
        conjugate_mat.setdiag(np.array([2, 2] * (N//2)))
    if np.abs(W)!= 0:
        H+=sci.sparse.diags((W)*(2*np.random.rand(H.shape[0])-1), offsets=0, format="csr")
        Dx.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        Dy.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        conjugate_mat.setdiag(np.array([2, 2] * (H.shape[0]//2)))
    if np.abs(E)!= 0:
        H+=sci.sparse.diags(np.array([-E, E] * (N//2)), offsets=0, format="csr")
        Dx.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        Dy.setdiag(np.array([1e-9, 1e-9] * (H.shape[0]//2)))
        conjugate_mat.setdiag(np.array([2, 2] * (H.shape[0]//2)))

    ## Calculating volume of the system
    Lx, Ly, Lz = np.max(positions[:,0])- np.min(positions[:,0]), np.max(positions[:,1])- np.min(positions[:,1]), np.max(positions[:,2])- np.min(positions[:,2])
    if type_H=='ELL':
        H_ell=ell_matrix(H,Dx,Dy,conjugate_mat)
        H_ell.Omega=Lx*Ly
        return H_ell
    elif type_H=='CSR':
        return H,Dx.data,Dy.data,Lx*Ly,conjugate_mat.data

