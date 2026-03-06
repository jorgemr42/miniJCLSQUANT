import scipy as sci
import numpy as np
from time import perf_counter
import copy

def lattice_hexagonal(N):
    """
    Def:
        Returns the positions of N atoms in an hexagonal lattice with graphene lattice constant.
    Inputs:
        N: Number of atoms, if int(np.sqrt(N//4))/np.sqrt(N//4)!=1.0 then the ouput wont have exactly N.
    Outputs:
        S:  array of size (N,2) where the S[:,0] is the x position and S[:,1] in the y position.
    """
    ## N is the number of atoms that we want our system to have it

    a = 0.24595   # [nm] unit cell length
    a_cc = a/np.sqrt(3)  # [nm] carbon-carbon distance

    N_tot=N//4 #Number of unit cells in the 4 atom basis
    
    if int(np.sqrt(N_tot))/np.sqrt(N_tot)!=1.0:
        print('The number of total atoms wont be the one inserted if you want otherwise insert an even power of 2')


    a1=np.array([np.sqrt(3)*a_cc/2,a_cc/2])
    a2=np.array([0,a_cc])
    S_0=np.zeros((4,2),dtype=np.float64)
    S_0[0,:]=np.array([0,0])
    S_0[1,:]=a2
    S_0[2,:]=a2+a1
    S_0[3,:]=a2+a1+a2
    N_cells_x=int(np.sqrt(N_tot))
    N_cells_y=N_cells_x

    Sx=np.zeros((N_cells_x*4,2),dtype=np.float64)
    Sx[0:4]=S_0
    for i in np.arange(4,4*N_cells_x,4):
        Sx[i:i+4,0],Sx[i:i+4,1]=S_0[:,0]+(i//4)*np.sqrt(3)*a_cc,S_0[:,1]


    S=np.zeros((N_cells_x*4*N_cells_y,2),dtype=np.float64)
    S[0:4*N_cells_x]=Sx
    for j in np.arange(N_cells_x*4,N_cells_x*4*N_cells_y,N_cells_x*4):
        S[j:j+4*N_cells_x,0],S[j:j+4*N_cells_x,1]=Sx[:,0],Sx[:,1]+(j//(4*N_cells_x))*3*a_cc

    return S
 

def vacancies(S,n,seed=None):
    """
    Def:
        Delete a percentage of the site of an array of sites.
    Inputs:
        W : Array of sites in the order S[:,i] coordinate i of all the sites.
        n : Percetag eof sites that you want to remove.
        seed : Random seed
    Outputs:
        S_1 : Array of sites with already deleted sites. 
    """

    N_d=int(S.shape[0]*n/100)

    if N_d==0:
        print('You have insert a very small percentage increase either the system size or the percentage.')


    rng = np.random.default_rng(seed) 
    random_sites = rng.choice(S.shape[0], size=N_d, replace=False)
    random_sites = np.array(random_sites, order='C', dtype=np.int32)
    
    return np.delete(S,random_sites,axis=0)
def vacancies_even(S,n,seed=None):
    """
    Def:
        Delete a percentage of the site of an array of sites.
    Inputs:
        W : Array of sites in the order S[:,i] coordinate i of all the sites.
        n : Percetag eof sites that you want to remove.
        seed : Random seed
    Outputs:
        S_1 : Array of sites with already deleted sites. 
    """

    N_d=int(S.shape[0]*n/100)

    if N_d==0:
        print('You have insert a very small percentage increase either the system size or the percentage.')


    rng = np.random.default_rng(seed) 
    random_sites = 2*rng.choice(S.shape[0]//2, size=N_d, replace=False)
    random_sites = np.array(random_sites, order='C', dtype=np.int32)
    
    return np.delete(S,random_sites,axis=0)