import scipy as sci
import numba as numba
import numpy as np
from time import perf_counter
import copy
import tqdm

def lattice_hexagonal(N):
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
    