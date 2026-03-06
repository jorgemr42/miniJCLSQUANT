
import scipy as sci
import numpy as np
from time import perf_counter
import copy
from tqdm import tqdm
from math import pi,sqrt
from jclsquant.kernel_and_moments import *
from jclsquant.recurrence_functions import *
from jclsquant.minimizer_thermal import *
from jclsquant.ell_matrix import *

from jclsquant.modifiers import *
from jclsquant.cython_modules.blas_funs import dot,axpy2,vdot # type: ignore
from .constants import *


####### Functions auxiliaries of operators
def random_vector_generator(shape_H, seed=None):
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(0, 2*np.pi, shape_H))

####### Operators

## Density of states (dos)
def kpm_dos(H,M=None,random_vector=None):
    """
    Def:
        Density of states using KPM.
    Inputs:
        H : Hamiltonian in ELL format.
        M : (hidden) Number of moments.
        random_vector : (hidden) random vector , by default is computed as expected.
    Outputs:
        delta_mat: Array of shape (len(Ef_vec),2) where [:,0] are the unormalized Fermi energies and [:,1] the DOS.
    """
    if M is None:
        M=int(np.sqrt(H.shape[0]))
    Ef_vec=np.linspace(H.bounds[0]+1e-3,H.bounds[1]-1e-3,2*M+1)
    delta_mat=np.zeros((2*M+1,2))
    delta_mat[:,0]=Ef_vec

    if random_vector is None:
        random_vector=random_vector_generator(H.shape[0])
    
    H_kpm=H.modifier(modifier_bounds)
    delta_vec=rec_A_tab(M,H_kpm,random_vector)    
    
    kernel=JacksonKernel(M).astype(np.complex128)

    for i in range(len(Ef_vec)):
        delta_mat[i,1]=np.real(dot(1+0j,moments_delta(Ef_vec[i]/(0.5*(H.bounds[1]-H.bounds[0]))-(0.5*(H.bounds[1]+H.bounds[0]))/(0.5*(H.bounds[1]-H.bounds[0])),M).astype(np.complex128)*kernel,delta_vec,M,H.n_threads))

    return delta_mat


## n with different vectors left and right

def kpm_n_dos_n(H,M=None,random_vector_l=None,random_vector_r=None,bounds=True,proyector=None):
    """
    Def:
        Density of states using KPM with a different random vector for the ket and the bra vector.
    Inputs:
        H : Hamiltonian in ELL format.
        M : (hidden) Number of moments.
        random_vector : (hidden) random vector multiplied by the left , by default is computed as expected.
        random_vector : (hidden) random vector multiplied by the right, by default is computed as expected.
        bounds : (hidden) If you want to reescalate the Hamiltonian inside the funciton by default is True .
        proyector : (hidden) If you want to compute the right random vector by an ELL matrix before doing the DOS, by default is None .
    Outputs:
        delta_mat: Array of shape (len(Ef_vec),2) where [:,0] are the unormalized Fermi energies and [:,1] the DOS.
    """
    if M is None:
        M=int(np.sqrt(H.shape[0]))

    Ef_vec=np.linspace(H.bounds[0]+1e-3,H.bounds[1]-1e-3,2*M+1)


    n_mat=np.zeros((2*M+1,2))
    n_mat[:,0]=Ef_vec
    

    dos_n_mat=np.zeros((2*M+1,2))
    dos_n_mat[:,0]=Ef_vec
    

    if random_vector_l is None and random_vector_r is None:
        random_vector_r=random_vector_generator(H.shape[0])
        random_vector_l=np.copy(random_vector_r)

    if proyector is not None:
        random_vector_l_2=np.zeros(len(random_vector_l),dtype=np.complex128,order='C')

        proyector.dot(1.0+0j,0+0j,random_vector_l,random_vector_l_2)
        # vec_mul(1,random_vector_l,proyector,random_vector_l_2,len(random_vector_l_2),H.n_threads)
    
        if bounds==True:

            H_kpm=H.modifier(modifier_bounds)
            dos_n_vec,tr_n_vec=rec_A_tab2v_2(M,H_kpm,random_vector_l_2,random_vector_r)

        elif bounds==False:

            dos_n_vec,tr_n_vec=rec_A_tab2v_2(M,H,random_vector_l_2,random_vector_r)
    else:
        if bounds==True:

            H_kpm=H.modifier(modifier_bounds)
            dos_n_vec,tr_n_vec=rec_A_tab2v_2(M,H_kpm,random_vector_l,random_vector_r)

        elif bounds==False:

            dos_n_vec,tr_n_vec=rec_A_tab2v_2(M,H,random_vector_l,random_vector_r)


    for i in range(len(Ef_vec)):

        dos_n_mat[i,1]=np.real(dot(1+0j,moments_delta(Ef_vec[i]/(0.5*(H.bounds[1]-H.bounds[0]))-(0.5*(H.bounds[1]+H.bounds[0]))/(0.5*(H.bounds[1]-H.bounds[0])),M).astype(np.complex128)*JacksonKernel(M).astype(np.complex128),dos_n_vec,M,H.n_threads))
        n_mat[i,1]=dos_n_mat[i,1]/np.real(dot(1+0j,moments_delta(Ef_vec[i]/(0.5*(H.bounds[1]-H.bounds[0]))-(0.5*(H.bounds[1]+H.bounds[0]))/(0.5*(H.bounds[1]-H.bounds[0])),M).astype(np.complex128)*JacksonKernel(M).astype(np.complex128),tr_n_vec,M,H.n_threads))

    return n_mat,dos_n_mat




def kpm_rho_neq(H,t_vec=None,tau=None,modifier_id=None,modifier_params=None,Temp=None,mu=None,observale_list=None,M=None,random_vector=None,proyector=None):
    """
    Def:
        Time-evolution of the density matrix even if tau!=0 it does not include the relaxation.
    Inputs:
        H: Hamiltonian in ELL.
        t_vec : (hidden) time vector in fs.
        tau : (hidden) Relaxation time in fs by default in 0 even if !=0 this function does not include the minimization .
        modifier_id : (hidden) Choosing the light type linear , circle , linear packed , circle packed . 
        modifier_params : (hidden) Parameters that enter into the modifier .
        Temp : (hidden) Temperature in Kelvin .
        mu : (hidden) Chemical potential by default set to 0 eV .
        observable_list : a list of the following form [[observable_id,n_meass,observable_params],...]
        M : (hidden) number of moments of the KPM expansion by default set to int(sqrt(N)).
        random_vector: (hidden) random vector , by default is computed as expected.
        proyector: (hidden) if you want to multiply before doing the n_dos by a proyector, by defualt is None.
    Outputs:
        If you have introduced n operator:
            n_mat: number of carriers with shape (n_meass,2*M_n+1,2) 
            dos_n_mat: Non-equilibrium density of states with shape (n_meass,2*M_n+1,2).
        If no operator is introduced:
            F_t0 : Density matrix at t=0 .
            U_t0 : Time evolution operator at t=0 .
            F : Density matrix at t=end .
            U : Time evolution operator at t=end .
    """

    if M is None:
        M=int(np.sqrt(H.shape[0]))
    
    if mu is None:
        print('No Fermi energy has been included by default is set to 0 .')
        mu=0
    if random_vector is None:
        random_vector=random_vector_generator(H.shape[0])

    if t_vec is None:
        print('No time vector has been inserted by default is set from 0 to 5*tau_phi with tau_phi=hbar*M/(pi*bounds[1]) with tau_phi/100 steps')
        t_vec=np.linspace(0,5*hbar_fs*M/(pi*H.bounds[1]),500)
    if Temp is None:
        print('No temperature inserted by default is set to zero.')
        moments_kernel_FD=moments_FD_T0(0,M)*JacksonKernel(M)
    if modifier_params is None:
        print('You must introduce an array that contains the elements necesary to apply the time evolution to the hamiltonian .')
        return
    if modifier_id is None:
        print('You must introduce an string with one of the following : linear , circle , linear packed , circle packed .')
        return

    ### Moments for time evolution
    delta_t=t_vec[1]
    # delta_t_kpm=delta_t*(0.5*(H.bounds[1]-H.bounds[0]))/hbar_fs
    delta_t_kpm=delta_t*H.bounds[1]/hbar_fs
    moments_U_vec=moments_U(delta_t_kpm,M)

    m=0
    while np.abs(moments_U_vec[m])>1e-15:
        m+=1
    M2=m

    moments_U_vec=moments_U(delta_t_kpm,M2)

    ### Reescalating the Hamiltonian and moments for the FD
    H_kpm=H.modifier(modifier_bounds)
    

    if Temp==0:
        moments_kernel_FD=moments_FD_T0((mu-(0.5*(H.bounds[1]+H.bounds[0])))/(0.5*(H.bounds[1]-H.bounds[0])),M)*JacksonKernel(M)
    else:
        # moments_kernel_FD=moments_FD_T((mu-(0.5*(H.bounds[1]+H.bounds[0])))/(0.5*(H.bounds[1]-H.bounds[0])),((Temp*kb))/(0.5*(H.bounds[1]-H.bounds[0])),M)*JacksonKernel(M)
        moments_kernel_FD=moments_FD_T((mu)/(0.5*(H.bounds[1]-H.bounds[0])),((Temp*kb))/(0.5*(H.bounds[1]-H.bounds[0])),M)*JacksonKernel(M)

    ### Unwrap of n computation 
    k_n=-1

    for i in range(len(observale_list)):

        if observale_list[i][0] == 'n':
            
            n_meass=observale_list[i][1]
            M_n=observale_list[i][2]

            if n_meass>1:
                n_meass_vec=np.linspace(0,len(t_vec)-1,n_meass,dtype=int)
            else:
                n_meass_vec=np.array([len(t_vec)-1],dtype=int)

            k_n=0
            
            n_mat=np.zeros((n_meass,2*M_n+1,2))
            dos_n_mat=np.zeros((n_meass,2*M_n+1,2))

        else:
            k_n=-1


    ## Time evolution
    H_time=H_kpm.deep_copy()

    aux=np.zeros(len(random_vector),dtype=np.complex128)

    for i in range(len(t_vec)):

        H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec[i])
        
        if i==0:

            U=np.copy(random_vector)
            F=rec_A_vec(M,H_time,moments_kernel_FD,random_vector)
            F_t0=F
            U_t0=U
            if tau!=0:
                H_time.dot(1+0j,0+0j,F,aux)
                E_0=2*H_time.bounds[1]*np.real(vdot(1+0j,random_vector,aux,len(random_vector),H_time.n_threads))
                moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)
        
        else:
            if tau==0:

                U=rec_A_vec(M2,H_time,moments_U_vec,U)
                F=rec_A_vec(M2,H_time,moments_U_vec,F)

            else:


                U=rec_A_vec(M2,H_time,moments_U_vec,U)
                F=rec_A_vec(M2,H_time,moments_U_vec,F)


                if i%(len(t_vec)//20)==0:
                    H_time.dot(1+0j,0+0j,F,aux)
                    E_t=2*H_time.bounds[1]*np.real(vdot(1+0j,U,aux,len(U),H_time.n_threads))
                    deltaE=(E_t-E_0)/H.shape[0]

                    mu,Temp=minimization(deltaE,dos_equil,[mu,Temp*kb],niter=10000)

                    print('Chemical potential : '+str(mu)+' [eV]')
                    print('Temperature : '+str(Temp)+' [K]')

                    E_0=E_t
                    moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)


                axpy2((delta_t/tau),(1-delta_t/tau),rec_A_vec(M,H_time,moments_kernel_FD_equil,U),F,len(F),H.n_threads)
                

    
        if k_n >= 0:
            if i == n_meass_vec[k_n]:
                print('Computing dos : '+str(k_n))
                if k_n==0 and n_meass>1:
                    _,dos_f=kpm_n_dos_n(H_time,M_n,F,F,False)
                    _,dos_u=kpm_n_dos_n(H_time,M_n,U,U,False)

                    n_mat[k_n,:],dos_n_mat[k_n,:]=kpm_n_dos_n(H_kpm,M_n,U,F,False,proyector)
                else:
                    _,dos_f=kpm_n_dos_n(H_time,M_n,F,F,False)
                    _,dos_u=kpm_n_dos_n(H_time,M_n,U,U,False)
  

                    n_mat[k_n,:],dos_n_mat[k_n,:]=kpm_n_dos_n(H_time,M_n,U,F,False,proyector)
                
                k_n+=1


    if k_n>=0 :
        return n_mat,dos_n_mat
    else:
        return F_t0,U_t0,F,U
    






