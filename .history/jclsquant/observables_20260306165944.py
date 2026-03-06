
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


#######
# Cambiar observables a que les metas el hamiltoniano te calcule los bounds y se los ponga.
# Que tu tengas que introducir el hamiltoniano normal



### Functions auxiliaries to operators
def random_vector_generator(shape_H, seed=None):
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(0, 2*np.pi, shape_H))

### Operators

## Density of states (dos)
def kpm_dos(H,M=None,random_vector=None):
    """
    Def:
        Mean value of the DOS making the table and the sum the moments
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
        Mean value of the DOS making the table and the sum the moments
    Inputs:
        H : Hamiltonian in ELL format.
        M : (hidden) Number of moments.
        random_vector : (hidden) random vector , by default is computed as expected.
        bounds : (hidden) If you want to reescalate the Hamiltonian inside the funciton by default is True .
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



### Time dependent conductivity
def kpm_sigma_time_nequil(H,direction_1=None,direction_2=None,t_vec=None,U_F_t0=None,U_t0=None,M=None,bounds=True,velocities=None):
    """
    Def:
        Time-dependent conducitivity
    Inputs:
        H: Hamiltonian in CSR or ELL.
        direction_1 : (hidden) Direction of the index 'x'/0, 'y'/1 or 'z'/2.
        direction_2 : (hidden) Direction of the index 'x'/0, 'y'/1 or 'z'/2.
        t_vec : (hidden) time vector in fs.
        M : (hidden) number of moments of the KPM expansion by default set to int(sqrt(N)).
        random_vector: (hidden) random vector , by default is computed as expected.
        
    Outputs:
        delta_vec: Complex array of shape len(t_vec) .
    Some notes:
        tau_broadening=M*hbar/(pi*bounds_arra[1]) if bounds are symmetrical.
        To obtain the conductivity in units of e**2/h you should multiply the result by 2*pi/(Omega*hbar)
    """
    ### The checks and default definitions
    if M is None:
        M=int(np.sqrt(H.shape[0]))
    

    if t_vec is None:
        print('No time vector has been inserted by default is set from 0 to 5*tau_phi with tau_phi=hbar*M/(pi*bounds[1]) with tau_phi/100 steps')
        t_vec=np.linspace(0,5*hbar_fs*M/(pi*H.bounds[1]),500)
    if U_F_t0 is None or U_t0 is None:
        print('You should introduce U_F and U for computing this observable .')

    ### Moments for time evolution
    delta_t=t_vec[1]
    delta_t_kpm=delta_t*H.bounds[1]/hbar_fs
    moments_U_vec=moments_U(delta_t_kpm,M)
    m=0
    while np.abs(moments_U_vec[m])>1e-15 and m<(M-1):
        m+=1
    M2=m
    
    moments_U_vec=moments_U(delta_t_kpm,M2)

    ### Reescalating the Hamiltonian and moments for the FD
    if bounds==True:

        H_kpm=H.modifier(modifier_bounds)
        V1 = H.modifier(modifier_velocity,direction_1)
        V2_kpm = H.modifier(modifier_velocity_bounds,direction_2)
        V1_r2=H.modifier(modifier_velocity_r_comm,direction_1,direction_2)

    elif bounds==False:
        
        H_kpm=H.deep_copy()
        H2=H.modifier(modifier_bounds_inverse)
        V1 = H2.modifier(modifier_velocity,direction_1)
        V2_kpm = H2.modifier(modifier_velocity_bounds,direction_2)
        V1_r2=H2.modifier(modifier_velocity_r_comm,direction_1,direction_2)

    ## Necessary arrays for the loop . 
    sigma_vec=np.zeros((len(t_vec)),dtype=np.complex128)

    aux=np.zeros(H.shape[0],dtype=np.complex128,order='C')
    aux_2=np.zeros(H.shape[0],dtype=np.complex128,order='C')
    aux_vec=np.zeros(H.shape[0],dtype=np.complex128,order='C')
        
    for j in (range(len(t_vec))):
        if j==0:
            U_F=U_F_t0            
            mean_x=U_t0
            U=U_t0
            
            V1_r2.dot(1.0+0j,0+0j,U,aux_2)

            sigma_vec[j]=1j*(vdot(1+0j,U_F,aux_2,H.shape[0],H.n_threads))
        elif j==1:
            mean_x=rec_com_A_vec(M2,H_kpm,V2_kpm,moments_U_vec,U)
            
            U_F=rec_A_vec(M2,H_kpm,moments_U_vec,U_F)

            U=rec_A_vec(M2,H_kpm,moments_U_vec,U)

            V1.dot(1.0+0j,0+0j,mean_x,aux)
            term=vdot(1+0j,U_F,aux,H.shape[0],H.n_threads)

            V1_r2.dot(1.0+0j,0+0j,U,aux_2)

            sigma_vec[j]=1j*(-term+np.conj(term)+vdot(1+0j,U_F,aux_2,H.shape[0],H.n_threads))
        else:

            mean_x=rec_A_vec(M2,H_kpm,moments_U_vec,mean_x)
            
            axpy2(1+0j,1+0j,rec_com_A_vec(M2,H_kpm,V2_kpm,moments_U_vec,U),mean_x, len(U_F_t0),H_kpm.n_threads)
            
            U=rec_A_vec(M2,H_kpm,moments_U_vec,U)
            U_F=rec_A_vec(M2,H_kpm,moments_U_vec,U_F)
            


            V1.dot(1.0+0j,0+0j,mean_x,aux)
            term=vdot(1+0j,U_F,aux,H.shape[0],H.n_threads)

            V1_r2.dot(1.0+0j,0+0j,U,aux_2)

            sigma_vec[j]=1j*(-term+np.conj(term)+vdot(1+0j,U_F,aux_2,H.shape[0],H.n_threads))
        
    return sigma_vec


### Time dependent conductivity
def kpm_sigma_time_nequil2(H,direction_1=None,direction_2=None,t_vec=None,U_F_t0=None,U_t0=None,M=None,bounds=True):
    """
    Def:
        Time-dependent conducitivity
    Inputs:
        H: Hamiltonian in CSR or ELL.
        direction_1 : (hidden) Direction of the index 'x'/0, 'y'/1 or 'z'/2.
        direction_2 : (hidden) Direction of the index 'x'/0, 'y'/1 or 'z'/2.
        t_vec : (hidden) time vector in fs.
        M : (hidden) number of moments of the KPM expansion by default set to int(sqrt(N)).
        random_vector: (hidden) random vector , by default is computed as expected.
        
    Outputs:
        delta_vec: Complex array of shape len(t_vec) .
    Some notes:
        tau_broadening=M*hbar/(pi*bounds_arra[1]) if bounds are symmetrical.
        To obtain the conductivity in units of e**2/h you should multiply the result by 2*pi/(Omega*hbar)
    """
    ### The checks and default definitions
    if M is None:
        M=int(np.sqrt(H.shape[0]))
    

    if t_vec is None:
        print('No time vector has been inserted by default is set from 0 to 5*tau_phi with tau_phi=hbar*M/(pi*bounds[1]) with tau_phi/100 steps')
        t_vec=np.linspace(0,5*hbar_fs*M/(pi*H.bounds[1]),500)
    if U_F_t0 is None or U_t0 is None:
        print('You should introduce U_F and U for computing this observable .')

    ### Moments for time evolution
    delta_t=t_vec[1]
    delta_t_kpm=delta_t*H.bounds[1]/hbar_fs
    moments_U_vec=moments_U(delta_t_kpm,M)
    m=0
    while np.abs(moments_U_vec[m])>1e-15:
        m+=1
    M2=m
    moments_U_vec=moments_U(delta_t_kpm,M2)

    ### Reescalating the Hamiltonian and moments for the FD
    if bounds==True:

        H_kpm=H.modifier(modifier_bounds)
        V1 = H.modifier(modifier_velocity,direction_1)
        V2_kpm = H.modifier(modifier_velocity_bounds,direction_2)
        V1_r2=H.modifier(modifier_velocity_r_comm,direction_1,direction_2)

    elif bounds==False:
        
        H_kpm=H.deep_copy()
        H2=H.modifier(modifier_bounds_inverse)
        V1 = H2.modifier(modifier_velocity,direction_1)
        V2_kpm = H2.modifier(modifier_velocity_bounds,direction_2)
        V1_r2=H2.modifier(modifier_velocity_r_comm,direction_1,direction_2)
        



    ## Necessary arrays for the loop . 
    sigma_vec=np.zeros((M2,len(t_vec)),dtype=np.complex128)

    aux=np.zeros(H.shape[0],dtype=np.complex128,order='C')
    aux_2=np.zeros(H.shape[0],dtype=np.complex128,order='C')
    aux_vec=np.zeros(H.shape[0],dtype=np.complex128,order='C')
    
    mean_x=np.zeros((M2,len(U_t0)),dtype=np.complex128)

    for j in (range(len(t_vec))):
        if j==0:
            U_F=U_F_t0
            for m in range(M2):
                mean_x[m,:]=U_t0/M2
            mean_x_2=U_t0
            U=U_t0
        else:

            for m in range(M2):

                mean_x[m,:]=rec_A_vec(M2,H_kpm,moments_U_vec,mean_x[m,:])
            mean_x+=rec_com_A_vec_tab(M2,H_kpm,V2_kpm,moments_U_vec,U)

            U=rec_A_vec(M2,H_kpm,moments_U_vec,U)
            U_F=rec_A_vec(M2,H_kpm,moments_U_vec,U_F)

        for m in range(M2):
            V1.dot(1.0+0j,0+0j,mean_x[m,:],aux)
            term=vdot(1+0j,U_F,aux,H.shape[0],H.n_threads)

            V1_r2.dot(1.0+0j,0+0j,U,aux_2)

            sigma_vec[m,j]=1j*(-term+np.conj(term)+vdot(1+0j,U_F,aux_2,H.shape[0],H.n_threads))
    return sigma_vec


### Time dependent conductivity
def kpm_sigma_time_nequil3(H,direction_1=None,direction_2=None,t_vec=None,U_F_t0=None,U_t0=None,M=None,bounds=True,S=None):
    """
    Def:
        Time-dependent conducitivity
    Inputs:
        H: Hamiltonian in CSR or ELL.
        direction_1 : (hidden) Direction of the index 'x'/0, 'y'/1 or 'z'/2.
        direction_2 : (hidden) Direction of the index 'x'/0, 'y'/1 or 'z'/2.
        t_vec : (hidden) time vector in fs.
        M : (hidden) number of moments of the KPM expansion by default set to int(sqrt(N)).
        random_vector: (hidden) random vector , by default is computed as expected.
        
    Outputs:
        delta_vec: Complex array of shape len(t_vec) .
    Some notes:
        tau_broadening=M*hbar/(pi*bounds_arra[1]) if bounds are symmetrical.
        To obtain the conductivity in units of e**2/h you should multiply the result by 2*pi/(Omega*hbar)
    """
    ### The checks and default definitions
    if M is None:
        M=int(np.sqrt(H.shape[0]))
    

    if t_vec is None:
        print('No time vector has been inserted by default is set from 0 to 5*tau_phi with tau_phi=hbar*M/(pi*bounds[1]) with tau_phi/100 steps')
        t_vec=np.linspace(0,5*hbar_fs*M/(pi*H.bounds[1]),500)
    if U_F_t0 is None or U_t0 is None:
        print('You should introduce U_F and U for computing this observable .')

    ### Moments for time evolution
    delta_t=t_vec[1]
    delta_t_kpm=delta_t*H.bounds[1]/hbar_fs
    moments_U_vec=moments_U(delta_t_kpm,M)
    m=0
    while np.abs(moments_U_vec[m])>1e-15:
        m+=1
    M2=m
    moments_U_vec=moments_U(delta_t_kpm,M2)

    ### Reescalating the Hamiltonian and moments for the FD
    if bounds==True:

        H_kpm=H.modifier(modifier_bounds)
        V1 = H.modifier(modifier_velocity,direction_1)
        V2_kpm = H.modifier(modifier_velocity_bounds,direction_2)
        V1_r2=H.modifier(modifier_velocity_r_comm,direction_1,direction_2)

    elif bounds==False:
        
        H_kpm=H.deep_copy()
        H2=H.modifier(modifier_bounds_inverse)
        V1 = H2.modifier(modifier_velocity,direction_1)
        V2_kpm = H2.modifier(modifier_velocity_bounds,direction_2)
        V2 = H2.modifier(modifier_velocity,direction_2)
        V1_r2=H2.modifier(modifier_velocity_r_comm,direction_1,direction_2)
        

    # r_vec=position_operator_sum(H_kpm,V2,M,U_t0)

    if direction_2=='x':
        L=np.max(S[:,0])-np.min(S[:,0])
        r=sci.sparse.diags(L*(np.exp(2*np.pi*1j*S[:,0]/L)-1)/(2*np.pi*1j), offsets=0, format="csr")
        r_dag=sci.sparse.diags(L*(np.exp(2*np.pi*-1j*S[:,0]/L)-1)/(2*np.pi*-1j), offsets=0, format="csr")
        r_ell=ell_matrix(r)
        r_dag_ell=ell_matrix(r_dag)

    else:
        L=np.max(S[:,1])-np.min(S[:,1])
        r=sci.sparse.diags(L*(np.exp(2*np.pi*1j*S[:,1]/L)-1)/(2*np.pi*1j), offsets=0, format="csr")
        r_dag=sci.sparse.diags(L*(np.exp(2*np.pi*-1j*S[:,1]/L)-1)/(2*np.pi*-1j), offsets=0, format="csr")
        # r_vec=r.dot(r_vec)
        r_ell=ell_matrix(r)
        r_dag_ell=ell_matrix(r_dag)
        
    #


    ## Necessary arrays for the loop . 
    sigma_vec=np.zeros((len(t_vec)),dtype=np.complex128)

    aux=np.zeros(H.shape[0],dtype=np.complex128,order='C')
    U=np.zeros(H.shape[0],dtype=np.complex128,order='C')
    U_dag=np.zeros(H.shape[0],dtype=np.complex128,order='C')

    term_1=0+0j
    term_2=0+0j
    for j in (range(len(t_vec))):
        if j==0:
            U_F=U_F_t0     
            # U=r_vec       
            r_ell.dot(1+0j,0+0j,U_t0,U)
            r_dag_ell.dot(1+0j,0+0j,np.conj(U_t0),U_dag)
        else:

            U=rec_A_vec(M2,H_kpm,moments_U_vec,U)
            U_dag=rec_A_vec(M2,H_kpm,np.conj(moments_U_vec),U_dag)
            U_F=rec_A_vec(M2,H_kpm,moments_U_vec,U_F)

            V1.dot(1.0+0j,0+0j,U_F,aux)

            # term_1=vdot(1+0j,U,aux,H.shape[0],H.n_threads)
            term_1=dot(1+0j,U_dag,aux,H.shape[0],H.n_threads)

            V1.dot(1.0+0j,0+0j,U,aux)

            term_2=vdot(1+0j,U_F,aux,H.shape[0],H.n_threads)

        sigma_vec[j]=1j*(term_2)
    return sigma_vec


def kpm_rho_neq(H,t_vec=None,tau=None,modifier_id=None,modifier_params=None,Temp=None,mu=None,observale_list=None,M=None,random_vector=None,dos_equil=None,proyector=None):
    """
    Def:
        Time-evolution of the density matrixeven if tau!=0 it does not include the relaxation.
    Inputs:
        H: Hamiltonian in CSR or ELL.
        t_vec : (hidden) time vector in fs.
        tau : (hidden) Relaxation time in fs by default in 0 even if !=0 this function does not include the minimization .
        modifier_id : (hidden) Choosing the light type linear , circle , linear packed , circle packed . 
        modifier_params : (hidden) Parameters that enter into the modifier .
        Temp : (hidden) Temperature in Kelvin .
        Temp_equil : (hidden) Temperature in Kelvin towards the system will relax only matters if tau!=0 .
        Ef : (hidden) Fermi velocity by default set to 0 eV .
        observable_list : a list of the following form [[observable_id,n_meass,observable_params],...]
        M : (hidden) number of moments of the KPM expansion by default set to int(sqrt(N)).
        random_vector: (hidden) random vector , by default is computed as expected.
        
    Outputs:
        F_t0 : Density matrix at t=0 .
        U_t0 : Time evolution operator at t=0 .
        F : Density matrix at t=end .
        F : Time evolution operator at t=end .
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
    k_sigma=-1
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

        elif observale_list[i][0] == 'sigma_nequil':
            sigma_meass=observale_list[i][1]
            direction_1=observale_list[i][2]
            direction_2=observale_list[i][3]
            t_vec_sigma=observale_list[i][4]
            M_sigma=observale_list[i][5]

            if sigma_meass>1:
                sigma_meass_vec=np.linspace(0,len(t_vec)-1,sigma_meass,dtype=int)
            else:
                sigma_meass_vec=np.array([len(t_vec)-1],dtype=int)
            
            k_sigma=0
            sigma_vec=np.zeros((sigma_meass,len(t_vec_sigma)),dtype=np.complex128)
    
        else:
            k_n=-1
            k_sigma=-1
            print('There is no observable selected the function will return F(t=0) , U(t=0) , F(t=end) and U(t=end) .')


    ## Time evolution
    H_time=H_kpm.deep_copy()

    aux=np.zeros(len(random_vector),dtype=np.complex128)
    for i in tqdm(range(len(t_vec))):

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
        if k_sigma >= 0:
            if i == sigma_meass_vec[k_sigma]:
                print('Computing sigma :'+str(k_sigma))
                if i==0:
                    sigma_vec[k_sigma,:]=kpm_sigma_time_nequil(H,direction_1,direction_2,t_vec_sigma,F,U,M_sigma,bounds=True)

                else:
                    sigma_vec[k_sigma,:]=kpm_sigma_time_nequil(H_time,direction_1,direction_2,t_vec_sigma,F,U,M_sigma,bounds=False)
                
                k_sigma+=1


    if k_n>=0 and k_sigma<0:
        return n_mat,dos_n_mat
    elif k_n<0 and k_sigma>=0:
        return sigma_vec
    elif k_n>=0 and k_sigma>=0:
        print('Final Temperature : '+str(Temp)+' [K]')
        print('Final Chemical potential : '+str(mu)+' [eV]')
        return n_mat,dos_n_mat,sigma_vec
    else:
        return F_t0,U_t0,F,U
    











def kpm_total_energy(H=None,mu=None,Temp=None,M=None,random_vector=None):
    
    if Temp==0:
        moments_kernel_FD=moments_FD_T0(mu/H.bounds[1],M)*JacksonKernel(M)
    else:
        moments_kernel_FD=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)
    
    H_kpm=H.modifier(modifier_bounds)

    ## Part of the Hamiltonian
    H_vec=np.zeros(len(random_vector),dtype=np.complex128,order='C')
    H_kpm.dot(1+0j,0+0j,random_vector,H_vec)

    ## Part of the density matrix
    rho=rec_A_vec(M,H_kpm,moments_kernel_FD,random_vector)


    return vdot(1+0j,rho,H_vec,len(random_vector),H.n_threads)



##################### Position operator #####################


def position_operator_mas(H_kpm=None,V=None,M=None,random_vector=None,alpha_integral=0.99):
    # H=H_kpm_0.modifier(modifier_bounds_inverse)
    # H.bounds=H.bounds/alpha_integral
    # H_kpm=H.modifier(modifier_bounds)


    E_vec=np.linspace(-alpha_integral,alpha_integral,2*M)


    moments_table_big,_=table_G_delta(M,alpha_integral,H_kpm.bounds[1],H_kpm.n_threads)

    moments_table=sci.integrate.simpson(moments_table_big,x=E_vec,axis=2)


    T_delta_0=np.zeros(len(random_vector),dtype=np.complex128,order='C')
    T_delta_1=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    r_vec=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    T_delta_v=np.zeros(len(random_vector),dtype=np.complex128,order='C')


    
    T_delta_0=np.copy(random_vector,order='C')
    V.dot(1+0j,0+0j,T_delta_0,T_delta_v)

    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,0],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    H_kpm.dot(1+0j,0+0j,random_vector,T_delta_1)
    V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
   
    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,1],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    for n in range(2,M):
        
        H_kpm.dot(2.0+0j,-1+0j,T_delta_1,T_delta_0)
        temp = T_delta_0
        T_delta_0 = T_delta_1
        T_delta_1 = temp 

        V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
        axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,n],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)


    return r_vec




def position_operator_min(H_kpm=None,V=None,M=None,random_vector=None,alpha_integral=0.99):
    # H=H_kpm_0.modifier(modifier_bounds_inverse)
    # H.bounds=H.bounds/alpha_integral
    # H_kpm=H.modifier(modifier_bounds)


    E_vec=np.linspace(-alpha_integral,alpha_integral,2*M)


    _,moments_table_big=table_G_delta(M,alpha_integral,H_kpm.bounds[1],H_kpm.n_threads)

    moments_table=sci.integrate.simpson(moments_table_big,x=E_vec,axis=2)


    T_delta_0=np.zeros(len(random_vector),dtype=np.complex128,order='C')
    T_delta_1=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    r_vec=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    T_delta_v=np.zeros(len(random_vector),dtype=np.complex128,order='C')


    
    T_delta_0=np.copy(random_vector,order='C')
    V.dot(1+0j,0+0j,T_delta_0,T_delta_v)

    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,0],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    H_kpm.dot(1+0j,0+0j,random_vector,T_delta_1)
    V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
   
    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,1],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    for n in range(2,M):
        
        H_kpm.dot(2.0+0j,-1+0j,T_delta_1,T_delta_0)
        temp = T_delta_0
        T_delta_0 = T_delta_1
        T_delta_1 = temp 

        V.dot(1+0j,0+0j,T_delta_1,T_delta_v)

        axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,n],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)


    return r_vec

def position_operator_mas_2(H_kpm=None,V=None,M=None,random_vector=None,alpha_integral=0.99):
    start=perf_counter()
    # H=H_kpm_0.modifier(modifier_bounds_inverse)
    # H.bounds=H.bounds/alpha_integral
    # H_kpm=H.modifier(modifier_bounds)

    E_vec=np.linspace(-alpha_integral,alpha_integral,1000)


    moments_table_big=np.zeros((M,M,len(E_vec)),dtype=np.complex128)
    moments_G_vec=np.zeros(M,dtype=np.complex128)
    moments_delta_vec=np.zeros(M,dtype=np.complex128)
    
    for i in range(len(E_vec)):
        moments_G_vec=((moments_Gmas_2(E_vec[i],np.pi/M,M)))*JacksonKernel(M)     
        moments_delta_vec=moments_delta_2(E_vec[i],np.pi/M,M)*JacksonKernel(M)
        for n in range(M):
            for m in range(M):
                moments_table_big[m,n,i]=moments_G_vec[m]*moments_delta_vec[n]

    
    moments_table=sci.integrate.simpson(moments_table_big,x=E_vec,axis=2)


    T_delta_0=np.zeros(len(random_vector),dtype=np.complex128,order='C')
    T_delta_1=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    r_vec=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    T_delta_v=np.zeros(len(random_vector),dtype=np.complex128,order='C')


    
    T_delta_0=np.copy(random_vector,order='C')
    V.dot(1+0j,0+0j,T_delta_0,T_delta_v)
    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,0],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    H_kpm.dot(1+0j,0+0j,random_vector,T_delta_1)
    V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
   
    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,1],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    for n in range(2,M):
        
        H_kpm.dot(2.0+0j,-1+0j,T_delta_1,T_delta_0)
        temp = T_delta_0
        T_delta_0 = T_delta_1
        T_delta_1 = temp 

        V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
        axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,n],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)


    return 1j*r_vec/(H_kpm.bounds[1])



def position_operator_sum(H_kpm=None,V=None,M=None,random_vector=None,alpha_integral=0.99):
    # H=H_kpm_0.modifier(modifier_bounds_inverse)
    # H.bounds=H.bounds/alpha_integral
    # H_kpm=H.modifier(modifier_bounds)


    E_vec=np.linspace(-alpha_integral,alpha_integral,2*M)


    moments_table_big=np.zeros((M,M,len(E_vec)),dtype=np.complex128)
    moments_G_vec=np.zeros(M,dtype=np.complex128)
    moments_delta_vec=np.zeros(M,dtype=np.complex128)
    
    for i in range(len(E_vec)):
        moments_G_vec_mas=((moments_Gmas(E_vec[i],M)))*JacksonKernel(M)     
        moments_G_vec_min=((moments_Gmin(E_vec[i],M)))*JacksonKernel(M)     
        moments_delta_vec=moments_delta(E_vec[i],M)*JacksonKernel(M)
        for n in range(M):
            for m in range(M):
                moments_table_big[m,n,i]=moments_G_vec_mas[m]*moments_delta_vec[n]-moments_G_vec_min[n]*moments_delta_vec[m]

    
    moments_table=sci.integrate.simpson(moments_table_big,x=E_vec,axis=2)


    T_delta_0=np.zeros(len(random_vector),dtype=np.complex128,order='C')
    T_delta_1=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    r_vec=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    T_delta_v=np.zeros(len(random_vector),dtype=np.complex128,order='C')


    
    T_delta_0=np.copy(random_vector,order='C')
    V.dot(1+0j,0+0j,T_delta_0,T_delta_v)

    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,0],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    H_kpm.dot(1+0j,0+0j,random_vector,T_delta_1)
    V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
   
    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,1],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    for n in range(2,M):
        
        H_kpm.dot(2.0+0j,-1+0j,T_delta_1,T_delta_0)
        temp = T_delta_0
        T_delta_0 = T_delta_1
        T_delta_1 = temp 

        V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
        axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,n],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)


    return 1j*0.5*r_vec/(H_kpm.bounds[1])

def position_operator_rest(H_kpm=None,V=None,M=None,random_vector=None,alpha_integral=0.99):
    # H=H_kpm_0.modifier(modifier_bounds_inverse)
    # H.bounds=H.bounds/alpha_integral
    # H_kpm=H.modifier(modifier_bounds)


    E_vec=np.linspace(-alpha_integral,alpha_integral,2*M)


    moments_table_big=np.zeros((M,M,len(E_vec)),dtype=np.complex128)
    moments_G_vec=np.zeros(M,dtype=np.complex128)
    moments_delta_vec=np.zeros(M,dtype=np.complex128)
    
    for i in range(len(E_vec)):
        moments_G_vec_mas=((moments_Gmas(E_vec[i],M)))*JacksonKernel(M)     
        moments_G_vec_min=((moments_Gmin(E_vec[i],M)))*JacksonKernel(M)     
        moments_delta_vec=moments_delta(E_vec[i],M)*JacksonKernel(M)
        for n in range(M):
            for m in range(M):
                moments_table_big[m,n,i]=moments_G_vec_mas[m]*moments_delta_vec[n]+moments_G_vec_min[n]*moments_delta_vec[m]

    
    moments_table=sci.integrate.simpson(moments_table_big,x=E_vec,axis=2)


    T_delta_0=np.zeros(len(random_vector),dtype=np.complex128,order='C')
    T_delta_1=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    r_vec=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    T_delta_v=np.zeros(len(random_vector),dtype=np.complex128,order='C')


    
    T_delta_0=np.copy(random_vector,order='C')
    V.dot(1+0j,0+0j,T_delta_0,T_delta_v)

    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,0],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    H_kpm.dot(1+0j,0+0j,random_vector,T_delta_1)
    V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
   
    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,1],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    for n in range(2,M):
        
        H_kpm.dot(2.0+0j,-1+0j,T_delta_1,T_delta_0)
        temp = T_delta_0
        T_delta_0 = T_delta_1
        T_delta_1 = temp 

        V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
        axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,n],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)


    return 0.5*r_vec/(H_kpm.bounds[1])


def position_operator_min_2(H_kpm=None,V=None,M=None,random_vector=None,alpha_integral=0.99):
    # H=H_kpm_0.modifier(modifier_bounds_inverse)
    # H.bounds=H.bounds/alpha_integral
    # H_kpm=H.modifier(modifier_bounds)


    E_vec=np.linspace(-alpha_integral,alpha_integral,2*M)


    moments_table_big=np.zeros((M,M,len(E_vec)),dtype=np.complex128)
    moments_G_vec=np.zeros(M,dtype=np.complex128)
    moments_delta_vec=np.zeros(M,dtype=np.complex128)
    
    for i in range(len(E_vec)):
        moments_G_vec=(moments_Gmin(E_vec[i],M))*JacksonKernel(M)     
        moments_delta_vec=moments_delta(E_vec[i],M)*JacksonKernel(M)
        for n in range(M):
            for m in range(M):
                moments_table_big[m,n,i]=-1j*moments_delta_vec[m]*moments_G_vec[n]

    
    moments_table=sci.integrate.simpson(moments_table_big,x=E_vec,axis=2)

    print(np.conj(moments_table.T))

    T_delta_0=np.zeros(len(random_vector),dtype=np.complex128,order='C')
    T_delta_1=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    r_vec=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    T_delta_v=np.zeros(len(random_vector),dtype=np.complex128,order='C')


    
    T_delta_0=np.copy(random_vector,order='C')
    V.dot(1+0j,0+0j,T_delta_0,T_delta_v)

    axpy2(1+0j,1+0j,moments_table[0,0]*T_delta_v,r_vec,len(random_vector),H_kpm.n_threads)
    # axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,0],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    # H_kpm.dot(1+0j,0+0j,random_vector,T_delta_1)
    # V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
   
    # axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,1],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    # for n in range(2,M):
        
    #     H_kpm.dot(2.0+0j,-1+0j,T_delta_1,T_delta_0)
    #     temp = T_delta_0
    #     T_delta_0 = T_delta_1
    #     T_delta_1 = temp 

    #     V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
    #     axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table[:,n],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)


    end=perf_counter()
    return r_vec/(H_kpm.bounds[1])



def position_operator_tot(H_kpm=None,V=None,M=None,random_vector=None,alpha_integral=0.99):
    start=perf_counter()
    # H=H_kpm_0.modifier(modifier_bounds_inverse)
    # H.bounds=H.bounds/alpha_integral
    # H_kpm=H.modifier(modifier_bounds)


    E_vec=np.linspace(-alpha_integral,alpha_integral,2*M)


    moments_table_big=np.zeros((M,M,len(E_vec)),dtype=np.complex128)
    moments_G_vec=np.zeros(M,dtype=np.complex128)
    moments_delta_vec=np.zeros(M,dtype=np.complex128)
    
    for i in range(len(E_vec)):
        moments_G_vec=(moments_Gmas(E_vec[i],M))*JacksonKernel(M)     
        moments_delta_vec=moments_delta(E_vec[i],M)*JacksonKernel(M)
        for n in range(M):
            for m in range(M):
                moments_table_big[m,n,i]=moments_G_vec[m]*moments_delta_vec[n]-moments_G_vec[m]*moments_delta_vec[n]

    
    moments_table_mas=1j*sci.integrate.simpson(moments_table_big,x=E_vec,axis=2)
    moments_table_min=-1j*sci.integrate.simpson(np.conj(moments_table_big),x=E_vec,axis=2)

    # print(moments_table_mas-np.conj(moments_table_min))
    # print(moments_table_mas)
    # print(moments_table_min)

    T_delta_0=np.zeros(len(random_vector),dtype=np.complex128,order='C')
    T_delta_1=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    r_vec=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    T_delta_v=np.zeros(len(random_vector),dtype=np.complex128,order='C')


    print('start this recursion')
    
    T_delta_0=np.copy(random_vector,order='C')
    V.dot(1+0j,0+0j,T_delta_0,T_delta_v)
    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table_mas[:,0],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    H_kpm.dot(1+0j,0+0j,random_vector,T_delta_1)
    V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
   
    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table_mas[:,1],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)
    
    for n in range(2,M):
        
        H_kpm.dot(2.0+0j,-1+0j,T_delta_1,T_delta_0)
        temp = T_delta_0
        T_delta_0 = T_delta_1
        T_delta_1 = temp 

        V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
        axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table_mas[:,n],T_delta_v),r_vec,len(random_vector),H_kpm.n_threads)

    r_vec=r_vec

    end=perf_counter()
    print('Time it took the position operator : '+str(end-start))

    T_delta_0=np.zeros(len(random_vector),dtype=np.complex128,order='C')
    T_delta_1=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    r_vec_min=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    T_delta_v=np.zeros(len(random_vector),dtype=np.complex128,order='C')


    print('start this recursion')
    
    T_delta_0=np.copy(random_vector,order='C')
    V.dot(1+0j,0+0j,T_delta_0,T_delta_v)

    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table_min[0,:],T_delta_v),r_vec_min,len(random_vector),H_kpm.n_threads)
    
    H_kpm.dot(1+0j,0+0j,random_vector,T_delta_1)
    V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
   
    axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table_min[1,:],T_delta_v),r_vec_min,len(random_vector),H_kpm.n_threads)
    
    for n in range(2,M):
        
        H_kpm.dot(2.0+0j,-1+0j,T_delta_1,T_delta_0)
        temp = T_delta_0
        T_delta_0 = T_delta_1
        T_delta_1 = temp 

        V.dot(1+0j,0+0j,T_delta_1,T_delta_v)
        axpy2(1+0j,1+0j,rec_A_vec(M,H_kpm,moments_table_min[n,:],T_delta_v),r_vec_min,len(random_vector),H_kpm.n_threads)

    r_vec_min=r_vec_min
    end=perf_counter()
    print('Time it took the position operator : '+str(end-start))


    print(np.vdot(random_vector,r_vec))
    print(np.vdot(random_vector,r_vec_min))
    # print(np.vdot(random_vector,r_vec_min+r_vec))
    # print(np.vdot(random_vector,r_vec-r_vec_min))


    # plt.plot(np.real(r_vec))
    # plt.plot(np.real(r_vec_min),'--')
    # plt.show()
    # plt.plot(np.imag(r_vec))
    # plt.plot(np.imag(r_vec_min),'--')

    # plt.show()

    # plt.plot(np.real(r_vec)-np.real(r_vec_min))
    # plt.show()
    # plt.plot(np.imag(r_vec)-np.imag(r_vec_min))
    # plt.show()
    # print(np.vdot(r_vec,r_vec_min))
    # print(H_kpm.shape[0])
    return r_vec/(H_kpm.bounds[1]),r_vec_min/(H_kpm.bounds[1])




### Time dependent conductivity
def kpm_sigma_time_nequil_pos(H,direction_1=None,direction_2=None,t_vec=None,U_F_t0=None,U_t0=None,M=None,bounds=True,S_x=None):
    """
    Def:
        Time-dependent conducitivity
    Inputs:
        H: Hamiltonian in CSR or ELL.
        direction_1 : (hidden) Direction of the index 'x'/0, 'y'/1 or 'z'/2.
        direction_2 : (hidden) Direction of the index 'x'/0, 'y'/1 or 'z'/2.
        t_vec : (hidden) time vector in fs.
        M : (hidden) number of moments of the KPM expansion by default set to int(sqrt(N)).
        random_vector: (hidden) random vector , by default is computed as expected.
        
    Outputs:
        delta_vec: Complex array of shape len(t_vec) .
    Some notes:
        tau_broadening=M*hbar/(pi*bounds_arra[1]) if bounds are symmetrical.
        To obtain the conductivity in units of e**2/h you should multiply the result by 2*pi/(Omega*hbar)
    """
    ### The checks and default definitions
    if M is None:
        M=int(np.sqrt(H.shape[0]))
    

    if t_vec is None:
        print('No time vector has been inserted by default is set from 0 to 5*tau_phi with tau_phi=hbar*M/(pi*bounds[1]) with tau_phi/100 steps')
        t_vec=np.linspace(0,5*hbar_fs*M/(pi*H.bounds[1]),500)
    if U_F_t0 is None or U_t0 is None:
        print('You should introduce U_F and U for computing this observable .')

    ### Moments for time evolution
    delta_t=t_vec[1]
    delta_t_kpm=delta_t*H.bounds[1]/hbar_fs
    moments_U_vec=moments_U(delta_t_kpm,M)
    m=0
    while np.abs(moments_U_vec[m])>1e-15 and m<(M-1):
        m+=1
    M2=m
    moments_U_vec=moments_U(delta_t_kpm,M2)

    ### Reescalating the Hamiltonian and moments for the FD
    if bounds==True:

        H_kpm=H.modifier(modifier_bounds)
        V1 = H.modifier(modifier_velocity,direction_1)
        V2 = H.modifier(modifier_velocity,direction_2)

    elif bounds==False:
        
        H_kpm=H.deep_copy()
        H2=H.modifier(modifier_bounds_inverse)
        V1 = H2.modifier(modifier_velocity,direction_1)
        V2 = H2.modifier(modifier_velocity,direction_2)
        
    ## Necessary arrays for the loop . 
    sigma_vec=np.zeros((len(t_vec)),dtype=np.complex128)

    aux=np.zeros(H.shape[0],dtype=np.complex128,order='C')

    for j in range(len(t_vec)):
        if j==0:
            U_F=U_F_t0            
            U_r=position_operator_mas_2(H_kpm,V2,M,U_t0)

        else:
            U_r=rec_A_vec(M2,H_kpm,moments_U_vec,U_r)
            U_F=rec_A_vec(M2,H_kpm,moments_U_vec,U_F)

        V1.dot(1.0+0j,0+0j,U_r,aux)
        sigma_vec[j]=vdot(1+0j,U_F,aux,H.shape[0],H.n_threads)

    
    return -1j*2*sigma_vec


def kpm_rho_neq_pos(H,t_vec=None,tau=None,modifier_id=None,modifier_params=None,Temp=None,mu=None,observale_list=None,M=None,random_vector=None,dos_equil=None):
    """
    Def:
        Time-evolution of the density matrixeven if tau!=0 it does not include the relaxation.
    Inputs:
        H: Hamiltonian in CSR or ELL.
        t_vec : (hidden) time vector in fs.
        tau : (hidden) Relaxation time in fs by default in 0 even if !=0 this function does not include the minimization .
        modifier_id : (hidden) Choosing the light type linear , circle , linear packed , circle packed . 
        modifier_params : (hidden) Parameters that enter into the modifier .
        Temp : (hidden) Temperature in Kelvin .
        Temp_equil : (hidden) Temperature in Kelvin towards the system will relax only matters if tau!=0 .
        Ef : (hidden) Fermi velocity by default set to 0 eV .
        observable_list : a list of the following form [[observable_id,n_meass,observable_params],...]
        M : (hidden) number of moments of the KPM expansion by default set to int(sqrt(N)).
        random_vector: (hidden) random vector , by default is computed as expected.
        
    Outputs:
        F_t0 : Density matrix at t=0 .
        U_t0 : Time evolution operator at t=0 .
        F : Density matrix at t=end .
        F : Time evolution operator at t=end .
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
        moments_kernel_FD=moments_FD_T0(mu/H.bounds[1],M)*JacksonKernel(M)
    else:
        moments_kernel_FD=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)

    ### Unwrap of n computation 
    k_n=-1
    k_sigma=-1
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

        elif observale_list[i][0] == 'sigma_nequil':
            sigma_meass=observale_list[i][1]
            direction_1=observale_list[i][2]
            direction_2=observale_list[i][3]
            t_vec_sigma=observale_list[i][4]
            M_sigma=observale_list[i][5]

            if sigma_meass>1:
                sigma_meass_vec=np.linspace(0,len(t_vec)-1,sigma_meass,dtype=int)
            else:
                sigma_meass_vec=np.array([len(t_vec)-1],dtype=int)
            
            k_sigma=0
            sigma_vec=np.zeros((sigma_meass,len(t_vec_sigma)),dtype=np.complex128)
    
        else:
            k_n=-1
            k_sigma=-1
            print('There is no observable selected the function will return F(t=0) , U(t=0) , F(t=end) and U(t=end) .')



    ## Time evolution
    H_time=H_kpm.deep_copy()
    aux=np.zeros(len(random_vector),dtype=np.complex128)
    for i in (range(len(t_vec))):
        H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec[i])
        if i==0:
            U=np.copy(random_vector)
            F=rec_A_vec(M,H_kpm,moments_kernel_FD,random_vector)
            F_t0=F
            U_t0=U
            if tau!=0:
                H_kpm.dot(1+0j,0+0j,F,aux)
                E_0=2*H_kpm.bounds[1]*np.real(vdot(1+0j,random_vector,aux,len(random_vector),H_time.n_threads))
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
                    n_mat[k_n,:],dos_n_mat[k_n,:]=kpm_n_dos_n(H,M_n,U,F,bounds=True)
                else:
                    n_mat[k_n,:],dos_n_mat[k_n,:]=kpm_n_dos_n(H_time,M_n,U,F,bounds=False)
                
                k_n+=1
        if k_sigma >= 0:
            if i == sigma_meass_vec[k_sigma]:
                print('Computing sigma :'+str(k_sigma))
                if i==0:
                    sigma_vec[k_sigma,:]=kpm_sigma_time_nequil_pos(H,direction_1,direction_2,t_vec_sigma,F,U,M_sigma,bounds=True)

                else:
                    sigma_vec[k_sigma,:]=kpm_sigma_time_nequil_pos(H_time,direction_1,direction_2,t_vec_sigma,F,U,M_sigma,bounds=False)
                
                k_sigma+=1


    if k_n>=0 and k_sigma<0:
        return n_mat,dos_n_mat
    elif k_n<0 and k_sigma>=0:
        return sigma_vec
    elif k_n>=0 and k_sigma>=0:
        print('Final Temperature : '+str(Temp)+' [K]')
        print('Final Chemical potential : '+str(mu)+' [eV]')
        return n_mat,dos_n_mat,sigma_vec
    else:
        return F_t0,U_t0,F,U
    


def kpm_angular_momentum(H,V1,V2,M=None,random_vector=None,bounds=True,alpha_integral=0.99):

    if bounds == True:
        H_kpm=H.modifier(modifier_bounds)
    elif bounds == False:
        H_kpm=H.deep_copy()
        H=H.modifier(modifier_bounds_inverse)

    ## We have to make : r1*V2-V1*r2 == term_1-term_2


    # Term_1
    vector_1=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    V2.dot(1+0j,0+0j,random_vector,vector_1)

    pos_1=position_operator_mas(H_kpm,V1,M,vector_1,alpha_integral)

    # Term_2
    pos_2=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    vector_2=position_operator_min(H_kpm,V2,M,random_vector,alpha_integral)

    V1.dot(1+0j,0+0j,vector_2,pos_2)

    # Doing the substraction

    axpy2(1+0j,-1+0j,pos_1,pos_2,len(random_vector),H_kpm.n_threads)



    return pos_2


from tqdm import tqdm

def kpm_sigma_time_orbital(H,direction_1=None,direction_2=None,t_vec=None,mu_vec=None,Temp=None,M=None,random_vector=None,velocities=None):
    """
    Def:
        Time-dependent conducitivity
    Inputs:
        H: Hamiltonian in CSR or ELL.
        direction_1 : (hidden) Direction of the index 'x'/0, 'y'/1 or 'z'/2.
        direction_2 : (hidden) Direction of the index 'x'/0, 'y'/1 or 'z'/2.
        t_vec : (hidden) time vector in fs.
        Ef : (hidden) Fermi velocity by default set to 0 eV .
        Temp : (hidden) Temperature in Kelvin .
        M : (hidden) number of moments of the KPM expansion by default set to int(sqrt(N)).
        random_vector: (hidden) random vector , by default is computed as expected.
        
    Outputs:
        delta_vec: Complex array of shape len(t_vec) .
    Some notes:
        tau_broadening=M*hbar/(pi*bounds_arra[1]) if bounds are symmetrical.
        To obtain the conductivity in units of e**2/h you should multiply the result by 2*pi/(Omega*hbar)
    """
    ### The checks and default definitions
    if M is None:
        M=int(np.sqrt(H.shape[0]))
    

    if random_vector is None:
        random_vector=random_vector_generator(H.shape[0])

    if t_vec is None:
        print('No time vector has been inserted by default is set from 0 to 5*tau_phi with tau_phi=hbar*M/(pi*bounds[1]) with tau_phi/100 steps')
        t_vec=np.linspace(0,5*hbar_fs*M/(pi*H.bounds[1]),500)
 

    ### Moments for time evolution
    delta_t=t_vec[1]
    delta_t_kpm=delta_t*H.bounds[1]/hbar_fs
    moments_U_vec=moments_U(delta_t_kpm,M)
    m=0
    while np.abs(moments_U_vec[m])>1e-15 and m<(M-1):
        m+=1
    M2=m
    moments_U_vec=moments_U(delta_t_kpm,M2)

    ### Reescalating the Hamiltonian and moments for the FD
    
    H_kpm=H.modifier(modifier_bounds)

    # V1=H.modifier(modifier_velocity,direction_1)
    # V2=H.modifier(modifier_velocity,direction_2)
    # V2_kpm=H.modifier(modifier_velocity_bounds,direction_2)

    if velocities is None:
        V2=H.modifier(modifier_velocity,direction_2)
        V2_kpm = H.modifier(modifier_velocity_bounds,direction_2)
        V1 = H.modifier(modifier_velocity,direction_1)
    else:
        V1 = copy.deepcopy(H)
        V2 = copy.deepcopy(H)
        V2_kpm = copy.deepcopy(H)

        if direction_1 == 'x' and direction_2=='x':
            V1.data=np.copy(V1.dx_vec)   
            V2.data=np.copy(V2.dx_vec)   
            axpy2(1j/((H.bounds[1]-H.bounds[0])/2),0+0j, V2_kpm.dx_vec,V2_kpm.data, len(V2_kpm.data),H.n_threads)
    

        if direction_1 == 'x' and direction_2=='y':
            V1.data=np.copy(V1.dx_vec)   
            V2.data=np.copy(V1.dy_vec)   
            axpy2(1j/((H.bounds[1]-H.bounds[0])/2),0+0j, V2_kpm.dy_vec,V2_kpm.data, len(V2_kpm.data),H.n_threads)
    

        if direction_1 == 'y' and direction_2=='x':
            V1.data=np.copy(V1.dy_vec)   
            V2.data=np.copy(V2.dx_vec)   
            axpy2(1j/((H.bounds[1]-H.bounds[0])/2),0+0j, V2_kpm.dx_vec,V2_kpm.data, len(V2_kpm.data),H.n_threads)
    

        if direction_1 == 'y' and direction_2=='y':
            V1.data=np.copy(V1.dy_vec)   
            V2.data=np.copy(V2.dy_vec)   
            axpy2(1j/((H.bounds[1]-H.bounds[0])/2),0+0j, V2_kpm.dy_vec,V2_kpm.data, len(V2_kpm.data),H.n_threads)
    

    ## Lets compute first the anticommutator, L*V1+V1*L == term_1+term_2

    # Term 1
    vector_1=np.zeros(len(random_vector),dtype=np.complex128,order='C')
    V1.dot(1+0j,0+0j,random_vector,vector_1)
    
    L1=kpm_angular_momentum(H_kpm,V1,V2,M,vector_1,bounds=False)

    # Term 2

    vector_2=kpm_angular_momentum(H_kpm,V1,V2,M,random_vector,bounds=False)

    L2=np.zeros(len(random_vector),dtype=np.complex128,order='C')

    V1.dot(1+0j,0+0j,vector_2,L2)

    # Summation

    axpy2(1+0j,1+0j,L1,L2,len(random_vector),H_kpm.n_threads)

    ## Necessary arrays for the loop . 
    sigma_vec=np.zeros((len(mu_vec),len(t_vec)),dtype=np.complex128)
    for j in (range(len(t_vec))):
        if j==0:
            U_dag=random_vector            
            U_dag_L=L2

        else:
            U_dag=rec_A_vec(M2,H_kpm,np.conj(moments_U_vec),U_dag)

            U_dag_L=rec_A_vec(M2,H_kpm,np.conj(moments_U_vec),U_dag_L)

        for i in range(len(mu_vec)):
            if Temp==0:
                moments_kernel_FD=moments_FD_T0(mu_vec[i]/H.bounds[1],M)*JacksonKernel(M)
            else:
                moments_kernel_FD=moments_FD_T(mu_vec[i]/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)

            sigma_vec[i,j]=vdot(1+0j,U_dag,rec_com_A_vec(M,H_kpm,V2_kpm,moments_kernel_FD,U_dag_L),H.shape[0],H.n_threads)

    return sigma_vec