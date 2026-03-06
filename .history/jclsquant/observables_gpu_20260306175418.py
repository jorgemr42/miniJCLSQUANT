from .cuda_cython.obv_gpu import kpm_rho_neq_gpu_cuda,kpm_rho_neq_gpu_cuda_k,for_time_sigma_nequil_gpu,for_time_sigma_gpu,rec_A_tab_gpu,rec_A_tab_2_gpu,kpm_rho_neq_tau_gpu,kpm_position_operator_gpu,kpm_angular_momentum_gpu_c,kpm_sigma_time_orbital_c,kpm_msd_gpu_c
from jclsquant.cython_modules.Extra_funs import table_G_delta

from jclsquant.modifiers import *
from jclsquant.kernel_and_moments import *
from jclsquant.constants import *
from jclsquant.hams import *
from jclsquant.observables import *


## Density of states (dos)
def kpm_dos_gpu(H,M=None,random_vector=None):
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

    delta_vec=np.zeros(M,dtype=np.complex128)

    rec_A_tab_gpu(H_kpm.data,H_kpm.indices,H.len_row,random_vector,delta_vec,M)
    
    kernel=JacksonKernel(M)
    for i in range(len(Ef_vec)):
        delta_mat[i,1]=np.real(dot(1+0j,moments_delta(Ef_vec[i]/(0.5*(H.bounds[1]-H.bounds[0]))-(0.5*(H.bounds[1]+H.bounds[0]))/(0.5*(H.bounds[1]-H.bounds[0])),M).astype(np.complex128)*kernel,delta_vec,M,H.n_threads))

    return delta_mat



def kpm_dos_2_gpu(H,M=None,random_vector_l=None,random_vector_r=None,bounds=True,proyector=None):
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
    delta_vec=np.zeros(M,dtype=np.complex128)


    if proyector is not None:
        random_vector_l_2=np.zeros(len(random_vector_l),dtype=np.complex128,order='C')
        # vec_mul(1,random_vector_l,proyector,random_vector_l_2,len(random_vector_l_2),H.n_threads)
        proyector.dot(1.0+0j,0+0j,random_vector_l,random_vector_l_2)
    
        if bounds==True:

            H_kpm=H.modifier(modifier_bounds)
            rec_A_tab_2_gpu(H_kpm.data,H_kpm.indices,H.len_row,random_vector_l_2,random_vector_r,delta_vec,M)

        elif bounds==False:

            rec_A_tab_2_gpu(H.data,H.indices,H.len_row,random_vector_l_2,random_vector_r,delta_vec,M)
    else:
        if bounds==True:

            H_kpm=H.modifier(modifier_bounds)
            rec_A_tab_2_gpu(H_kpm.data,H_kpm.indices,H.len_row,random_vector_l,random_vector_r,delta_vec,M)

        elif bounds==False:

            rec_A_tab_2_gpu(H.data,H.indices,H.len_row,random_vector_l,random_vector_r,delta_vec,M)

    kernel_vec=JacksonKernel(M)

    for i in range(len(Ef_vec)):
        delta_mat[i,1]=np.real(dot(1+0j,moments_delta(Ef_vec[i]/(0.5*(H.bounds[1]-H.bounds[0]))-(0.5*(H.bounds[1]+H.bounds[0]))/(0.5*(H.bounds[1]-H.bounds[0])),M).astype(np.complex128)*kernel_vec,delta_vec,M,H.n_threads))

    return delta_mat


def kpm_rho_neq_gpu(H,t_vec=None,tau=0,modifier_id=None,modifier_params=None,Temp=None,Ef=None,observale_list=None,M=None,random_vector=None,proyector=None):
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
    
    if Ef is None:
        print('No Fermi energy has been included by default is set to 0 .')
        Ef=0
    if random_vector is None:
        random_vector=random_vector_generator(H.shape[0])

    if t_vec is None:
        print('No time vector has been inserted by default is set from 0 to 5*tau_phi with tau_phi=hbar*M/(pi*bounds[1]) with tau_phi/100 steps')
        t_vec=np.linspace(0,5*hbar_fs*M/(np.pi*(0.5*(H.bounds[1]-H.bounds[0]))),500)
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
    delta_t_kpm=delta_t*(0.5*(H.bounds[1]-H.bounds[0]))/hbar_fs

    moments_U_vec=moments_U(delta_t_kpm,M)
    m=0
    while np.abs(moments_U_vec[m])>1e-15:
        m+=1
    M2=m
    moments_U_vec=moments_U(delta_t_kpm,M2)

    ### Reescalating the Hamiltonian and moments for the FD

    H_kpm=H.modifier(modifier_bounds)
    if Temp==0:
        moments_kernel_FD=moments_FD_T0((Ef-(0.5*(H.bounds[1]+H.bounds[0])))/(0.5*(H.bounds[1]-H.bounds[0])),M)*JacksonKernel(M)
    else:
        moments_kernel_FD=moments_FD_T((Ef-(0.5*(H.bounds[1]+H.bounds[0])))/(0.5*(H.bounds[1]-H.bounds[0])),((Temp*kb))/(0.5*(H.bounds[1]-H.bounds[0])),M)*JacksonKernel(M)

    ### Unwrap of n computation 
    k_n=-1
    k_sigma=-1
    for i in range(len(observale_list)):

        if observale_list[i][0] == 'n':            
            M_n=observale_list[i][2]

            k_n=0

        elif observale_list[i][0] == 'sigma_nequil':
            direction_1=observale_list[i][2]
            direction_2=observale_list[i][3]
            t_vec_sigma=observale_list[i][4]
            M_sigma=observale_list[i][5]

            k_sigma=0

        else:
            print('There is no observable selected the function will return F(t=0) , U(t=0) , F(t=end) and U(t=end) .')


    U=np.copy(random_vector,order='C')
    U_F=np.copy(random_vector,order='C')
    
    moments_kernel_FD=moments_kernel_FD.astype(np.complex128)
    t_vec=t_vec.astype(np.complex128)
    if modifier_id== 'circle_packed':
        modifier_id_gpu=0

        A0,w,pol,Tp=modifier_params


        if pol=='r':
            modifier_params_gpu=np.array([A0,w,Tp,1],dtype=np.complex128)
        else:
            modifier_params_gpu=np.array([A0,w,Tp,-1],dtype=np.complex128)
        
        if H.space=='r':
            kpm_rho_neq_gpu_cuda(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec,U,U_F,modifier_id_gpu,modifier_params_gpu,M,M2,np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(2,dtype=np.complex128,order='C'))
        elif H.space=='k':
            kpm_rho_neq_gpu_cuda_k(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec,U,U_F,modifier_id_gpu,modifier_params_gpu,M,M2,np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(2,dtype=np.complex128,order='C'),H.bounds+0j)
    elif modifier_id=='linear_packed':
        modifier_id_gpu=0

        A0,w,Tp=modifier_params


        modifier_params_gpu=np.array([A0,w,Tp,0],dtype=np.complex128)
        
        if H.space=='r':
            kpm_rho_neq_gpu_cuda(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec,U,U_F,modifier_id_gpu,modifier_params_gpu,M,M2,np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(2,dtype=np.complex128,order='C'))
        if H.space=='k':
            kpm_rho_neq_gpu_cuda_k(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec,U,U_F,modifier_id_gpu,modifier_params_gpu,M,M2,np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(2,dtype=np.complex128,order='C'),H.bounds+0j)
    elif modifier_id == 'phonons_packed':

        modifier_id_gpu=1


        a0,b,Aq,wq,pol,S=modifier_params
        
        
        e_pol=np.zeros((H_kpm.shape[0],2),dtype=complex,order='C')

        if pol=='chiral_+':
            q=np.array([4*np.pi/(3*np.sqrt(3)*a0),0],dtype=complex)
            e_pol[:,0]=np.array([1/np.sqrt(2), 1/np.sqrt(2)] * (H_kpm.shape[0]//2))
            e_pol[:,1]=np.array([1j/np.sqrt(2), -1j/np.sqrt(2)] * (H_kpm.shape[0]//2))
        elif pol=='chiral_-':
            q=np.array([-4*np.pi/(3*np.sqrt(3)*a0),0],dtype=complex)
            e_pol[:,0]=np.array([1/np.sqrt(2), 1/np.sqrt(2)] * (H_kpm.shape[0]//2))
            e_pol[:,1]=np.array([-1j/np.sqrt(2), 1j/np.sqrt(2)] * (H_kpm.shape[0]//2))
        elif pol=='optical':
            q=np.array([0,0],dtype=complex)
            e_pol[:,0]=np.array([0, 0] * (H_kpm.shape[0]//2))
            e_pol[:,1]=np.array([1, -1] * (H_kpm.shape[0]//2))
        modifier_params_gpu=np.array([a0,b,Aq,wq],dtype=np.complex128)
        
        if H.space=='r':
            kpm_rho_neq_gpu_cuda(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec,U,U_F,modifier_id_gpu,modifier_params_gpu,M,M2,np.ascontiguousarray(S[:,0]),np.ascontiguousarray(S[:,1]),np.ascontiguousarray(e_pol[:,0]),np.ascontiguousarray(e_pol[:,1]),q)
        elif H.space=='k':
            print('The modifers of the phonons are not implemented for the moment')
            kpm_rho_neq_gpu_cuda_k(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec,U,U_F,modifier_id_gpu,modifier_params_gpu,M,M2,np.ascontiguousarray(S[:,0]),np.ascontiguousarray(S[:,1]),np.ascontiguousarray(e_pol[:,0]),np.ascontiguousarray(e_pol[:,1]),q,H.bounds+0j)

    #### Computation of the conductivity and the number of electrons
    H_time=H_kpm.deep_copy()
    
    H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec[-1])

    if k_n>=0 and k_sigma<0:
        
        dos_n_vec=kpm_dos_2_gpu(H_time,M_n,U,U_F,False,proyector)
        dos=kpm_dos_2_gpu(H_time,M_n,U,U,False,proyector)


        return dos_n_vec,dos

    if k_n>=0 and k_sigma>=0:
        dos_n_vec=kpm_dos_2_gpu(H_time,M_n,U,U_F,False,proyector)
        return dos_n_vec









def kpm_rho_neq_gpu_sigma_tau(H,t_vec=None,tau=None,modifier_id=None,modifier_params=None,Temp=None,mu=None,observale_list=None,M=None,random_vector=None,dos_equil=None):
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
        t_vec=np.linspace(0,5*hbar_fs*M/(np.pi*H.bounds[1]),500)
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
    k_sigma=-1
    k_n=-1
    for i in range(len(observale_list)):

        if observale_list[i][0] == 'n':            
            M_n=observale_list[i][2]

            k_n=0

        elif observale_list[i][0] == 'sigma_nequil':
            direction_1=observale_list[i][2]
            direction_2=observale_list[i][3]
            t_vec_sigma=observale_list[i][4]
            M_sigma=observale_list[i][5]

            k_sigma=0

        else:
            print('There is no observable selected the function will return F(t=0) , U(t=0) , F(t=end) and U(t=end) .')





    

    A0,w,pol,Tp=modifier_params


    if pol=='r':
        modifier_params_gpu=np.array([A0,w,Tp,1],dtype=np.complex128)
    else:
        modifier_params_gpu=np.array([A0,w,Tp,-1],dtype=np.complex128)
    
    
    aux=np.zeros(len(random_vector),dtype=np.complex128)
    H_time=H_kpm.deep_copy()
    
    delta_t_tau=t_vec[1]/tau

    #### t==0:
    moments_kernel_FD=moments_kernel_FD.astype(np.complex128)
    t_vec=t_vec.astype(np.complex128)
    U=np.copy(random_vector,order='C')
    F=rec_A_vec(M,H_kpm,moments_kernel_FD,random_vector)
    if tau!=0:
        H_kpm.dot(1+0j,0+0j,F,aux)
        E_0=2*H_kpm.bounds[1]*np.real(vdot(1+0j,random_vector,aux,len(random_vector),H_kpm.n_threads))
        moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)    
    ####Rest of times
    t_index=np.arange(0, len(t_vec), len(t_vec)//20)
    for i in range(len(t_index)) :
        if i==0:
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[i]+1:t_index[i+1]+1],dtype=np.complex128)
            kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)
            
            H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])

            #Minimization part
            print('Chemical potential : '+str(mu)+' [eV]')
            print('Temperature : '+str(Temp)+' [K]')
            H_time.dot(1+0j,0+0j,F,aux)
                
            E_t=2*H_time.bounds[1]*np.real(vdot(1+0j,U,aux,len(U),H_time.n_threads))
            
            deltaE=(E_t-E_0)/H.shape[0]   
            #print(deltaE)         
            mu,Temp=minimization(deltaE,dos_equil,[mu,Temp*kb],niter=10000)

            E_0=E_t
            
            moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)
        elif i ==(len(t_index)-1):
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[-1]+1:],dtype=np.complex128)
            
            kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)

        else:
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[i]+1:t_index[i+1]+1],dtype=np.complex128)


            kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)
            H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])            
            print('Chemical potential : '+str(mu)+' [eV]')
            print('Temperature : '+str(Temp)+' [K]')
            #Minimization part
            H_time.dot(1+0j,0+0j,F,aux)
                
            E_t=2*H_time.bounds[1]*np.real(vdot(1+0j,U,aux,len(U),H_time.n_threads))
            
            deltaE=(E_t-E_0)/H.shape[0]
            
            mu,Temp=minimization(deltaE,dos_equil,[mu,Temp*kb],niter=10000)
            E_0=E_t
            moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)

    print('Final Temperature : '+str(Temp)+' [K]')
    print('Final Chemical potential : '+str(mu)+' [eV]')
    
    #### Computation of the conductivity and the number of electrons
    H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec[-1])

    if k_n>=0 and k_sigma<0:
        dos_n_vec=kpm_dos_2_gpu(H_time,M_n,U,F,bounds=False)
        return dos_n_vec

    if k_n>=0 and k_sigma>=0:
        dos_n_vec=kpm_dos_2_gpu(H_time,M_n,U,F,bounds=False)
        return dos_n_vec
    

def kpm_rho_neq_gpu_sigma_tau_2(H,t_vec=None,tau=None,modifier_id=None,modifier_params=None,Temp=None,mu=None,observale_list=None,M=None,random_vector=None,dos_equil=None):
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
        t_vec=np.linspace(0,5*hbar_fs*M/(np.pi*H.bounds[1]),500)
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
    for i in range(len(observale_list)):

        if observale_list[i][0] == 'n':            
            M_n=observale_list[i][2]
            t_vec_meass_n=observale_list[i][3]

            k_n=0
            dos_n_vec=np.zeros((len(t_vec_meass_n),2*M_n+1,2),dtype=np.complex128)
            t_vec_meass_new_n=np.zeros(len(t_vec_meass_n))
            t_vec_meass_new_sigma=t_vec_meass_new_n
            t_vec_meass_sigma=t_vec_meass_n
            k_sigma=-1
        elif observale_list[i][0] == 'sigma_nequil':
            direction_1=observale_list[i][2]
            direction_2=observale_list[i][3]
            t_vec_sigma=observale_list[i][4]
            M_sigma=observale_list[i][5]
            t_vec_meass_sigma=observale_list[i][6]

            k_sigma=0

            sigma_vec=np.zeros((len(t_vec_meass_sigma),len(t_vec_sigma)),dtype=np.complex128)
            t_vec_meass_new_sigma=np.zeros(len(t_vec_meass_sigma))


        else:
            k_n=-1
            k_sigma=-1
            print('There is no observable selected the function will return F(t=0) , U(t=0) , F(t=end) and U(t=end) .')

    if modifier_id=='circle_packed':
        A0,w,pol,Tp=modifier_params

        if pol=='r':
            modifier_params_gpu=np.array([A0,w,Tp,1],dtype=np.complex128)
        else:
            modifier_params_gpu=np.array([A0,w,Tp,-1],dtype=np.complex128)
    elif modifier_id=='linear_packed':
        A0,w,Tp=modifier_params

        modifier_params_gpu=np.array([A0,w,Tp,0],dtype=np.complex128)
    else:
        print('You must introduce a valid mdofier')
        return 
    
    aux=np.zeros(len(random_vector),dtype=np.complex128)
    H_time=H_kpm.deep_copy()
    delta_t_tau=t_vec[1]/tau

    #### t==0:
    moments_kernel_FD=moments_kernel_FD.astype(np.complex128)
    t_vec=t_vec.astype(np.complex128)

    U=np.copy(random_vector,order='C')
    F=rec_A_vec(M,H_kpm,moments_kernel_FD,random_vector)
    if tau!=0:
        H_kpm.dot(1+0j,0+0j,F,aux)
        E_0=2*H_kpm.bounds[1]*np.real(vdot(1+0j,random_vector,aux,len(random_vector),H_kpm.n_threads))
        moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)    
    ####Rest of times
    t_index=np.arange(0, len(t_vec), len(t_vec)//20)
    
    if len(t_index)<len(t_vec_meass_sigma) or len(t_index)<len(t_vec_meass_n):

        t_index=np.arange(0, len(t_vec), len(t_vec)//(np.max(np.array([len(t_vec_meass_sigma),len(t_vec_meass_new_n)]))))
        
    
    
    for i in range(len(t_index)) :
        if i==0:
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[i]+1:t_index[i+1]+1],dtype=np.complex128)
            kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)
            
            H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])

            #Minimization part
            print('Chemical potential : '+str(mu)+' [eV]')
            print('Temperature : '+str(Temp)+' [K]')
            H_time.dot(1+0j,0+0j,F,aux)
                
            E_t=2*H_time.bounds[1]*np.real(vdot(1+0j,U,aux,len(U),H_time.n_threads))
            
            deltaE=(E_t-E_0)/H.shape[0]   
            #print(deltaE)         
            mu,Temp=minimization(deltaE,dos_equil,[mu,Temp*kb],niter=10000)

            E_0=E_t
            
            moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)
        elif i ==(len(t_index)-1):
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[-1]+1:],dtype=np.complex128)
            
            kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)
            
            H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])            

        else:
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[i]+1:t_index[i+1]+1],dtype=np.complex128)


            kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)
            
            
            H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])   
                     
            print('Chemical potential : '+str(mu)+' [eV]')
            print('Temperature : '+str(Temp)+' [K]')
            #Minimization part
            H_time.dot(1+0j,0+0j,F,aux)
                
            E_t=2*H_time.bounds[1]*np.real(vdot(1+0j,U,aux,len(U),H_time.n_threads))
            
            deltaE=(E_t-E_0)/H.shape[0]
            
            mu,Temp=minimization(deltaE,dos_equil,[mu,Temp*kb],niter=10000)
            E_0=E_t
            moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)


        if k_sigma<len(t_vec_meass_sigma) and k_n<len(t_vec_meass_n):
            if k_sigma>=0 and k_n<0 and np.real(t_vec_meass_sigma[k_sigma])>=np.real(t_vec_segment[0]) and np.real(t_vec_meass_sigma[k_sigma])<=np.real(t_vec_segment[-1]):
                k_sigma+=1

            elif k_n>=0 and k_sigma<0 and np.real(t_vec_meass_n[k_n])>=np.real(t_vec_segment[0]) and np.real(t_vec_meass_n[k_n])<=np.real(t_vec_segment[-1]):
                dos_n_vec[k_n,:]=kpm_dos_2_gpu(H_time,M_n,U,F,bounds=False)
                k_n+=1

            elif k_n>=0 and k_sigma>=0 and np.real(t_vec_meass_n[k_n])>=np.real(t_vec_segment[0]) and np.real(t_vec_meass_n[k_n])<=np.real(t_vec_segment[-1]) and np.real(t_vec_meass_sigma[k_sigma])>=np.real(t_vec_segment[0]) and np.real(t_vec_meass_sigma[k_sigma])<=np.real(t_vec_segment[-1]):
                t_vec_meass_new_sigma[k_sigma]=np.real(t_vec_segment[-1])
                t_vec_meass_new_n[k_n]=np.real(t_vec_segment[-1])

                
                dos_n_vec[k_n,:]=kpm_dos_2_gpu(H_time,M_n,U,F,bounds=False)


                k_n+=1
                k_sigma+=1
            else:
                if (t_vec_meass_sigma[k_sigma]-t_vec_segment[-1])<(t_vec_segment[1]-t_vec_segment[0]):

                    t_vec_meass_new_sigma[k_sigma]=np.real(t_vec_segment[-1])
                    t_vec_meass_new_n[k_n]=np.real(t_vec_segment[-1])


                    dos_n_vec[k_n,:]=kpm_dos_2_gpu(H_time,M_n,U,F,bounds=False)
                    k_n+=1

                    if k_sigma>0:
                        k_sigma+=1
        print('k_n : '+str(k_n))
        print('k_sigma : '+str(k_sigma))

    print('Final Temperature : '+str(Temp)+' [K]')
    print('Final Chemical potential : '+str(mu)+' [eV]')
    print('FInal time that n was meassure : '+str(t_vec_meass_new_n)+' [fs]')
    np.save('./t_vec_meass',t_vec_meass_new_sigma)

    if k_n>=0 and k_sigma<0:
        return dos_n_vec

    elif k_n>=0 and k_sigma>=0:
        return dos_n_vec
    


def kpm_rho_neq_gpu_sigma_tau_3(H,t_vec=None,tau=None,modifier_id=None,modifier_params=None,Temp=None,mu=None,observale_list=None,M=None,random_vector=None,dos_equil=None):
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
        t_vec=np.linspace(0,5*hbar_fs*M/(np.pi*H.bounds[1]),500)
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
    for i in range(len(observale_list)):

        if observale_list[i][0] == 'n':            
            M_n=observale_list[i][2]
            t_vec_meass_n=observale_list[i][3]

            k_n=0
            dos_n_vec=np.zeros((len(t_vec_meass_n),2*M_n+1,2),dtype=np.complex128)


        elif observale_list[i][0] == 'sigma_nequil':
            direction_1=observale_list[i][2]
            direction_2=observale_list[i][3]
            t_vec_sigma=observale_list[i][4]
            M_sigma=observale_list[i][5]
            t_vec_meass_sigma=observale_list[i][6]

            k_sigma=0

            sigma_vec=np.zeros((len(t_vec_meass_sigma),len(t_vec_sigma)),dtype=np.complex128)


        else:
            k_n=-1
            k_sigma=-1
            print('There is no observable selected the function will return F(t=0) , U(t=0) , F(t=end) and U(t=end) .')



    t_vec_meass_new_sigma=np.zeros(len(t_vec_meass_sigma))
    t_vec_meass_new_n=np.zeros(len(t_vec_meass_n))

    

    A0,w,pol,Tp=modifier_params


    if pol=='r':
        modifier_params_gpu=np.array([A0,w,Tp,1],dtype=np.complex128)
    else:
        modifier_params_gpu=np.array([A0,w,Tp,-1],dtype=np.complex128)
    
    
    aux=np.zeros(len(random_vector),dtype=np.complex128)
    H_time=H_kpm.deep_copy()
    delta_t_tau=t_vec[1]/tau

    #### t==0:
    moments_kernel_FD=moments_kernel_FD.astype(np.complex128)
    t_vec=t_vec.astype(np.complex128)
    U=np.copy(random_vector,order='C')
    F=rec_A_vec(M,H_kpm,moments_kernel_FD,random_vector)
    if tau!=0:
        H_kpm.dot(1+0j,0+0j,F,aux)
        E_0=2*H_kpm.bounds[1]*np.real(vdot(1+0j,random_vector,aux,len(random_vector),H_kpm.n_threads))
        moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)    
    ####Rest of times
    t_index=np.arange(0, len(t_vec), len(t_vec)//20)
    
    if len(t_index)<len(t_vec_meass_sigma) or len(t_index)<len(t_vec_meass_n):

        t_index=np.arange(0, len(t_vec), len(t_vec)//(np.max(np.array([len(t_vec_meass_sigma),len(t_vec_meass_new_n)]))))
        
    
    
    for i in range(len(t_index)) :
        if i==0:
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[i]+1:t_index[i+1]+1],dtype=np.complex128)
            
            if tau==0:
                kpm_rho_neq_gpu_cuda(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec_segment,U,F,0,modifier_params_gpu,M,M2,np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(2,dtype=np.complex128,order='C'))
                H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])

            else :
                kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)

                H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])

                #Minimization part
                print('Chemical potential : '+str(mu)+' [eV]')
                print('Temperature : '+str(Temp)+' [K]')
                H_time.dot(1+0j,0+0j,F,aux)
                    
                E_t=2*H_time.bounds[1]*np.real(vdot(1+0j,U,aux,len(U),H_time.n_threads))
                
                deltaE=(E_t-E_0)/H.shape[0]   
                #print(deltaE)         
                mu,Temp=minimization(deltaE,dos_equil,[mu,Temp*kb],niter=10000)

                E_0=E_t
                
                moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)
        
        
        elif i ==(len(t_index)-1):
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[-1]+1:],dtype=np.complex128)
            
            if tau==0:
                kpm_rho_neq_gpu_cuda(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec_segment,U,F,0,modifier_params_gpu,M,M2,np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(2,dtype=np.complex128,order='C'))
            else :
                kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)

            
            H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])            

        else:
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[i]+1:t_index[i+1]+1],dtype=np.complex128)


            if tau==0:
                kpm_rho_neq_gpu_cuda(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec_segment,U,F,0,modifier_params_gpu,M,M2,np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(2,dtype=np.complex128,order='C'))
                H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])            

            else :
                kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)

                H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])            
                print('Chemical potential : '+str(mu)+' [eV]')
                print('Temperature : '+str(Temp)+' [K]')
                #Minimization part
                H_time.dot(1+0j,0+0j,F,aux)
                    
                E_t=2*H_time.bounds[1]*np.real(vdot(1+0j,U,aux,len(U),H_time.n_threads))
                
                deltaE=(E_t-E_0)/H.shape[0]
                
                mu,Temp=minimization(deltaE,dos_equil,[mu,Temp*kb],niter=10000)
                E_0=E_t
                moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)
            
        if k_sigma<len(t_vec_meass_sigma) and k_n<len(t_vec_meass_n):
            if k_sigma>=0 and k_n<0 and np.real(t_vec_meass_sigma[k_sigma])>=np.real(t_vec_segment[0]) and np.real(t_vec_meass_sigma[k_sigma])<=np.real(t_vec_segment[-1]):
                print('Start Sigma')
                
                k_sigma+=1

            elif k_n>=0 and k_sigma<0 and np.real(t_vec_meass_n[k_n])>=np.real(t_vec_segment[0]) and np.real(t_vec_meass_n[k_n])<=np.real(t_vec_segment[-1]):
                dos_n_vec[k_n,:]=kpm_dos_2_gpu(H_time,M_n,U,F,bounds=False)
                k_n+=1

            elif k_n>=0 and k_sigma>=0 and np.real(t_vec_meass_n[k_n])>=np.real(t_vec_segment[0]) and np.real(t_vec_meass_n[k_n])<=np.real(t_vec_segment[-1]) and np.real(t_vec_meass_sigma[k_sigma])>=np.real(t_vec_segment[0]) and np.real(t_vec_meass_sigma[k_sigma])<=np.real(t_vec_segment[-1]):
                t_vec_meass_new_sigma[k_sigma]=np.real(t_vec_segment[-1])
                t_vec_meass_new_n[k_n]=np.real(t_vec_segment[-1])
                print('Start Sigma')

                dos_n_vec[k_n,:]=kpm_dos_2_gpu(H_time,M_n,U,F,bounds=False)
                k_n+=1
                k_sigma+=1
            else:
                if (t_vec_meass_sigma[k_sigma]-t_vec_segment[-1])<(t_vec_segment[1]-t_vec_segment[0]):

                    t_vec_meass_new_sigma[k_sigma]=np.real(t_vec_segment[-1])
                    t_vec_meass_new_n[k_n]=np.real(t_vec_segment[-1])
                    print('Start Sigma')

                    dos_n_vec[k_n,:]=kpm_dos_2_gpu(H_time,M_n,U,F,bounds=False)
                    k_n+=1
                    k_sigma+=1


    print('Final Temperature : '+str(Temp)+' [K]')
    print('Final Chemical potential : '+str(mu)+' [eV]')
    print('FInal time that sigma was meassure : '+str(t_vec_meass_new_sigma)+' [fs]')
    print('FInal time that n was meassure : '+str(t_vec_meass_new_n)+' [fs]')
    np.save('./t_vec_meass',t_vec_meass_new_sigma)
    if k_sigma>=0 and k_n<0:
        return sigma_vec
    elif k_n>=0 and k_sigma<0:
        return dos_n_vec

    elif k_n>=0 and k_sigma>=0:
        return dos_n_vec,sigma_vec
    



def kpm_harmonics_gpu(H,t_vec=None,tau=None,modifier_id=None,modifier_params=None,Temp=None,mu=None,t_vec_meass=None,M=None,random_vector=None,dos_equil=None):
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
        t_vec=np.linspace(0,5*hbar_fs*M/(np.pi*H.bounds[1]),500)
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

    t_vec_meass_new=np.zeros(len(t_vec_meass))
    harmonic_vector_x=np.zeros(len(t_vec_meass),dtype=np.complex128)
    harmonic_vector_y=np.zeros(len(t_vec_meass),dtype=np.complex128)

    direction_1='x'
    direction_2='y'
    k_sigma=0

    ## Laser modifier
    A0,w,pol,Tp=modifier_params


    if pol=='r':
        modifier_params_gpu=np.array([A0,w,Tp,1],dtype=np.complex128)
    else:
        modifier_params_gpu=np.array([A0,w,Tp,-1],dtype=np.complex128)
    
    
    aux=np.zeros(len(random_vector),dtype=np.complex128)
    H_time=H_kpm.deep_copy()
    delta_t_tau=t_vec[1]/tau

    #### t==0:
    moments_kernel_FD=moments_kernel_FD.astype(np.complex128)
    t_vec=t_vec.astype(np.complex128)
    U=np.copy(random_vector,order='C')
    F=rec_A_vec(M,H_kpm,moments_kernel_FD,random_vector)
    if tau!=0:
        H_kpm.dot(1+0j,0+0j,F,aux)
        E_0=2*H_kpm.bounds[1]*np.real(vdot(1+0j,random_vector,aux,len(random_vector),H_kpm.n_threads))
        moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)    
    ####Rest of times
    t_index=np.arange(0, len(t_vec), len(t_vec)//20)
    
    if len(t_index)<len(t_vec_meass):

        t_index=np.arange(0, len(t_vec), len(t_vec)//(np.max(np.array([len(t_vec_meass),len(t_vec_meass)]))))
        
    
    
    for i in range(len(t_index)) :
        if i==0:
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[i]+1:t_index[i+1]+1],dtype=np.complex128)
            start=perf_counter()

            if tau==0:
                kpm_rho_neq_gpu_cuda(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec_segment,U,F,0,modifier_params_gpu,M,M2,np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(2,dtype=np.complex128,order='C'))
                H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])
            else :
                kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)

                H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])

                #Minimization part
                print('Chemical potential : '+str(mu)+' [eV]')
                print('Temperature : '+str(Temp)+' [K]')
                H_time.dot(1+0j,0+0j,F,aux)
                    
                E_t=2*H_time.bounds[1]*np.real(vdot(1+0j,U,aux,len(U),H_time.n_threads))
                
                deltaE=(E_t-E_0)/H.shape[0]   
                #print(deltaE)         
                mu,Temp=minimization(deltaE,dos_equil,[mu,Temp*kb],niter=10000)

                E_0=E_t
                
                moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)
        
        elif i ==(len(t_index)-1):
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[-1]+1:],dtype=np.complex128)
            
            if tau==0:
                kpm_rho_neq_gpu_cuda(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec_segment,U,F,0,modifier_params_gpu,M,M2,np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(2,dtype=np.complex128,order='C'))
            else :
                kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)

            
            H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])            

        else:
            t_vec_segment=np.ascontiguousarray(t_vec[t_index[i]+1:t_index[i+1]+1],dtype=np.complex128)


            if tau==0:
                kpm_rho_neq_gpu_cuda(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD,t_vec_segment,U,F,0,modifier_params_gpu,M,M2,np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(len(H.data),dtype=np.complex128,order='C'),np.zeros(2,dtype=np.complex128,order='C'))
                H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])            

            else :
                kpm_rho_neq_tau_gpu(H_kpm.data,H_kpm.dx_vec,H_kpm.dy_vec,H_kpm.indices,H_kpm.len_row,moments_U_vec,moments_kernel_FD_equil.astype(np.complex128),t_vec_segment,U,F,modifier_params_gpu,M,M2,delta_t_tau)

                H_time.modifier_nocop(H_kpm,modifier_hoppings_c,modifier_id,modifier_params,t_vec_segment[-1])            
                print('Chemical potential : '+str(mu)+' [eV]')
                print('Temperature : '+str(Temp)+' [K]')
                #Minimization part
                H_time.dot(1+0j,0+0j,F,aux)
                    
                E_t=2*H_time.bounds[1]*np.real(vdot(1+0j,U,aux,len(U),H_time.n_threads))
                
                deltaE=(E_t-E_0)/H.shape[0]
                
                mu,Temp=minimization(deltaE,dos_equil,[mu,Temp*kb],niter=10000)
                E_0=E_t
                moments_kernel_FD_equil=moments_FD_T(mu/H.bounds[1],Temp*kb/H.bounds[1],M)*JacksonKernel(M)
        
        V1 = H_time.modifier(modifier_velocity,direction_1)
        V2 = H_time.modifier(modifier_velocity,direction_2)

        if k_sigma<len(t_vec_meass) :
            if k_sigma>=0 and np.real(t_vec_meass[k_sigma])>=np.real(t_vec_segment[0]) and np.real(t_vec_meass[k_sigma])<=np.real(t_vec_segment[-1]):
                t_vec_meass_new[k_sigma]=np.real(t_vec_segment[-1])
                print('Start harmonic '+str(k_sigma))
                
                V1.dot(1+0j,0+0j,F,aux)
                harmonic_vector_x[k_sigma]=vdot(1+0j,U,aux,len(U),H_time.n_threads)
                
                V2.dot(1+0j,0+0j,F,aux)
                harmonic_vector_y[k_sigma]=vdot(1+0j,U,aux,len(U),H_time.n_threads)
                
                k_sigma+=1

            elif  k_sigma>=0 and np.real(t_vec_meass[k_sigma])>=np.real(t_vec_segment[0]) and np.real(t_vec_meass[k_sigma])<=np.real(t_vec_segment[-1]):
                t_vec_meass_new[k_sigma]=np.real(t_vec_segment[-1])
                print('Start harmonic '+str(k_sigma))

                V1.dot(1+0j,0+0j,F,aux)
                harmonic_vector_x[k_sigma]=vdot(1+0j,U,aux,len(U),H_time.n_threads)
                
                V2.dot(1+0j,0+0j,F,aux)
                harmonic_vector_y[k_sigma]=vdot(1+0j,U,aux,len(U),H_time.n_threads)
                
                k_sigma+=1
            else:
                if (t_vec_meass[k_sigma]-t_vec_segment[-1])<(t_vec_segment[1]-t_vec_segment[0]):

                    t_vec_meass_new[k_sigma]=np.real(t_vec_segment[-1])
                    print('Start harmonic '+str(k_sigma))

                    V1.dot(1+0j,0+0j,F,aux)
                    harmonic_vector_x[k_sigma]=vdot(1+0j,U,aux,len(U),H_time.n_threads)
                    
                    V2.dot(1+0j,0+0j,F,aux)
                    harmonic_vector_y[k_sigma]=vdot(1+0j,U,aux,len(U),H_time.n_threads)
                
                    k_sigma+=1
        if k_sigma==1:
            end=perf_counter()
            print('One iteration : '+str(end-start)+' [s]')
            print('Expected time : '+str(len(t_vec_meass)*(end-start))+' [s]')
        print(harmonic_vector_x [:50])
        print(harmonic_vector_y [:50])


    print('Final Temperature : '+str(Temp)+' [K]')
    print('Final Chemical potential : '+str(mu)+' [eV]')
    print('FInal time that sigma was meassure : '+str(t_vec_meass_new)+' [fs]')
    np.save('./t_vec_meass',t_vec_meass_new)

    return harmonic_vector_x,harmonic_vector_y
    

