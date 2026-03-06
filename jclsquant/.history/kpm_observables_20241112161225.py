
import scipy as sci
import numba as numba
import numpy as np
from time import perf_counter
import copy
import tqdm
from math import pi,sqrt




def delta_KPM_calc_two_vec_v4(Ef,H,M,bounds_array,random_vector_l,random_vector_r):
    Ef=(Ef-(bounds_array[1]+bounds_array[0])/2)/((bounds_array[1]-bounds_array[0])/2)
    moments_jackson_delta_vec=jl.moments_delta(Ef,M)*jl.JacksonKernel(M)
    delta_0=sci.sparse.identity(H.shape[0]).dot(random_vector_r)
    delta_1=H.dot(random_vector_r)
    delta_vec=moments_jackson_delta_vec[0]*delta_0
    delta_vec+=moments_jackson_delta_vec[1]*delta_1

    for m2 in range(2,M):
        delta_1,delta_0=2*H.dot(delta_1)-delta_0,delta_1
        delta_vec+=moments_jackson_delta_vec[m2]*delta_1


    delta=np.real(np.vdot(random_vector_l,delta_vec))
    return delta


def delta_KPM_two_vec(H,M,bounds_array,random_vector_l,random_vector_r,Ef_vec):
    """
    Def:
        Mean value of the DOS with two different vectors 
    Inputs:
        H: Hamiltonian in CSR or Ell.
        M: Number of moments.
        bounds_array: array of bounds such that bounds_array[0]=-bounds_array[1].
        random_vector_l: random vector of the left (will be conjugated).
        random_vector_r: random vector of the right.
        Ef_vec: Fermi energies where the DOS will be computed, it will be reescaled with bounds.
        D_data: Array of the difference in position of each amtrix element. 
    Outputs:
        delta_mat: Array of shape (len(Ef_vec),2) where [:,0] are the unormalized Fermi energies and [:,1] the DOS.
    """
    delta_mat=np.zeros((len(Ef_vec),2))
    delta_mat[:,0]=Ef_vec
    for i in range(len(Ef_vec)):
        delta_mat[i,1]=delta_KPM_calc_two_vec_v4(Ef_vec[i],H,M,bounds_array,random_vector_l,random_vector_r)
    return delta_mat





def sigma_time_optical_T_Htime_ELL(H_ell_KPM,V1_ell,V2_ell,D1_data,D2_data,mod_params,Ef,tau_vec,T,M,eta,random_vector_list,repetitions,bounds_array):
    kb=8.617333*1e-5 #in [ev*K^-1]
    hbar=6.582e-16 # in [eV*s]
    ##Start of timers
    time_U_tot=0
    time_conmutator_tot=0
    time_H_tot=0

    ##Reescalations and velocities
        
    tau_vec2=tau_vec/hbar
    Ef=(Ef-(bounds_array[1]+bounds_array[0])/2)/((bounds_array[1]-bounds_array[0])/2)
    T_mod=(T*kb-(bounds_array[1]+bounds_array[0])/2)/((bounds_array[1]-bounds_array[0])/2)


    sigma_vec=np.zeros((repetitions,len(tau_vec)-1),dtype=complex)


    for i in tqdm(range(len(tau_vec)-1)):
        start_H=perf_counter()
        
        
        H_ell_time_KPM=H_ell_KPM.modifier(circle_light,mod_params,tau_vec2[i],D1_data,D2_data)
        V1_ell_time=V1_ell.modifier(circle_light,mod_params,tau_vec2[i],D1_data,D2_data)
        V2_ell_time=V2_ell.modifier(circle_light,mod_params,tau_vec2[i],D1_data,D2_data)
        
        end_H=perf_counter()
        time_H_tot+=end_H-start_H



        for j in range(repetitions):
            if j==0 and i==0:
                U_vec=np.zeros(((repetitions),np.shape(H_ell_KPM)[0]),dtype=complex)
                VU_vec=np.zeros(((repetitions),np.shape(H_ell_KPM)[0]),dtype=complex)
            if i==0:
                random_vector=random_vector_list[j]
                U_vec[j,:]=random_vector
                VU_vec[j,:]=V1_ell_time.dot(random_vector)

            sigma_vec[j,i],U_vec[j,:],VU_vec[j,:],time_U,time_conmutator=sigma_time_singleE_optical_T_Htime_ELL(H_ell_time_KPM,V1_ell_time,V2_ell_time,Ef,tau_vec[i+1]-tau_vec[i],tau_vec2[i],T,M,eta,U_vec[j,:],VU_vec[j,:],bounds_array)
            time_U_tot+=time_U
            time_conmutator_tot+=time_conmutator

    print('Time it takes to compute the H : '+str(round(time_H_tot/60,2))+' min')
    print('Time it takes to compute the U : '+str(round(time_U_tot/60,2))+' min')
    print('Time it takes to compute the conmutator : '+str(round(time_conmutator_tot/60,2))+' min')
    
    return sigma_vec
def sigma_time_singleE_optical_T_Htime_ELL(H_ell_time_KPM,V1_ell_time,V2_ell_time,Ef,delta_tau,tau2,T_mod,M,eta,vector_U,vector_VU,bounds_array):
    ## Reescalation of the quantities according to the bounds
    delta_tau=((bounds_array[1]-bounds_array[0])/2)*delta_tau/hbar

    ## Computations of the Kernels and moments 

    if T_mod==0:
        moments_jackson_FD_vec=moments_FD(Ef,M)*JacksonKernel(M)
    else:
        moments_jackson_FD_vec=moments_FD_T(Ef,T_mod,M)*JacksonKernel(M)
        
    moments_jackson_U_vec=moments_U(delta_tau,M)#*JacksonKernel(M) 
    
    start=perf_counter()


    U_dag_V_0=vector_VU
    U_dag_V_1=H_ell_time_KPM.dot(vector_VU)
    

    U_0=vector_U
    U_1=H_ell_time_KPM.dot(vector_U)
    
    U_dag_V=moments_jackson_U_vec[0]*U_dag_V_0
    U_dag_V+=moments_jackson_U_vec[1]*U_dag_V_1

    U_dag=np.conj(moments_jackson_U_vec[0])*U_0
    U_dag+=np.conj(moments_jackson_U_vec[1])*U_1
    
    U=moments_jackson_U_vec[0]*U_0
    U+=moments_jackson_U_vec[1]*U_1
    for m2 in range(2,M):
        U_dag_V_1,U_dag_V_0=2*(H_ell_time_KPM.dot(U_dag_V_1))-U_dag_V_0,U_dag_V_1
        U_1,U_0=2*(H_ell_time_KPM.dot(U_1))-U_0,U_1
        
        U_dag+=np.conj(moments_jackson_U_vec[m2])*U_1
        U+=(moments_jackson_U_vec[m2])*U_1
        U_dag_V+=np.conj(moments_jackson_U_vec[m2])*U_dag_V_1

    end=perf_counter()
    time_U=end-start
    start=perf_counter()
    
    T_0=U_dag_V
    T_1=H_ell_time_KPM.dot(U_dag_V)

    conmutator_mat_0=0*U_dag_V
    conmutator_mat_1=V2_ell_time.dot(U_dag_V)

    conmutator=moments_jackson_FD_vec[0]*conmutator_mat_0
    conmutator+=moments_jackson_FD_vec[1]*conmutator_mat_1
    
    for m2 in range(2,M):
        conmutator_mat_1,conmutator_mat_0=(2 * (V2_ell_time.dot(T_1)) + 2 * (H_ell_time_KPM.dot(conmutator_mat_1)) - conmutator_mat_0),conmutator_mat_1
        T_1,T_0=2*(H_ell_time_KPM.dot(T_1))-T_0,T_1
        conmutator+=moments_jackson_FD_vec[m2]*conmutator_mat_1
    
    sigma=1j*np.vdot(U_dag,conmutator)
    end=perf_counter()
    time_conmutator=end-start
    return sigma*np.exp(-eta*tau2),U,V1_ell_time.dot(U),time_U,time_conmutator


