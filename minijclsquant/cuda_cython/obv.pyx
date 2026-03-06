# distutils: language=c++
# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython

##############

cdef extern from "cuda/obv.h":
    void rec_A_tab(double complex* data ,int* indices,int len_row,int N,double complex* A_vec,double complex* v,int M) #nogil
    void rec_A_tab_2(double complex* data ,int* indices,int len_row,int N,double complex* A_vec,double complex* v_l,double complex* v_r,int M) #nogil
    void for_time(double complex* data, int* indices, int len_row, int N , double complex* moments_kernel,int len_t,double complex* U,double complex* U_c,int M) #nogil
    void for_time_sigma(double complex* data,double complex* data2,double complex* data3, int* indices, int len_row, int N , double complex* moments_kernel_U, double complex* moments_kernel_FD,int len_t,double complex* sigma_vec,double complex* v,int M,int M2) #nogil
    void for_time_sigma_nequil(double complex* data_H,double complex* data_V1,double complex* data_V2,double complex* data_V1_r2, int* indices, int len_row, int N , double complex* moments_kernel_U,int len_t,double complex* sigma_vec,double complex* U_t0,double complex* U_F_t0,int M,int M2) #nogil
    void kpm_rho_neq(double complex* data_H,double complex* dx_vec,double complex* dy_vec, int* indices, int len_row, int N , double complex* moments_kernel_U,double complex* moments_kernel_FD,double complex* t_vec,int len_t,double complex* U,double complex* U_F,int modifier_id,double complex* modifier_params,int len_modifer,int M,int M2,double complex* S_x,double complex* S_y,double complex* e_pol_x,double complex* e_pol_y,double complex* q) #nogil
    void kpm_rho_neq_k(double complex* data_H,double complex* dx_vec,double complex* dy_vec, int* indices, int len_row, int N , double complex* moments_kernel_U,double complex* moments_kernel_FD,double complex* t_vec,int len_t,double complex* U,double complex* U_F,int modifier_id,double complex* modifier_params,int len_modifer,int M,int M2,double complex* S_x,double complex* S_y,double complex* e_pol_x,double complex* e_pol_y,double complex* q,complex* bounds) #nogil
    void kpm_rho_neq_tau(double complex* data_H,double complex* dx_vec,double complex* dy_vec, int* indices, int len_row, int N , double complex* moments_kernel_U,double complex* moments_kernel_FD,double complex* t_vec,int len_t,double complex* U,double complex* U_F,double complex* modifier_params,int len_modifer,int M,int M2,double delta_t_tau) #nogil
    void position_operator(double complex* data_H,double complex* data_V, int* indices, int len_row, int N , double complex* moments_kernel_G_delta,double complex* pos,double complex* v,int M)
    void angular_momentum(double complex* data_H,double complex* data_V1,double complex* data_V2, int* indices, int len_row, int N , double complex* moments_kernel_Gmas_delta, double complex* moments_kernel_Gmin_delta,double complex* pos,double complex* v,int M)
    void sigma_time_orbital(double complex* data_H,double complex* data_V1,double complex* data_V2,double complex* data_V2_kpm, int* indices, int len_row, int N , double complex* moments_kernel_Gmas_delta, double complex* moments_kernel_Gmin_delta,double complex* sigma_vec,double complex* moments_kernel_FD,double complex* mu_vec,int len_mu,double complex* moments_U,int M2,int len_t,double complex* v,int M)
    void msd(double complex* data_H,double complex* data_V, int* indices, int len_row, int N , double complex* moments_kernel_U,int len_t,double complex* msd_vec,double complex* random_vector,int M,int M2)

def rec_A_tab_gpu(double complex[::1] data ,int[::1] indices,int len_row,double complex[::1] v,double complex[::1] A_vec,int M): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """

    cdef int N = np.shape(v)[0];  # Assuming a, b, and c all have the same length

    # Call the CUDA function with the raw pointers
    rec_A_tab(&data[0], &indices[0], len_row,N,&A_vec[0], &v[0], M)
    
    return 

def rec_A_tab_2_gpu(double complex[::1] data ,int[::1] indices,int len_row,double complex[::1] v_l,double complex[::1] v_r,double complex[::1] A_vec,int M): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """

    cdef int N = np.shape(v_r)[0];  # Assuming a, b, and c all have the same length

    # Call the CUDA function with the raw pointers
    rec_A_tab_2(&data[0], &indices[0], len_row,N,&A_vec[0], &v_l[0],&v_r[0], M)
    
    return 


def for_time_gpu(double complex[::1] data ,int[::1] indices,int len_row,double complex[::1] moments_kernel,int len_t,double complex[::1] U,double complex[::1] U_c,int M): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """
    cdef int N = np.shape(U)[0];  # Assuming a, b, and c all have the same length
    # Call the CUDA function with the raw pointers
    for_time(&data[0], &indices[0],len_row,N , &moments_kernel[0],len_t,&U[0],&U_c[0],M)
    
    return 



def for_time_sigma_gpu(double complex[::1] data ,double complex[::1] data2,double complex[::1] data3,int[::1] indices,int len_row,double complex[::1] moments_kernel_U,double complex[::1] moments_kernel_FD,int len_t,double complex[::1] sigma_vec,double complex[::1] v,int M,int M2): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """
    cdef int N = np.shape(v)[0];  # Assuming a, b, and c all have the same length
    # Call the CUDA function with the raw pointers
    for_time_sigma(&data[0],&data2[0],&data3[0],&indices[0],len_row,N, &moments_kernel_U[0],&moments_kernel_FD[0],len_t,&sigma_vec[0],&v[0],M,M2)
    return 



def for_time_sigma_nequil_gpu(double complex[::1] data_H,double complex[::1] data_V1,double complex[::1] data_V2,double complex[::1] data_V1_r2, int[::1] indices, int len_row, double complex[::1] moments_kernel_U,int len_t,double complex[::1] sigma_vec,double complex[::1] U_t0,double complex[::1] U_F_t0,int M,int M2): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """
    cdef int N = np.shape(U_F_t0)[0];  # Assuming a, b, and c all have the same length
    # Call the CUDA function with the raw pointers
    for_time_sigma_nequil(&data_H[0],&data_V1[0],&data_V2[0],&data_V1_r2[0], &indices[0],len_row,N ,&moments_kernel_U[0],len_t,&sigma_vec[0],&U_t0[0],&U_F_t0[0],M,M2)
    return 



def kpm_rho_neq_gpu_cuda(double complex[::1] data_H,double complex[::1] dx_vec,double complex[::1] dy_vec, int[::1] indices, int len_row, double complex[::1] moments_kernel_U,double complex[::1] moments_kernel_FD,double complex[::1] t_vec,double complex[::1] U,double complex[::1] U_F,int modifier_id,double complex[::1] modifier_params,int M,int M2,double complex[::1] S_x,double complex[::1] S_y,double complex[::1] e_pol_x,double complex[::1] e_pol_y,double complex[::1] q): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """
    cdef int N = np.shape(U_F)[0];  
    cdef int len_t = len(t_vec);  
    cdef int len_modifier = len(modifier_params);  

    # Call the CUDA function with the raw pointers
    kpm_rho_neq(&data_H[0],&dx_vec[0],&dy_vec[0],&indices[0],len_row,N ,&moments_kernel_U[0],&moments_kernel_FD[0],&t_vec[0],len_t,&U[0],&U_F[0],modifier_id,&modifier_params[0],len_modifier,M,M2,&S_x[0],&S_y[0],&e_pol_x[0],&e_pol_y[0],&q[0])    
    
    return 


def kpm_rho_neq_gpu_cuda_k(double complex[::1] data_H,double complex[::1] dx_vec,double complex[::1] dy_vec, int[::1] indices, int len_row, double complex[::1] moments_kernel_U,double complex[::1] moments_kernel_FD,double complex[::1] t_vec,double complex[::1] U,double complex[::1] U_F,int modifier_id,double complex[::1] modifier_params,int M,int M2,double complex[::1] S_x,double complex[::1] S_y,double complex[::1] e_pol_x,double complex[::1] e_pol_y,double complex[::1] q,double complex[::1] bounds): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """
    cdef int N = np.shape(U_F)[0];  
    cdef int len_t = len(t_vec);  
    cdef int len_modifier = len(modifier_params);  

    # Call the CUDA function with the raw pointers
    kpm_rho_neq_k(&data_H[0],&dx_vec[0],&dy_vec[0],&indices[0],len_row,N ,&moments_kernel_U[0],&moments_kernel_FD[0],&t_vec[0],len_t,&U[0],&U_F[0],modifier_id,&modifier_params[0],len_modifier,M,M2,&S_x[0],&S_y[0],&e_pol_x[0],&e_pol_y[0],&q[0],&bounds[0])    
    
    return 

def kpm_rho_neq_tau_gpu(double complex[::1] data_H,double complex[::1] dx_vec,double complex[::1] dy_vec, int[::1] indices, int len_row, double complex[::1] moments_kernel_U,double complex[::1] moments_kernel_FD,double complex[::1] t_vec,double complex[::1] U,double complex[::1] U_F,double complex[::1] modifier_params,int M,int M2,double delta_t_tau): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """
    cdef int N = np.shape(U_F)[0];  
    cdef int len_t = len(t_vec);  
    cdef int len_modifier = len(modifier_params);  

    # Call the CUDA function with the raw pointers
    kpm_rho_neq_tau(&data_H[0],&dx_vec[0],&dy_vec[0],&indices[0],len_row,N ,&moments_kernel_U[0],&moments_kernel_FD[0],&t_vec[0],len_t,&U[0],&U_F[0],&modifier_params[0],len_modifier,M,M2,delta_t_tau)    
    
    return 

def kpm_position_operator_gpu(double complex[::1] data_H,double complex[::1] data_V, int[::1] indices, int len_row, double complex[::1] moments_kernel_G_delta,double complex[::1] pos,double complex[::1] v,int M): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """
    cdef int N = np.shape(v)[0];  


    # Call the CUDA function with the raw pointers
    position_operator(&data_H[0],&data_V[0], &indices[0], len_row, N , &moments_kernel_G_delta[0],&pos[0],&v[0],M)
    return 

def kpm_angular_momentum_gpu_c(double complex[::1] data_H,double complex[::1] data_V1,double complex[::1] data_V2, int[::1] indices, int len_row, double complex[::1] moments_kernel_Gmas_delta, double complex[::1] moments_kernel_Gmin_delta,double complex[::1] pos,double complex[::1] v,int M): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """
    cdef int N = np.shape(v)[0];  


    # Call the CUDA function with the raw pointers
    angular_momentum(&data_H[0],&data_V1[0],&data_V2[0], &indices[0],len_row, N ,&moments_kernel_Gmas_delta[0], &moments_kernel_Gmin_delta[0],&pos[0],&v[0],M)

    return 

def kpm_sigma_time_orbital_c(double complex[::1] data_H,double complex[::1] data_V1,double complex[::1] data_V2,double complex[::1] data_V2_kpm, int[::1] indices, int len_row, double complex[::1] moments_kernel_Gmas_delta, double complex[::1] moments_kernel_Gmin_delta,double complex[::1] sigma_vec,double complex[::1] moments_kernel_FD,double complex[::1] mu_vec,len_mu,double complex[::1] moments_U,M2,len_t,double complex[::1] v,int M): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """
    cdef int N = np.shape(v)[0];  

    # Call the CUDA function with the raw pointers
    sigma_time_orbital(&data_H[0],&data_V1[0],&data_V2[0],&data_V2_kpm[0], &indices[0], len_row, N , &moments_kernel_Gmas_delta[0], &moments_kernel_Gmin_delta[0],&sigma_vec[0],&moments_kernel_FD[0],&mu_vec[0],len_mu,&moments_U[0],M2,len_t,&v[0],M)
    return 

def kpm_msd_gpu_c(double complex[::1] data_H,double complex[::1] data_V, int[::1] indices, int len_row, double complex[::1] moments_kernel_U, int len_t,double complex[::1] msd_vec,double complex[::1] v,int M,int M2): 
    """
    
    Make the operation y=aAx+by the GPU via CUDA
    
    """
    cdef int N = np.shape(v)[0];  

    # Call the CUDA function with the raw pointers
    msd(&data_H[0],&data_V[0], &indices[0], len_row,N , &moments_kernel_U[0],len_t,&msd_vec[0],&v[0],M,M2)
    return 





