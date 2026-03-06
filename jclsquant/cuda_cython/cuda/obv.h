# include <complex>
#include <thrust/complex.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

//// Blas
extern "C" __global__ void add_device(thrust::complex<double> *x, thrust::complex<double> *y, thrust::complex<double> a, thrust::complex<double> b,int N);
extern "C" __global__ void aAxby_device( thrust::complex<double> a, thrust::complex<double> b,thrust::complex<double> *x,thrust::complex<double> *y, thrust::complex<double> *data, int *indices,int len_row,int N);
#ifndef OBV_H
#define OBV_H
struct vdot {__host__ __device__ thrust::complex<double> operator()(const thrust::complex<double>& x, const thrust::complex<double>& y) const { return thrust::conj(x) * y;}};
struct exponential {const thrust::complex<double> alpha;const thrust::complex<double> beta; exponential(thrust::complex<double> _alpha, thrust::complex<double> _beta) : alpha(_alpha), beta(_beta) {} __host__ __device__ thrust::complex<double> operator()(const thrust::tuple<thrust::complex<double>, thrust::complex<double>, thrust::complex<double>>& t) const {thrust::complex<double> y = thrust::get<0>(t);thrust::complex<double> a = thrust::get<1>(t);thrust::complex<double> b = thrust::get<2>(t);return y * thrust::exp(alpha * a + beta * b);}};
#endif // OBV_H
//// Recursions
extern "C" void rec_A(thrust::complex<double>* data_raw_d, int* indices_raw_d, int len_row, int N ,thrust::device_vector<thrust::complex<double>> moments_kernel_thr,thrust::complex<double>* v_kpm_raw,int M);
extern "C" void rec_comm_A_vec(thrust::complex<double>* data_raw_d,thrust::complex<double>* data2_raw_d, int* indices_raw_d, int len_row, int N ,thrust::device_vector<thrust::complex<double>> moments_kernel_thr,thrust::complex<double>* v_kpm_raw,int M);
extern "C" void rec_A_tab(std::complex <double>* data, int* indices, int len_row, int N , std::complex <double>* A_vec,std::complex <double>* v,int M);
extern "C" void rec_A_tab_b(thrust::complex <double>* data_raw_d, int* indices_raw_d, int len_row, int N , thrust::complex<double>* A_vec_raw,thrust::device_vector<thrust::complex<double>> v_thr_d,int M);
extern "C" void rec_A_tab_2(std::complex <double>* data, int* indices, int len_row, int N , std::complex <double>* A_vec,std::complex <double>* v_l,std::complex <double>* v_r,int M);
//// Modifiers 
extern "C" void light_modifier_circular_packed_c(thrust::complex<double> A0,thrust::complex<double> t,thrust::complex<double> w,thrust::complex<double> Tp,thrust::complex<double> eta,thrust::device_vector<thrust::complex<double>>& data,thrust::device_vector<thrust::complex<double>>& dx_vec,thrust::device_vector<thrust::complex<double>>& dy_vec,int N);
extern "C"     void hopping_pos_phonons_modifier(thrust::complex<double>* data,const int* indices,thrust::complex<double>* dx_vec,thrust::complex<double>* dy_vec,thrust::complex<double> a0,thrust::complex<double> b,thrust::complex<double> Aq,const thrust::complex<double>* e_pol_x,const thrust::complex<double>* e_pol_y,thrust::complex<double> wq,thrust::complex<double> t,const thrust::complex<double>* Sx,const thrust::complex<double>* Sy,const thrust::complex<double>* q,int len_row,int size);
//// Observables
extern "C" void for_time(std::complex <double>* data, int* indices, int len_row, int N , std::complex <double>* moments_kernel,int len_t,std::complex <double>* U,std::complex <double>* U_c,int M);
extern "C" void for_time_sigma(std::complex <double>* data,std::complex <double>* data2,std::complex <double>* data3, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,std::complex <double>* moments_kernel_FD,int len_t,std::complex <double>* sigma_vec,std::complex <double>* v,int M,int M2);
extern "C" void for_time_sigma_nequil(std::complex <double>* data_H,std::complex <double>* data_V1,std::complex <double>* data_V2,std::complex <double>* data_V1_r2, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,int len_t,std::complex <double>* sigma_vec,std::complex <double>* U_t0,std::complex <double>* U_F_t0,int M,int M2);
extern "C" void kpm_rho_neq(std::complex <double>* data_H,std::complex <double>* dx_vec,std::complex <double>* dy_vec, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,std::complex <double>* moments_kernel_FD,std::complex <double>* t_vec,int len_t,std::complex <double>* U,std::complex <double>* U_F,int modifier_id,std::complex<double>* modifier_params,int len_modifer,int M,int M2,std::complex<double>* S_x,std::complex<double>* S_y,std::complex<double>* e_pol_x,std::complex<double>* e_pol_y,std::complex<double>* q);
extern "C" void kpm_rho_neq_k(std::complex <double>* data_H,std::complex <double>* dx_vec,std::complex <double>* dy_vec, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,std::complex <double>* moments_kernel_FD,std::complex <double>* t_vec,int len_t,std::complex <double>* U,std::complex <double>* U_F,int modifier_id,std::complex<double>* modifier_params,int len_modifer,int M,int M2,std::complex<double>* S_x,std::complex<double>* S_y,std::complex<double>* e_pol_x,std::complex<double>* e_pol_y,std::complex<double>* q,std::complex<double>* bounds);
extern "C" void kpm_rho_neq_tau(std::complex <double>* data_H,std::complex <double>* dx_vec,std::complex <double>* dy_vec, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,std::complex <double>* moments_kernel_FD,std::complex <double>* t_vec,int len_t,std::complex <double>* U,std::complex <double>* U_F,std::complex<double>* modifier_params,int len_modifer,int M,int M2,double delta_t_tau);
extern "C" void position_operator(std::complex <double>* data_H,std::complex <double>* data_V, int* indices, int len_row, int N , std::complex <double>* moments_kernel_G_delta,std::complex <double>* pos,std::complex <double>* v,int M);
extern "C" void angular_momentum(std::complex <double>* data_H,std::complex <double>* data_V1,std::complex <double>* data_V2, int* indices, int len_row, int N , std::complex <double>* moments_kernel_Gmas_delta, std::complex <double>* moments_kernel_Gmin_delta,std::complex <double>* pos,std::complex <double>* v,int M);
extern "C" void sigma_time_orbital(std::complex <double>* data_H,std::complex <double>* data_V1,std::complex <double>* data_V2,std::complex <double>* data_V2_kpm, int* indices, int len_row, int N , std::complex <double>* moments_kernel_Gmas_delta, std::complex <double>* moments_kernel_Gmin_delta,std::complex <double>* sigma_vec,std::complex <double>* moments_kernel_FD,std::complex <double>* mu_vec,int len_mu,std::complex <double>* moments_U,int M2,int len_t,std::complex <double>* v,int M);
extern "C" void msd(std::complex <double>* data_H,std::complex <double>* data_V, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,int len_t,std::complex <double>* msd_vec,std::complex <double>* random_vector,int M,int M2);