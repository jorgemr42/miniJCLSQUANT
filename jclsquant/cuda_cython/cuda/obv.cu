#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include "thrust/inner_product.h"
#include "thrust/functional.h"
#include <thrust/for_each.h>

#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <iostream>
#include <complex>
#include "obv.h"



extern "C"{


    void msd(std::complex <double>* data_H,std::complex <double>* data_V, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,int len_t,std::complex <double>* msd_vec,std::complex <double>* random_vector,int M,int M2){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;

        // Allocate device vectors of the operator and copy them in device 
        thrust::device_vector<thrust::complex<double>> data_H_thr(data_H, data_H + len_data);
        thrust::device_vector<thrust::complex<double>> data_V_thr(data_V, data_V + len_data);
        thrust::device_vector<int> indices_thr(indices, indices + len_data);

        thrust::device_vector<thrust::complex<double>> random_vector_thr(random_vector, random_vector + N);
        thrust::device_vector<thrust::complex<double>> moments_kernel_U_thr(moments_kernel_U, moments_kernel_U + M2);
        thrust::device_vector<thrust::complex<double>> msd_vec_thr(msd_vec, msd_vec + M*len_t);
        
        //Auxiliary vectors for the recursion
        
        thrust::device_vector<thrust::complex<double>> msd_vec_slice_thr(M);
        thrust::device_vector<thrust::complex<double>> aux_thr(N);


        // Create raw pointers that can be used by the Kernel
        thrust::complex<double> *data_H_raw = thrust::raw_pointer_cast(data_H_thr.data());
        thrust::complex<double> *data_V_raw = thrust::raw_pointer_cast(data_V_thr.data());
        int *indices_raw = thrust::raw_pointer_cast(indices_thr.data());
        thrust::complex<double> *msd_vec_slice_raw = thrust::raw_pointer_cast(msd_vec_slice_thr.data());


        //Create the vectors U and mean_x for the recursion and initialize them with random vector
        thrust::device_vector<thrust::complex<double>> mean_x_thr = random_vector_thr;
        thrust::device_vector<thrust::complex<double>> U_thr = random_vector_thr;

        thrust::complex<double> *mean_x_raw = thrust::raw_pointer_cast(mean_x_thr.data());
        thrust::complex<double> *U_raw = thrust::raw_pointer_cast(U_thr.data());
        thrust::complex<double> *aux_raw = thrust::raw_pointer_cast(aux_thr.data());

        //Iteration t=1 

        rec_comm_A_vec(data_H_raw,data_V_raw,indices_raw,len_row,N ,moments_kernel_U_thr,mean_x_raw,M2);
        

        rec_A_tab_b(data_H_raw,indices_raw,len_row,N,msd_vec_slice_raw,mean_x_thr,M);

        thrust::copy(msd_vec_slice_thr.begin(),msd_vec_slice_thr.end(),msd_vec_thr.begin()+M);            
        
        rec_A(data_H_raw,indices_raw,len_row,N ,moments_kernel_U_thr,U_raw,M2);
        



        for(int i = 2 ; i < len_t ; i++){

            rec_A(data_H_raw,indices_raw,len_row,N ,moments_kernel_U_thr,mean_x_raw,M2);
            
            thrust::copy(U_thr.begin(),U_thr.end(),aux_thr.begin());            
            rec_comm_A_vec(data_H_raw,data_V_raw,indices_raw,len_row,N ,moments_kernel_U_thr,aux_raw,M2);
            
            add_device<<<n_blocks, n_threads>>>(aux_raw, mean_x_raw, thrust::complex<double>(1.0, 0.0), thrust::complex<double>(1.0, 0.0),N);
            
            rec_A_tab_b(data_H_raw,indices_raw,len_row,N,msd_vec_slice_raw,mean_x_thr,M);

            thrust::copy(msd_vec_slice_thr.begin(),msd_vec_slice_thr.end(),msd_vec_thr.begin()+i*M);            
        
            rec_A(data_H_raw,indices_raw,len_row,N ,moments_kernel_U_thr,U_raw,M2);

        }

        thrust::copy(msd_vec_thr.begin(), msd_vec_thr.end(), msd_vec); 

    }




    void for_time(std::complex <double>* data, int* indices, int len_row, int N , std::complex <double>* moments_kernel,int len_t,std::complex <double>* U,std::complex <double>* U_c,int M){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;

        // Allocate device vectors of the operator and copy them in device 
        thrust::device_vector<thrust::complex<double>> data_thr_d(data, data + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);

        thrust::device_vector<thrust::complex<double>> U_thr_d(U, U + N);
        thrust::device_vector<thrust::complex<double>> U_c_thr_d(U_c, U_c + N);
        thrust::device_vector<thrust::complex<double>> moments_kernel_thr(moments_kernel, moments_kernel + M);


        // Create raw pointers that can be used by the Kernel
        thrust::complex<double> *data_raw_d = thrust::raw_pointer_cast(data_thr_d.data());
        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *U_raw_d = thrust::raw_pointer_cast(U_thr_d.data());
        thrust::complex<double> *U_c_raw_d = thrust::raw_pointer_cast(U_c_thr_d.data());

        for(int i = 0 ; i < len_t ; i++){
            rec_A(data_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr,U_raw_d,M);
            rec_A(data_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr,U_c_raw_d,M);

        }

        thrust::copy(U_thr_d.begin(), U_thr_d.end(), U); 
        thrust::copy(U_c_thr_d.begin(), U_c_thr_d.end(), U_c); 

    }

    void for_time_sigma(std::complex <double>* data_H,std::complex <double>* data_V1,std::complex <double>* data_V2, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,std::complex <double>* moments_kernel_FD,int len_t,std::complex <double>* sigma_vec,std::complex <double>* v,int M,int M2){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;

        // Allocate device vectors of the operator and copy them in device 
        thrust::device_vector<thrust::complex<double>> data_H_thr_d(data_H, data_H + len_data);
        thrust::device_vector<thrust::complex<double>> data_V1_thr_d(data_V1, data_V1 + len_data);
        thrust::device_vector<thrust::complex<double>> data_V2_thr_d(data_V2, data_V2 + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);

        thrust::device_vector<thrust::complex<double>> v_thr_d(v,v+N);
        thrust::device_vector<thrust::complex<double>> U_thr_d(v_thr_d);
        thrust::device_vector<thrust::complex<double>> U_c_thr_d(v_thr_d);


        thrust::device_vector<thrust::complex<double>> moments_kernel_thr_FD(moments_kernel_FD, moments_kernel_FD + M);
        thrust::device_vector<thrust::complex<double>> moments_kernel_thr_U(moments_kernel_U, moments_kernel_U + M2);
        thrust::device_vector<thrust::complex<double>> sigma_vec_thr(sigma_vec, sigma_vec + len_t);


        // Create raw pointers that can be used by the Kernel
        thrust::complex<double> *data_H_raw_d = thrust::raw_pointer_cast(data_H_thr_d.data());
        thrust::complex<double> *data_V1_raw_d = thrust::raw_pointer_cast(data_V1_thr_d.data());
        thrust::complex<double> *data_V2_raw_d = thrust::raw_pointer_cast(data_V2_thr_d.data());
        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *U_raw_d = thrust::raw_pointer_cast(U_thr_d.data());
        thrust::complex<double> *U_c_raw_d = thrust::raw_pointer_cast(U_c_thr_d.data());

        //Create auxilary vector
        thrust::device_vector<thrust::complex<double>> aux_thr_d(N);
        thrust::complex<double> *aux_raw_d = thrust::raw_pointer_cast(aux_thr_d.data());

        // //Loop
        //Iteration i==0
        rec_comm_A_vec(data_H_raw_d,data_V2_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_FD,U_c_raw_d,M);
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), U_raw_d,aux_raw_d, data_V1_raw_d, indices_raw_d, len_row, N);
        sigma_vec_thr[0] = thrust::inner_product(U_c_thr_d.begin(), U_c_thr_d.end(), aux_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());            
        //Rest of the iterations
        for(int i = 1 ; i < len_t ; i++){
            rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_raw_d,M2);
            rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_c_raw_d,M2);

            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), U_raw_d,aux_raw_d, data_V1_raw_d, indices_raw_d, len_row, N);
            sigma_vec_thr[i] = thrust::inner_product(U_c_thr_d.begin(), U_c_thr_d.end(), aux_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());            

        }

        thrust::copy(sigma_vec_thr.begin(), sigma_vec_thr.end(), sigma_vec); 

    }
    void for_time_sigma_nequil(std::complex <double>* data_H,std::complex <double>* data_V1,std::complex <double>* data_V2,std::complex <double>* data_V1_r2, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,int len_t,std::complex <double>* sigma_vec,std::complex <double>* U_t0,std::complex <double>* U_F_t0,int M,int M2){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;

        // Allocate device vectors of the operator and copy them in device 
        thrust::device_vector<thrust::complex<double>> data_H_thr_d(data_H, data_H + len_data);
        thrust::device_vector<thrust::complex<double>> data_V1_thr_d(data_V1, data_V1 + len_data);
        thrust::device_vector<thrust::complex<double>> data_V2_thr_d(data_V2, data_V2 + len_data);
        thrust::device_vector<thrust::complex<double>> data_V1_r2_thr_d(data_V1_r2, data_V1_r2 + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);

        thrust::device_vector<thrust::complex<double>> U_F_thr_d(U_F_t0,U_F_t0+N);
        thrust::device_vector<thrust::complex<double>> mean_x_thr_d(U_t0,U_t0+N);
        thrust::device_vector<thrust::complex<double>> U_thr_d(U_t0,U_t0+N);

        thrust::device_vector<thrust::complex<double>> moments_kernel_thr_U(moments_kernel_U, moments_kernel_U + M2);
        thrust::device_vector<thrust::complex<double>> sigma_vec_thr(sigma_vec, sigma_vec + len_t);


        // Create raw pointers that can be used by the Kernel
        thrust::complex<double> *data_H_raw_d = thrust::raw_pointer_cast(data_H_thr_d.data());
        thrust::complex<double> *data_V1_raw_d = thrust::raw_pointer_cast(data_V1_thr_d.data());
        thrust::complex<double> *data_V2_raw_d = thrust::raw_pointer_cast(data_V2_thr_d.data());
        thrust::complex<double> *data_V1_r2_raw_d = thrust::raw_pointer_cast(data_V1_r2_thr_d.data());
        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *U_F_raw_d = thrust::raw_pointer_cast(U_F_thr_d.data());
        thrust::complex<double> *mean_x_raw_d = thrust::raw_pointer_cast(mean_x_thr_d.data());
        thrust::complex<double> *U_raw_d = thrust::raw_pointer_cast(U_thr_d.data());

        //Create auxilary vector
        thrust::device_vector<thrust::complex<double>> aux_thr_d(N);
        thrust::complex<double> *aux_raw_d = thrust::raw_pointer_cast(aux_thr_d.data());

        thrust::device_vector<thrust::complex<double>> aux2_thr_d(N);
        thrust::complex<double> *aux2_raw_d = thrust::raw_pointer_cast(aux2_thr_d.data());
        thrust::complex<double> term;
        thrust::complex<double> term2;

        thrust::device_vector<thrust::complex<double>> U_aux_thr_d(N);
        thrust::complex<double> *U_aux_raw_d = thrust::raw_pointer_cast(U_aux_thr_d.data());
        // //Loop

        //Iteration i==0


        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), mean_x_raw_d,aux_raw_d, data_V1_raw_d, indices_raw_d, len_row, N);
        term = thrust::inner_product(U_F_thr_d.begin(), U_F_thr_d.end(), aux_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), U_raw_d,aux2_raw_d, data_V1_r2_raw_d, indices_raw_d, len_row, N);
        term2=thrust::inner_product(U_F_thr_d.begin(), U_F_thr_d.end(), aux2_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());
        sigma_vec_thr[0]=(-term+thrust::conj(term)+term2);
        
        
        // //Rest of the iterations
        for(int i = 1 ; i < len_t ; i++){
            
            // Update of Mean_x
            thrust::copy(U_thr_d.begin(), U_thr_d.end(), U_aux_thr_d.begin()); 

            rec_comm_A_vec(data_H_raw_d,data_V2_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_aux_raw_d,M2);
            rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,mean_x_raw_d,M2);
            add_device<<<n_blocks, n_threads>>>(U_aux_raw_d, mean_x_raw_d, thrust::complex<double>(1.0, 0.0), thrust::complex<double>(1.0, 0.0),N);
            // Update of U
            rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_raw_d,M2);
            // Update of U_F
            rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_F_raw_d,M2);

            // Sum to the conducitivity
            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), mean_x_raw_d,aux_raw_d, data_V1_raw_d, indices_raw_d, len_row, N);
            term = thrust::inner_product(U_F_thr_d.begin(), U_F_thr_d.end(), aux_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                        thrust::plus<thrust::complex<double>>(),  vdot());
            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), U_raw_d,aux2_raw_d, data_V1_r2_raw_d, indices_raw_d, len_row, N);
            term2=thrust::inner_product(U_F_thr_d.begin(), U_F_thr_d.end(), aux2_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                        thrust::plus<thrust::complex<double>>(),  vdot());
            sigma_vec_thr[i]=(-term+thrust::conj(term)+term2);

        }

        thrust::copy(sigma_vec_thr.begin(), sigma_vec_thr.end(), sigma_vec); 

    }


   void kpm_rho_neq(std::complex <double>* data_H,std::complex <double>* dx_vec,std::complex <double>* dy_vec, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,std::complex <double>* moments_kernel_FD,std::complex <double>* t_vec,int len_t,std::complex <double>* U,std::complex <double>* U_F,int modifier_id,std::complex<double>* modifier_params,int len_modifer,int M,int M2,std::complex<double>* S_x,std::complex<double>* S_y,std::complex<double>* e_pol_x,std::complex<double>* e_pol_y,std::complex<double>* q){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;

        // Allocate device vectors of the operator and copy them in device 
        thrust::device_vector<thrust::complex<double>> data_H_thr_d(data_H, data_H + len_data);
        thrust::device_vector<thrust::complex<double>> dx_vec_thr_d(dx_vec, dx_vec + len_data);
        thrust::device_vector<thrust::complex<double>> dy_vec_thr_d(dy_vec, dy_vec + len_data);
        thrust::device_vector<thrust::complex<double>> dx_vec_time_thr_d(dx_vec, dx_vec + len_data);
        thrust::device_vector<thrust::complex<double>> dy_vec_time_thr_d(dy_vec, dy_vec + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);
        thrust::device_vector<thrust::complex<double>> e_pol_x_thr_d(e_pol_x, e_pol_x + N);
        thrust::device_vector<thrust::complex<double>> e_pol_y_thr_d(e_pol_y, e_pol_y + N);
        thrust::device_vector<thrust::complex<double>> S_y_thr_d(S_y, S_y + N);
        thrust::device_vector<thrust::complex<double>> S_x_thr_d(S_x, S_x + N);
        thrust::device_vector<thrust::complex<double>> q_thr_d(q, q + 2);

        thrust::device_vector<thrust::complex<double>> U_thr_d(U,U+N); //initialize U with v
        thrust::device_vector<thrust::complex<double>> U_F_thr_d(U_F,U_F+N); // Empty U_F

        thrust::device_vector<thrust::complex<double>> moments_kernel_thr_U(moments_kernel_U, moments_kernel_U + M2);
        thrust::device_vector<thrust::complex<double>> moments_kernel_thr_FD(moments_kernel_FD, moments_kernel_FD + M);
        thrust::device_vector<thrust::complex<double>> t_vec_thr(t_vec,t_vec + len_t);
        thrust::device_vector<thrust::complex<double>> modifier_params_thr(modifier_params, modifier_params + len_modifer);


        // // Create raw pointers that can be used by the Kernel
        thrust::complex<double> *data_H_raw_d = thrust::raw_pointer_cast(data_H_thr_d.data());
        thrust::complex<double> *dx_vec_raw_d = thrust::raw_pointer_cast(dx_vec_thr_d.data());        
        thrust::complex<double> *dy_vec_raw_d = thrust::raw_pointer_cast(dy_vec_thr_d.data());        
        thrust::complex<double> *dx_vec_time_raw_d = thrust::raw_pointer_cast(dx_vec_time_thr_d.data());
        thrust::complex<double> *dy_vec_time_raw_d = thrust::raw_pointer_cast(dy_vec_time_thr_d.data());
        thrust::complex<double> *t_vec_raw_d = thrust::raw_pointer_cast(t_vec_thr.data());
        thrust::complex<double> *e_pol_x_raw_d = thrust::raw_pointer_cast(e_pol_x_thr_d.data());
        thrust::complex<double> *e_pol_y_raw_d = thrust::raw_pointer_cast(e_pol_y_thr_d.data());
        thrust::complex<double> *S_x_raw_d = thrust::raw_pointer_cast(S_x_thr_d.data());
        thrust::complex<double> *S_y_raw_d = thrust::raw_pointer_cast(S_y_thr_d.data());
        thrust::complex<double> *q_raw_d = thrust::raw_pointer_cast(q_thr_d.data());
        thrust::complex<double> *modifier_params_raw_d = thrust::raw_pointer_cast(modifier_params_thr.data());
        



        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *U_F_raw_d = thrust::raw_pointer_cast(U_F_thr_d.data());
        thrust::complex<double> *U_raw_d = thrust::raw_pointer_cast(U_thr_d.data());


        // //Loop

        //Iteration i==0
        
        thrust::device_vector<thrust::complex<double>> data_H_time_thr_d(data_H, data_H + len_data);
        thrust::complex<double> *data_H_time_raw_d = thrust::raw_pointer_cast(data_H_time_thr_d.data());


        if (modifier_id == 0){
        // modifier_params=[A0,w,Tp,eta]

            light_modifier_circular_packed_c( modifier_params_thr[0],t_vec_thr[0],modifier_params_thr[1],modifier_params_thr[2],modifier_params_thr[3],data_H_time_thr_d,dx_vec_thr_d,dy_vec_thr_d,len_data);
        }
        else if (modifier_id ==1){

            //modifier_params=thrust::complex<double> a0,thrust::complex<double> b,thrust::complex<double> Aq,const thrust::complex<double>* e_pol_x,const thrust::complex<double>* e_pol_y,thrust::complex<double> wq,double t,int* Sx,int* Sy,thrust::complex<double>* q
            hopping_pos_phonons_modifier(data_H_time_raw_d,indices_raw_d,dx_vec_raw_d,dy_vec_raw_d,modifier_params_thr[0],modifier_params_thr[1],modifier_params_thr[2],e_pol_x_raw_d,e_pol_y_raw_d,modifier_params_thr[3],t_vec_thr[0],S_x_raw_d,S_y_raw_d,q_raw_d,len_row,len_data);                        
        }

        rec_A(data_H_time_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_FD,U_F_raw_d,M);
        
        
        // // //Rest of the iterations
        for(int i = 1 ; i < len_t ; i++){
            thrust::copy(data_H_thr_d.begin(), data_H_thr_d.end(), data_H_time_thr_d.begin()); 


            if (modifier_id == 0){

                light_modifier_circular_packed_c( modifier_params_thr[0],t_vec_thr[i],modifier_params_thr[1],modifier_params_thr[2],modifier_params_thr[3],data_H_time_thr_d,dx_vec_thr_d,dy_vec_thr_d,len_data);
            }
            else if (modifier_id ==1){
                thrust::copy(dx_vec_thr_d.begin(), dx_vec_thr_d.end(), dx_vec_time_thr_d.begin()); 
                thrust::copy(dy_vec_thr_d.begin(), dy_vec_thr_d.end(), dy_vec_time_thr_d.begin()); 
                //modifier_params=thrust::complex<double> a0,thrust::complex<double> b,thrust::complex<double> Aq,const thrust::complex<double>* e_pol_x,const thrust::complex<double>* e_pol_y,thrust::complex<double> wq,double t,int* Sx,int* Sy,thrust::complex<double>* q
                hopping_pos_phonons_modifier(data_H_time_raw_d,indices_raw_d,dx_vec_time_raw_d,dy_vec_time_raw_d,modifier_params_thr[0],modifier_params_thr[1],modifier_params_thr[2],e_pol_x_raw_d,e_pol_y_raw_d,modifier_params_thr[3],t_vec_thr[i],S_x_raw_d,S_y_raw_d,q_raw_d,len_row,len_data);                        

            }
            
            rec_A(data_H_time_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_F_raw_d,M2);
            rec_A(data_H_time_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_raw_d,M2);
            

        }


        thrust::copy(U_thr_d.begin(),U_thr_d.end(), U); 
        thrust::copy(U_F_thr_d.begin(),U_F_thr_d.end(), U_F); 


    }

   void kpm_rho_neq_k(std::complex <double>* data_H,std::complex <double>* dx_vec,std::complex <double>* dy_vec, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,std::complex <double>* moments_kernel_FD,std::complex <double>* t_vec,int len_t,std::complex <double>* U,std::complex <double>* U_F,int modifier_id,std::complex<double>* modifier_params,int len_modifer,int M,int M2,std::complex<double>* S_x,std::complex<double>* S_y,std::complex<double>* e_pol_x,std::complex<double>* e_pol_y,std::complex<double>* q,std::complex<double>* bounds){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;

        // Allocate device vectors of the operator and copy them in device 
        thrust::device_vector<thrust::complex<double>> data_H_thr_d(data_H, data_H + len_data);
        thrust::device_vector<thrust::complex<double>> dx_vec_thr_d(dx_vec, dx_vec + len_data);
        thrust::device_vector<thrust::complex<double>> dy_vec_thr_d(dy_vec, dy_vec + len_data);
        thrust::device_vector<thrust::complex<double>> dx_vec_time_thr_d(dx_vec, dx_vec + len_data);
        thrust::device_vector<thrust::complex<double>> dy_vec_time_thr_d(dy_vec, dy_vec + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);
        thrust::device_vector<thrust::complex<double>> e_pol_x_thr_d(e_pol_x, e_pol_x + N);
        thrust::device_vector<thrust::complex<double>> e_pol_y_thr_d(e_pol_y, e_pol_y + N);
        thrust::device_vector<thrust::complex<double>> S_y_thr_d(S_y, S_y + N);
        thrust::device_vector<thrust::complex<double>> S_x_thr_d(S_x, S_x + N);
        thrust::device_vector<thrust::complex<double>> q_thr_d(q, q + 2);

        thrust::device_vector<thrust::complex<double>> U_thr_d(U,U+N); //initialize U with v
        thrust::device_vector<thrust::complex<double>> U_F_thr_d(U_F,U_F+N); // Empty U_F

        thrust::device_vector<thrust::complex<double>> moments_kernel_thr_U(moments_kernel_U, moments_kernel_U + M2);
        thrust::device_vector<thrust::complex<double>> moments_kernel_thr_FD(moments_kernel_FD, moments_kernel_FD + M);
        thrust::device_vector<thrust::complex<double>> t_vec_thr(t_vec,t_vec + len_t);
        thrust::device_vector<thrust::complex<double>> modifier_params_thr(modifier_params, modifier_params + len_modifer);
        thrust::device_vector<thrust::complex<double>> bounds_thr(bounds, bounds + 2);


        // // Create raw pointers that can be used by the Kernel
        thrust::complex<double> *data_H_raw_d = thrust::raw_pointer_cast(data_H_thr_d.data());
        thrust::complex<double> *dx_vec_raw_d = thrust::raw_pointer_cast(dx_vec_thr_d.data());        
        thrust::complex<double> *dy_vec_raw_d = thrust::raw_pointer_cast(dy_vec_thr_d.data());        
        thrust::complex<double> *dx_vec_time_raw_d = thrust::raw_pointer_cast(dx_vec_time_thr_d.data());
        thrust::complex<double> *dy_vec_time_raw_d = thrust::raw_pointer_cast(dy_vec_time_thr_d.data());
        thrust::complex<double> *t_vec_raw_d = thrust::raw_pointer_cast(t_vec_thr.data());
        thrust::complex<double> *e_pol_x_raw_d = thrust::raw_pointer_cast(e_pol_x_thr_d.data());
        thrust::complex<double> *e_pol_y_raw_d = thrust::raw_pointer_cast(e_pol_y_thr_d.data());
        thrust::complex<double> *S_x_raw_d = thrust::raw_pointer_cast(S_x_thr_d.data());
        thrust::complex<double> *S_y_raw_d = thrust::raw_pointer_cast(S_y_thr_d.data());
        thrust::complex<double> *q_raw_d = thrust::raw_pointer_cast(q_thr_d.data());
        thrust::complex<double> *modifier_params_raw_d = thrust::raw_pointer_cast(modifier_params_thr.data());
        



        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *U_F_raw_d = thrust::raw_pointer_cast(U_F_thr_d.data());
        thrust::complex<double> *U_raw_d = thrust::raw_pointer_cast(U_thr_d.data());


        // //Loop

        //Iteration i==0
        
        thrust::device_vector<thrust::complex<double>> data_H_time_thr_d(data_H, data_H + len_data);
        thrust::complex<double> *data_H_time_raw_d = thrust::raw_pointer_cast(data_H_time_thr_d.data());

        thrust::complex<double> one(1,0);
        thrust::complex<double> dE=0.5*(bounds[1]-bounds[0]);

        if (modifier_id == 0){
            // modifier_params=[A0,w,Tp,eta]

            thrust::complex<double> A0=modifier_params_thr[0];
            thrust::complex<double> w=modifier_params_thr[1];
            thrust::complex<double> Tp=modifier_params_thr[2];
            thrust::complex<double> eta=modifier_params_thr[3];


            thrust::complex<double> t=t_vec_thr[0];
            
            thrust::complex<double> alpha( ((A0 * thrust::cos(w * t) / thrust::cosh((t - 2 * Tp) / (0.5673 * Tp)))/dE).real(),0.0);
            thrust::complex<double> beta(((eta * A0 * thrust::sin(w * t) / thrust::cosh((t - 2 * Tp) / (0.5673 * Tp)))/dE).real(),0.0);
            
            add_device<<<n_blocks, n_threads>>>(dx_vec_raw_d, data_H_time_raw_d, alpha, one,len_data);
            add_device<<<n_blocks, n_threads>>>(dy_vec_raw_d, data_H_time_raw_d, beta, one,len_data);


        
        }
        else if (modifier_id ==1){

            //modifier_params=thrust::complex<double> a0,thrust::complex<double> b,thrust::complex<double> Aq,const thrust::complex<double>* e_pol_x,const thrust::complex<double>* e_pol_y,thrust::complex<double> wq,double t,int* Sx,int* Sy,thrust::complex<double>* q
            hopping_pos_phonons_modifier(data_H_time_raw_d,indices_raw_d,dx_vec_raw_d,dy_vec_raw_d,modifier_params_thr[0],modifier_params_thr[1],modifier_params_thr[2],e_pol_x_raw_d,e_pol_y_raw_d,modifier_params_thr[3],t_vec_thr[0],S_x_raw_d,S_y_raw_d,q_raw_d,len_row,len_data);                        
        }

        rec_A(data_H_time_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_FD,U_F_raw_d,M);
        
        
        // // //Rest of the iterations
        for(int i = 1 ; i < len_t ; i++){
            thrust::copy(data_H_thr_d.begin(), data_H_thr_d.end(), data_H_time_thr_d.begin()); 


            if (modifier_id == 0){

                thrust::complex<double> A0=modifier_params_thr[0];
                thrust::complex<double> w=modifier_params_thr[1];
                thrust::complex<double> Tp=modifier_params_thr[2];
                thrust::complex<double> eta=modifier_params_thr[3];
                thrust::complex<double> t=t_vec_thr[i];
                
                thrust::complex<double> alpha( ((A0 * thrust::cos(w * t) / thrust::cosh((t - 2 * Tp) / (0.5673 * Tp)))/dE).real(),0.0);
                thrust::complex<double> beta(((eta * A0 * thrust::sin(w * t) / thrust::cosh((t - 2 * Tp) / (0.5673 * Tp)))/dE).real(),0.0);
            
                add_device<<<n_blocks, n_threads>>>(dx_vec_raw_d, data_H_time_raw_d, alpha, one,len_data);
                add_device<<<n_blocks, n_threads>>>(dy_vec_raw_d, data_H_time_raw_d, beta, one,len_data);
            
            }
            else if (modifier_id ==1){
                thrust::copy(dx_vec_thr_d.begin(), dx_vec_thr_d.end(), dx_vec_time_thr_d.begin()); 
                thrust::copy(dy_vec_thr_d.begin(), dy_vec_thr_d.end(), dy_vec_time_thr_d.begin()); 
                //modifier_params=thrust::complex<double> a0,thrust::complex<double> b,thrust::complex<double> Aq,const thrust::complex<double>* e_pol_x,const thrust::complex<double>* e_pol_y,thrust::complex<double> wq,double t,int* Sx,int* Sy,thrust::complex<double>* q
                hopping_pos_phonons_modifier(data_H_time_raw_d,indices_raw_d,dx_vec_time_raw_d,dy_vec_time_raw_d,modifier_params_thr[0],modifier_params_thr[1],modifier_params_thr[2],e_pol_x_raw_d,e_pol_y_raw_d,modifier_params_thr[3],t_vec_thr[i],S_x_raw_d,S_y_raw_d,q_raw_d,len_row,len_data);                        

            }
            
            rec_A(data_H_time_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_F_raw_d,M2);
            rec_A(data_H_time_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_raw_d,M2);
            

        }


        thrust::copy(U_thr_d.begin(),U_thr_d.end(), U); 
        thrust::copy(U_F_thr_d.begin(),U_F_thr_d.end(), U_F); 


    }
  void kpm_rho_neq_tau(std::complex <double>* data_H,std::complex <double>* dx_vec,std::complex <double>* dy_vec, int* indices, int len_row, int N , std::complex <double>* moments_kernel_U,std::complex <double>* moments_kernel_FD,std::complex <double>* t_vec,int len_t,std::complex <double>* U,std::complex <double>* U_F,std::complex<double>* modifier_params,int len_modifer,int M,int M2,double delta_t_tau){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;

        // Allocate device vectors of the operator and copy them in device 
        thrust::device_vector<thrust::complex<double>> data_H_thr_d(data_H, data_H + len_data);
        thrust::device_vector<thrust::complex<double>> dx_vec_thr_d(dx_vec, dx_vec + len_data);
        thrust::device_vector<thrust::complex<double>> dy_vec_thr_d(dy_vec, dy_vec + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);

        thrust::device_vector<thrust::complex<double>> U_thr_d(U,U+N); //initialize U with v
        thrust::device_vector<thrust::complex<double>> U_F_thr_d(U_F,U_F+N); // Empty U_F

        thrust::device_vector<thrust::complex<double>> moments_kernel_thr_U(moments_kernel_U, moments_kernel_U + M2);
        thrust::device_vector<thrust::complex<double>> moments_kernel_thr_FD(moments_kernel_FD, moments_kernel_FD + M);
        thrust::device_vector<thrust::complex<double>> t_vec_thr(t_vec,t_vec + len_t);
        thrust::device_vector<thrust::complex<double>> modifier_params_thr(modifier_params, modifier_params + len_modifer);


        // // Create raw pointers that can be used by the Kernel
        thrust::complex<double> *data_H_raw_d = thrust::raw_pointer_cast(data_H_thr_d.data());
        thrust::complex<double> *dx_vec_raw_d = thrust::raw_pointer_cast(dx_vec_thr_d.data());
        thrust::complex<double> *dy_vec_raw_d = thrust::raw_pointer_cast(dy_vec_thr_d.data());
        
        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *U_F_raw_d = thrust::raw_pointer_cast(U_F_thr_d.data());
        thrust::complex<double> *U_raw_d = thrust::raw_pointer_cast(U_thr_d.data());


        // //Loop

        //Iteration i==0
        // modifier_params=[A0,w,Tp,eta]
        
        thrust::device_vector<thrust::complex<double>> data_H_time_thr_d(data_H, data_H + len_data);
        thrust::complex<double> *data_H_time_raw_d = thrust::raw_pointer_cast(data_H_time_thr_d.data());

        thrust::device_vector<thrust::complex<double>> aux_thr_d(N);
        thrust::complex<double> *aux_raw_d = thrust::raw_pointer_cast(aux_thr_d.data());

        
        // // //Rest of the iterations
        for(int i = 0 ; i < len_t ; i++){   
            thrust::copy(data_H_thr_d.begin(), data_H_thr_d.end(), data_H_time_thr_d.begin()); 
            
            light_modifier_circular_packed_c( modifier_params[0],t_vec_thr[i],modifier_params[1],modifier_params[2],modifier_params[3],data_H_time_thr_d,dx_vec_thr_d,dy_vec_thr_d,len_data);
            
            rec_A(data_H_time_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_F_raw_d,M2);
            rec_A(data_H_time_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_U,U_raw_d,M2);
            
            thrust::copy(U_thr_d.begin(),U_thr_d.end(), aux_thr_d.begin()); 

            rec_A(data_H_time_raw_d,indices_raw_d,len_row,N ,moments_kernel_thr_FD,aux_raw_d,M);
            
            add_device<<<n_blocks, n_threads>>>(aux_raw_d, U_F_raw_d, thrust::complex<double>(delta_t_tau, 0.0), thrust::complex<double>(1.0-delta_t_tau, 0.0),N);

        }


        thrust::copy(U_thr_d.begin(),U_thr_d.end(), U); 
        thrust::copy(U_F_thr_d.begin(),U_F_thr_d.end(), U_F); 


    }



////////////////////////Computation of the position operator

    void position_operator(std::complex <double>* data_H,std::complex <double>* data_V, int* indices, int len_row, int N , std::complex <double>* moments_kernel_G_delta,std::complex <double>* pos,std::complex <double>* v,int M){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;

        // Allocate device vectors of the operator and copy them in device 
        thrust::device_vector<thrust::complex<double>> data_H_thr_d(data_H, data_H + len_data);
        thrust::device_vector<thrust::complex<double>> data_V_thr_d(data_V, data_V + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);

        thrust::device_vector<thrust::complex<double>> v_thr_d(v,v+N);
        thrust::device_vector<thrust::complex<double>> pos_thr_d(pos,pos+N);
        thrust::device_vector<thrust::complex<double>> moments_kernel_thr_G_delta(moments_kernel_G_delta, moments_kernel_G_delta + M*M);


        // Create raw pointers that can be used by the Kernel
        thrust::complex<double> *data_H_raw_d = thrust::raw_pointer_cast(data_H_thr_d.data());
        thrust::complex<double> *data_V_raw_d = thrust::raw_pointer_cast(data_V_thr_d.data());
        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *v_raw_d = thrust::raw_pointer_cast(v_thr_d.data());
        thrust::complex<double> *pos_raw_d = thrust::raw_pointer_cast(pos_thr_d.data());

        //Create auxilary vector
        thrust::device_vector<thrust::complex<double>> T_delta_0_thr_d(v, v+N);
        thrust::complex<double> *T_delta_0_raw = thrust::raw_pointer_cast(T_delta_0_thr_d.data());

        thrust::device_vector<thrust::complex<double>> T_delta_1_thr_d(N);
        thrust::device_vector<thrust::complex<double>> T_delta_v_thr_d(N);
        thrust::complex<double> *T_delta_1_raw = thrust::raw_pointer_cast(T_delta_1_thr_d.data());
        thrust::complex<double> *T_delta_v_raw = thrust::raw_pointer_cast(T_delta_v_thr_d.data());
        thrust::device_vector<thrust::complex<double>> moments_slice(M);

        // //Loop
        //M=0 iteration
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), T_delta_0_raw,T_delta_v_raw, data_V_raw_d, indices_raw_d, len_row, N);
        
        thrust::copy(moments_kernel_thr_G_delta.begin(),moments_kernel_thr_G_delta.begin() + M,moments_slice.begin());            
        
        rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_slice,T_delta_v_raw,M);
        


        add_device<<<n_blocks, n_threads>>>(T_delta_v_raw, pos_raw_d, thrust::complex<double>(1.0,0.0), thrust::complex<double>(1.0, 0.0),N);

        //M=1 interation
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), v_raw_d,T_delta_1_raw, data_H_raw_d, indices_raw_d, len_row, N);
        
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), T_delta_1_raw,T_delta_v_raw, data_V_raw_d, indices_raw_d, len_row, N);
        

        thrust::copy(moments_kernel_thr_G_delta.begin()+M,moments_kernel_thr_G_delta.begin() + M+M,moments_slice.begin());            

        rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_slice,T_delta_v_raw,M);
        
        add_device<<<n_blocks, n_threads>>>(T_delta_v_raw, pos_raw_d, thrust::complex<double>(1.0,0.0), thrust::complex<double>(1.0, 0.0),N);


        // rest of the loop

        for (int m = 2 ; m < M ; m++){

            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(2.0, 0.0),thrust::complex<double>(-1.0, 0.0), T_delta_1_raw, T_delta_0_raw, data_H_raw_d, indices_raw_d, len_row, N);

            // Swap the pointers
            thrust::swap(T_delta_0_thr_d,T_delta_1_thr_d);

            T_delta_0_raw  = thrust::raw_pointer_cast(T_delta_0_thr_d.data());
            T_delta_1_raw  = thrust::raw_pointer_cast(T_delta_1_thr_d.data());

            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), T_delta_1_raw,T_delta_v_raw, data_V_raw_d, indices_raw_d, len_row, N);
            
            thrust::copy(moments_kernel_thr_G_delta.begin()+M*m,moments_kernel_thr_G_delta.begin() + M*(m+1),moments_slice.begin());            
            
            rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_slice,T_delta_v_raw,M);

            // Add them
            add_device<<<n_blocks, n_threads>>>(T_delta_v_raw, pos_raw_d, thrust::complex<double>(1.0,0.0), thrust::complex<double>(1.0, 0.0),N);
        }

        thrust::copy(pos_thr_d.begin(),pos_thr_d.end(), pos); 


    }
    /// The same but with not copyng the vectors inside

    void position_operator_2(thrust::complex<double>* data_H_raw_d,thrust::complex<double>* data_V_raw_d, int* indices_raw_d, int len_row, int N , thrust::device_vector<thrust::complex<double>> moments_kernel_G_delta_thr_d,thrust::complex<double>* pos_raw_d,thrust::complex<double>* v_raw_d,int M){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);

        //Create auxilary vector
        thrust::device_vector<thrust::complex<double>> T_delta_0_thr_d(v_raw_d, v_raw_d+N);
        thrust::complex<double> *T_delta_0_raw = thrust::raw_pointer_cast(T_delta_0_thr_d.data());

        thrust::device_vector<thrust::complex<double>> T_delta_1_thr_d(N);
        thrust::device_vector<thrust::complex<double>> T_delta_v_thr_d(N);
        thrust::complex<double> *T_delta_1_raw = thrust::raw_pointer_cast(T_delta_1_thr_d.data());
        thrust::complex<double> *T_delta_v_raw = thrust::raw_pointer_cast(T_delta_v_thr_d.data());
        thrust::device_vector<thrust::complex<double>> moments_slice(M);

        // //Loop
        //M=0 iteration
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), T_delta_0_raw,T_delta_v_raw, data_V_raw_d, indices_raw_d, len_row, N);
        
        thrust::copy(moments_kernel_G_delta_thr_d.begin(),moments_kernel_G_delta_thr_d.begin() + M,moments_slice.begin());            
        
        rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_slice,T_delta_v_raw,M);
        


        add_device<<<n_blocks, n_threads>>>(T_delta_v_raw, pos_raw_d, thrust::complex<double>(1.0,0.0), thrust::complex<double>(1.0, 0.0),N);

        //M=1 interation
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), v_raw_d,T_delta_1_raw, data_H_raw_d, indices_raw_d, len_row, N);
        
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), T_delta_1_raw,T_delta_v_raw, data_V_raw_d, indices_raw_d, len_row, N);
        

        thrust::copy(moments_kernel_G_delta_thr_d.begin()+M,moments_kernel_G_delta_thr_d.begin() + 2*M,moments_slice.begin());            

        rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_slice,T_delta_v_raw,M);
        
        add_device<<<n_blocks, n_threads>>>(T_delta_v_raw, pos_raw_d, thrust::complex<double>(1.0,0.0), thrust::complex<double>(1.0, 0.0),N);


        // rest of the loop

        for (int m = 2 ; m < M ; m++){

            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(2.0, 0.0),thrust::complex<double>(-1.0, 0.0), T_delta_1_raw, T_delta_0_raw, data_H_raw_d, indices_raw_d, len_row, N);

            // Swap the pointers
            thrust::swap(T_delta_0_thr_d,T_delta_1_thr_d);

            T_delta_0_raw  = thrust::raw_pointer_cast(T_delta_0_thr_d.data());
            T_delta_1_raw  = thrust::raw_pointer_cast(T_delta_1_thr_d.data());

            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), T_delta_1_raw,T_delta_v_raw, data_V_raw_d, indices_raw_d, len_row, N);
            
            thrust::copy(moments_kernel_G_delta_thr_d.begin()+M*m,moments_kernel_G_delta_thr_d.begin() + M*(m+1),moments_slice.begin());            
            
            rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_slice,T_delta_v_raw,M);

            // Add them
            add_device<<<n_blocks, n_threads>>>(T_delta_v_raw, pos_raw_d, thrust::complex<double>(1.0,0.0), thrust::complex<double>(1.0, 0.0),N);
        }



    }
    void angular_momentum(std::complex <double>* data_H,std::complex <double>* data_V1,std::complex <double>* data_V2, int* indices, int len_row, int N , std::complex <double>* moments_kernel_Gmas_delta, std::complex <double>* moments_kernel_Gmin_delta,std::complex <double>* pos,std::complex <double>* v,int M){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;
        
        /// Copy to device complex Thrust vectors 

        thrust::device_vector<thrust::complex<double>> data_H_thr_d(data_H, data_H + len_data);
        thrust::device_vector<thrust::complex<double>> data_V1_thr_d(data_V1, data_V1 + len_data);
        thrust::device_vector<thrust::complex<double>> data_V2_thr_d(data_V2, data_V2 + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);

        thrust::device_vector<thrust::complex<double>> v_thr_d(v,v+N);
        thrust::device_vector<thrust::complex<double>> pos1_thr_d(N);
        thrust::device_vector<thrust::complex<double>> pos2_thr_d(N);
        thrust::device_vector<thrust::complex<double>> moments_kernel_Gmas_delta_thr_d(moments_kernel_Gmas_delta, moments_kernel_Gmas_delta + M*M);
        thrust::device_vector<thrust::complex<double>> moments_kernel_Gmin_delta_thr_d(moments_kernel_Gmin_delta, moments_kernel_Gmin_delta + M*M);

        /// Make the necessary raw pointers
        thrust::complex<double> *v_raw_d = thrust::raw_pointer_cast(v_thr_d.data());
        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *data_H_raw_d = thrust::raw_pointer_cast(data_H_thr_d.data());
        thrust::complex<double> *data_V1_raw_d = thrust::raw_pointer_cast(data_V1_thr_d.data());
        thrust::complex<double> *data_V2_raw_d = thrust::raw_pointer_cast(data_V2_thr_d.data());
        thrust::complex<double> *pos1_raw_d = thrust::raw_pointer_cast(pos1_thr_d.data());
        thrust::complex<double> *pos2_raw_d = thrust::raw_pointer_cast(pos2_thr_d.data());


        
        /// Auxiliary vectors

        thrust::device_vector<thrust::complex<double>> aux1_thr_d(N);
        thrust::complex<double> *aux1_raw_d = thrust::raw_pointer_cast(aux1_thr_d.data());
        thrust::device_vector<thrust::complex<double>> aux2_thr_d(N);
        thrust::complex<double> *aux2_raw_d = thrust::raw_pointer_cast(aux2_thr_d.data());




        /// First term
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), v_raw_d,aux1_raw_d, data_V2_raw_d, indices_raw_d, len_row, N);


        position_operator_2(data_H_raw_d,data_V1_raw_d,indices_raw_d,len_row,N ,moments_kernel_Gmas_delta_thr_d,pos1_raw_d,aux1_raw_d,M);



        /// Second term

        position_operator_2(data_H_raw_d,data_V2_raw_d,indices_raw_d,len_row,N ,moments_kernel_Gmin_delta_thr_d,aux2_raw_d,v_raw_d,M);

        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), aux2_raw_d,pos2_raw_d, data_V1_raw_d, indices_raw_d, len_row, N);

        
        /// Doing the substraction

        
        add_device<<<n_blocks, n_threads>>>(pos1_raw_d, pos2_raw_d, thrust::complex<double>(1.0,0.0), thrust::complex<double>(-1.0, 0.0),N);

        thrust::copy(pos2_thr_d.begin(),pos2_thr_d.end(), pos); 


        
    }
    /// The same but with not copyng the vectors inside
    void angular_momentum_2(thrust::complex <double>* data_H_raw_d,thrust::complex <double>* data_V1_raw_d,thrust::complex <double>* data_V2_raw_d, int* indices_raw_d, int len_row, int N , thrust::device_vector<thrust::complex<double>> moments_kernel_Gmas_delta_thr_d, thrust::device_vector<thrust::complex<double>> moments_kernel_Gmin_delta_thr_d,thrust::complex <double>* pos_raw_d,thrust::complex <double>* v_raw_d,int M){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);


        thrust::device_vector<thrust::complex<double>> pos1_thr_d(N);
        thrust::device_vector<thrust::complex<double>> pos2_thr_d(N);
        thrust::complex<double> *pos1_raw_d = thrust::raw_pointer_cast(pos1_thr_d.data());
        thrust::complex<double> *pos2_raw_d = thrust::raw_pointer_cast(pos2_thr_d.data());
        /// Auxiliary vectors

        thrust::device_vector<thrust::complex<double>> aux1_thr_d(N);
        thrust::complex<double> *aux1_raw_d = thrust::raw_pointer_cast(aux1_thr_d.data());
        thrust::device_vector<thrust::complex<double>> aux2_thr_d(N);
        thrust::complex<double> *aux2_raw_d = thrust::raw_pointer_cast(aux2_thr_d.data());




        /// First term
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), v_raw_d,aux1_raw_d, data_V2_raw_d, indices_raw_d, len_row, N);


        position_operator_2(data_H_raw_d,data_V1_raw_d,indices_raw_d,len_row,N ,moments_kernel_Gmas_delta_thr_d,pos1_raw_d,aux1_raw_d,M);



        /// Second term

        position_operator_2(data_H_raw_d,data_V2_raw_d,indices_raw_d,len_row,N ,moments_kernel_Gmin_delta_thr_d,aux2_raw_d,v_raw_d,M);

        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), aux2_raw_d,pos2_raw_d, data_V1_raw_d, indices_raw_d, len_row, N);

        
        /// Doing the substraction

        
        add_device<<<n_blocks, n_threads>>>(pos1_raw_d, pos2_raw_d, thrust::complex<double>(1.0,0.0), thrust::complex<double>(-1.0, 0.0),N);

        thrust::copy(pos2_thr_d.begin(),pos2_thr_d.end(), pos_raw_d); 
        
    }
    void sigma_time_orbital(std::complex <double>* data_H,std::complex <double>* data_V1,std::complex <double>* data_V2,std::complex <double>* data_V2_kpm, int* indices, int len_row, int N , std::complex <double>* moments_kernel_Gmas_delta, std::complex <double>* moments_kernel_Gmin_delta,std::complex <double>* sigma_vec,std::complex <double>* moments_kernel_FD,std::complex <double>* mu_vec,int len_mu,std::complex <double>* moments_U,int M2,int len_t,std::complex <double>* v,int M){
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;
        
        /// Copy to device complex Thrust vectors 

        thrust::device_vector<thrust::complex<double>> data_H_thr_d(data_H, data_H + len_data);
        thrust::device_vector<thrust::complex<double>> data_V1_thr_d(data_V1, data_V1 + len_data);
        thrust::device_vector<thrust::complex<double>> data_V2_thr_d(data_V2, data_V2 + len_data);
        thrust::device_vector<thrust::complex<double>> data_V2_kpm_thr_d(data_V2_kpm, data_V2_kpm + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);

        thrust::device_vector<thrust::complex<double>> v_thr_d(v,v+N);
        thrust::device_vector<thrust::complex<double>> L1_thr_d(N);
        thrust::device_vector<thrust::complex<double>> L2_thr_d(N);
        thrust::device_vector<thrust::complex<double>> moments_kernel_Gmas_delta_thr_d(moments_kernel_Gmas_delta, moments_kernel_Gmas_delta + M*M);
        thrust::device_vector<thrust::complex<double>> moments_kernel_Gmin_delta_thr_d(moments_kernel_Gmin_delta, moments_kernel_Gmin_delta + M*M);
        thrust::device_vector<thrust::complex<double>> moments_kernel_FD_thr_d(moments_kernel_FD, moments_kernel_FD + M*len_mu);
        thrust::device_vector<thrust::complex<double>> moments_U_thr_d(moments_U, moments_U + M2);
        thrust::device_vector<thrust::complex<double>> sigma_vec_thr_d(len_t*len_mu);

        /// Make the necessary raw pointers
        thrust::complex<double> *v_raw_d = thrust::raw_pointer_cast(v_thr_d.data());
        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *data_H_raw_d = thrust::raw_pointer_cast(data_H_thr_d.data());
        thrust::complex<double> *data_V1_raw_d = thrust::raw_pointer_cast(data_V1_thr_d.data());
        thrust::complex<double> *data_V2_raw_d = thrust::raw_pointer_cast(data_V2_thr_d.data());
        thrust::complex<double> *data_V2_kpm_raw_d = thrust::raw_pointer_cast(data_V2_kpm_thr_d.data());
        thrust::complex<double> *L1_raw_d = thrust::raw_pointer_cast(L1_thr_d.data());
        thrust::complex<double> *L2_raw_d = thrust::raw_pointer_cast(L2_thr_d.data());


        /// Auxiliary vectors 

        thrust::device_vector<thrust::complex<double>> aux1_thr_d(N);
        thrust::complex<double> *aux1_raw_d = thrust::raw_pointer_cast(aux1_thr_d.data());
        thrust::device_vector<thrust::complex<double>> aux2_thr_d(N);
        thrust::complex<double> *aux2_raw_d = thrust::raw_pointer_cast(aux2_thr_d.data());

        /////////Doing the anticomutator

        /// Term 1 
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), v_raw_d,aux1_raw_d, data_V1_raw_d, indices_raw_d, len_row, N);
        angular_momentum_2(data_H_raw_d,data_V1_raw_d,data_V2_raw_d,indices_raw_d,len_row,N ,moments_kernel_Gmas_delta_thr_d,moments_kernel_Gmin_delta_thr_d,L1_raw_d,aux1_raw_d,M);
        /// Term 2
        angular_momentum_2(data_H_raw_d,data_V1_raw_d,data_V2_raw_d,indices_raw_d,len_row,N ,moments_kernel_Gmas_delta_thr_d,moments_kernel_Gmin_delta_thr_d,aux2_raw_d,v_raw_d,M);
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), aux2_raw_d,L2_raw_d, data_V1_raw_d, indices_raw_d, len_row, N);

        /// Summation

        add_device<<<n_blocks, n_threads>>>(L1_raw_d, L2_raw_d, thrust::complex<double>(1.0,0.0), thrust::complex<double>(1.0, 0.0),N);

 
        /////////Doing the time evolutionç

        /// Auxiliary vectors
        thrust::device_vector<thrust::complex<double>> Udag_thr_d(v,v+N);
        thrust::complex<double> *Udag_raw_d = thrust::raw_pointer_cast(Udag_thr_d.data());
        thrust::device_vector<thrust::complex<double>> Udag_L_thr_d=L2_thr_d;
        thrust::complex<double> *Udag_L_raw_d = thrust::raw_pointer_cast(Udag_L_thr_d.data());
        thrust::device_vector<thrust::complex<double>> aux_Udag_L_thr_d(N);
        thrust::complex<double> *aux_Udag_L_raw_d = thrust::raw_pointer_cast(aux_Udag_L_thr_d.data());
        thrust::device_vector<thrust::complex<double>> moments_slice_FD(M);
        
        ////Loop
        //Iteration i==0
        int k=0;
        for(int i =0;i<len_mu;i++){
            thrust::copy(Udag_L_thr_d.begin(), Udag_L_thr_d.end(), aux_Udag_L_thr_d.begin()); 
            thrust::copy(moments_kernel_FD_thr_d.begin() + i*M,moments_kernel_FD_thr_d.begin() + (i+1)*M,moments_slice_FD.begin());            
            

            rec_comm_A_vec(data_H_raw_d,data_V2_kpm_raw_d,indices_raw_d,len_row,N ,moments_slice_FD,aux_Udag_L_raw_d,M);


            sigma_vec_thr_d[k] = thrust::inner_product(Udag_thr_d.begin(), Udag_thr_d.end(), aux_Udag_L_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                        thrust::plus<thrust::complex<double>>(),  vdot());   
            k+=1;
        } 


        //Rest of the iterations
        for(int j = 1 ; j < len_t ; j++){
            rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_U_thr_d,Udag_raw_d,M2);
            rec_A(data_H_raw_d,indices_raw_d,len_row,N ,moments_U_thr_d,Udag_L_raw_d,M2);



            for(int i =0;i<len_mu;i++){
                thrust::copy(Udag_L_thr_d.begin(), Udag_L_thr_d.end(), aux_Udag_L_thr_d.begin()); 
                thrust::copy(moments_kernel_FD_thr_d.begin() + i*M,moments_kernel_FD_thr_d.begin() + (i+1)*M,moments_slice_FD.begin());            


                rec_comm_A_vec(data_H_raw_d,data_V2_kpm_raw_d,indices_raw_d,len_row,N ,moments_slice_FD,aux_Udag_L_raw_d,M);

                sigma_vec_thr_d[k] = thrust::inner_product(Udag_thr_d.begin(), Udag_thr_d.end(), aux_Udag_L_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                            thrust::plus<thrust::complex<double>>(),  vdot());   
                k+=1;
                
                }      
        }


        thrust::copy(sigma_vec_thr_d.begin(), sigma_vec_thr_d.end(), sigma_vec); 



        
    }


}




