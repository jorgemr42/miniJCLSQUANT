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


    void rec_A_tab(std::complex <double>* data, int* indices, int len_row, int N , std::complex <double>* A_vec,std::complex <double>* v,int M){
        
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;

        // Allocate device vectors of the operator and copy them in device 
        thrust::device_vector<thrust::complex<double>> data_thr_d(data, data + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);

        thrust::device_vector<thrust::complex<double>> v_thr_d(v, v + N);
        thrust::device_vector<thrust::complex<double>> A_vec_thr_d(A_vec, A_vec + M);


        // Create raw pointers that can be used by the Kernel
        thrust::complex<double> *data_raw_d = thrust::raw_pointer_cast(data_thr_d.data());
        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *v_raw_d = thrust::raw_pointer_cast(v_thr_d.data());

        ////Start T_0 and T_1
        // T_0 this can be optimized

        thrust::device_vector<thrust::complex<double>> T_0(N);
        thrust::copy(v_raw_d, v_raw_d+N, T_0.begin());

        thrust::complex<double> *T_0_raw = thrust::raw_pointer_cast(T_0.data());

        // T_1

        thrust::device_vector<thrust::complex<double>> T_1(N,0);
        thrust::complex<double> *T_1_raw = thrust::raw_pointer_cast(T_1.data());


        thrust::complex<double> a(1.0, 0.0);
        thrust::complex<double> b(0.0, 0.0);



        aAxby_device<<<n_blocks, n_threads>>>(a,b, v_raw_d, T_1_raw, data_raw_d, indices_raw_d, len_row, N);



        // Build the device vector that will have the result

        thrust::complex<double> N_c(N, 0.0);


        A_vec_thr_d[0] = thrust::inner_product(v_thr_d.begin(), v_thr_d.end(), v_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());

        A_vec_thr_d[1] = thrust::inner_product(v_thr_d.begin(), v_thr_d.end(), T_1.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());



        // Set things for the recursion 
        a=thrust::complex<double>(2.0, 0.0);
        b=thrust::complex<double>(-1.0, 0.0);
        
        // Recursion 

        for (int m = 2 ; m < M ; m++){

            aAxby_device<<<n_blocks, n_threads>>>(a,b, T_1_raw, T_0_raw, data_raw_d, indices_raw_d, len_row, N);

            // Swap the pointers
            thrust::swap(T_0,T_1);

            T_0_raw = thrust::raw_pointer_cast(T_0.data());
            T_1_raw = thrust::raw_pointer_cast(T_1.data());

            // Make the vdot
            A_vec_thr_d[m] = thrust::inner_product(v_thr_d.begin(), v_thr_d.end(), T_1.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());

        }

        //Copy back from the raw Device pointers to original pointers
        thrust::copy(A_vec_thr_d.begin(), A_vec_thr_d.end(), A_vec); 


    }




    void rec_A_tab_b(thrust::complex <double>* data_raw_d, int* indices_raw_d, int len_row, int N , thrust::complex<double>* A_vec_raw,thrust::device_vector<thrust::complex<double>> v_thr_d,int M){
        
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);

        thrust::complex<double> *v_raw_d = thrust::raw_pointer_cast(v_thr_d.data());

        thrust::device_vector<thrust::complex<double>> A_vec_thr_d(A_vec_raw,A_vec_raw+M);


        ////Start T_0 and T_1
        // T_0 this can be optimized

        thrust::device_vector<thrust::complex<double>> T_0(N);
        thrust::copy(v_raw_d, v_raw_d+N, T_0.begin());

        thrust::complex<double> *T_0_raw = thrust::raw_pointer_cast(T_0.data());

        // T_1

        thrust::device_vector<thrust::complex<double>> T_1(N,0);
        thrust::complex<double> *T_1_raw = thrust::raw_pointer_cast(T_1.data());


        thrust::complex<double> a(1.0, 0.0);
        thrust::complex<double> b(0.0, 0.0);



        aAxby_device<<<n_blocks, n_threads>>>(a,b, v_raw_d, T_1_raw, data_raw_d, indices_raw_d, len_row, N);



        // Build the device vector that will have the result

        thrust::complex<double> N_c(N, 0.0);


        A_vec_thr_d[0] = thrust::inner_product(v_thr_d.begin(), v_thr_d.end(), v_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());



        A_vec_thr_d[1] = thrust::inner_product(v_thr_d.begin(), v_thr_d.end(), T_1.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());


        // Set things for the recursion 
        a=thrust::complex<double>(2.0, 0.0);
        b=thrust::complex<double>(-1.0, 0.0);
        
        // Recursion 

        for (int m = 2 ; m < M ; m++){

            aAxby_device<<<n_blocks, n_threads>>>(a,b, T_1_raw, T_0_raw, data_raw_d, indices_raw_d, len_row, N);

            // Swap the pointers
            thrust::swap(T_0,T_1);

            T_0_raw = thrust::raw_pointer_cast(T_0.data());
            T_1_raw = thrust::raw_pointer_cast(T_1.data());

            // Make the vdot
            A_vec_thr_d[m] = thrust::inner_product(v_thr_d.begin(), v_thr_d.end(), T_1.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());

        }
        thrust::copy(A_vec_thr_d.begin(), A_vec_thr_d.end(), A_vec_raw); 


    }





    void rec_A_tab_2(std::complex <double>* data, int* indices, int len_row, int N , std::complex <double>* A_vec,std::complex <double>* v_l,std::complex <double>* v_r,int M){
        
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);
        int len_data = len_row*N;

        // Allocate device vectors of the operator and copy them in device 
        thrust::device_vector<thrust::complex<double>> data_thr_d(data, data + len_data);
        thrust::device_vector<int> indices_thr_d(indices, indices + len_data);

        thrust::device_vector<thrust::complex<double>> v_r_thr_d(v_r, v_r + N);
        thrust::device_vector<thrust::complex<double>> v_l_thr_d(v_l, v_l + N);
        thrust::device_vector<thrust::complex<double>> A_vec_thr_d(A_vec, A_vec + M);


        // Create raw pointers that can be used by the Kernel
        thrust::complex<double> *data_raw_d = thrust::raw_pointer_cast(data_thr_d.data());
        int *indices_raw_d = thrust::raw_pointer_cast(indices_thr_d.data());
        thrust::complex<double> *v_r_raw_d = thrust::raw_pointer_cast(v_r_thr_d.data());
        thrust::complex<double> *v_l_raw_d = thrust::raw_pointer_cast(v_l_thr_d.data());

        ////Start T_0 and T_1
        // T_0 this can be optimized

        thrust::device_vector<thrust::complex<double>> T_0(N);
        thrust::copy(v_r_thr_d.begin(), v_r_thr_d.end(), T_0.begin());

        thrust::complex<double> *T_0_raw = thrust::raw_pointer_cast(T_0.data());

        // T_1

        thrust::device_vector<thrust::complex<double>> T_1(N,0);
        thrust::complex<double> *T_1_raw = thrust::raw_pointer_cast(T_1.data());


        thrust::complex<double> a(1.0, 0.0);
        thrust::complex<double> b(0.0, 0.0);



        aAxby_device<<<n_blocks, n_threads>>>(a,b, v_r_raw_d, T_1_raw, data_raw_d, indices_raw_d, len_row, N);



        // Build the device vector that will have the result

        thrust::complex<double> N_c(N, 0.0);


        A_vec_thr_d[0]=thrust::inner_product(v_l_thr_d.begin(), v_l_thr_d.end(), v_r_thr_d.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());
       

        A_vec_thr_d[1] = thrust::inner_product(v_l_thr_d.begin(), v_l_thr_d.end(), T_1.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());


        // Set things for the recursion 
        a=thrust::complex<double>(2.0, 0.0);
        b=thrust::complex<double>(-1.0, 0.0);
        
        // Recursion 

        for (int m = 2 ; m < M ; m++){

            aAxby_device<<<n_blocks, n_threads>>>(a,b, T_1_raw, T_0_raw, data_raw_d, indices_raw_d, len_row, N);

            // Swap the pointers
            thrust::swap(T_0,T_1);

            T_0_raw = thrust::raw_pointer_cast(T_0.data());
            T_1_raw = thrust::raw_pointer_cast(T_1.data());

            // Make the vdot
            A_vec_thr_d[m] = thrust::inner_product(v_l_thr_d.begin(), v_l_thr_d.end(), T_1.begin(), thrust::complex<double>(0.0,0.0),
                                    thrust::plus<thrust::complex<double>>(),  vdot());

        }

        //Copy back from the raw Device pointers to original pointers
        thrust::copy(A_vec_thr_d.begin(), A_vec_thr_d.end(), A_vec); 


    }



    void rec_A(thrust::complex<double>* data_raw_d, int* indices_raw_d, int len_row, int N ,thrust::device_vector<thrust::complex<double>> moments_kernel_thr,thrust::complex<double>* v_kpm_raw,int M){
        
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);

        ////Start T_0 and T_1

        // T_0 this can be optimized

        thrust::device_vector<thrust::complex<double>> T_0(v_kpm_raw, v_kpm_raw+N);

        thrust::complex<double> *T_0_raw = thrust::raw_pointer_cast(T_0.data());

        // T_1

        thrust::device_vector<thrust::complex<double>> T_1(N,0);
        thrust::complex<double> *T_1_raw = thrust::raw_pointer_cast(T_1.data());




        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), v_kpm_raw, T_1_raw, data_raw_d, indices_raw_d, len_row, N);


        // Build the device vector that will have the result
        //Make A_kpm[0]=moments_kerne[0]*T_0
        add_device<<<n_blocks, n_threads>>>(T_0_raw, v_kpm_raw, moments_kernel_thr[0], thrust::complex<double>(0.0, 0.0),N);
        //Make A_kpm[1]=moments_kerne_[1]*T_1
        add_device<<<n_blocks, n_threads>>>(T_1_raw, v_kpm_raw, moments_kernel_thr[1], thrust::complex<double>(1.0, 0.0),N);

        // Set things for the recursion 

        
        // Recursion 

        for (int m = 2 ; m < M ; m++){

            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(2.0, 0.0),thrust::complex<double>(-1.0, 0.0), T_1_raw, T_0_raw, data_raw_d, indices_raw_d, len_row, N);

            // Swap the pointers
            thrust::swap(T_0,T_1);

            T_0_raw = thrust::raw_pointer_cast(T_0.data());
            T_1_raw = thrust::raw_pointer_cast(T_1.data());

            // Make the vdot
            add_device<<<n_blocks, n_threads>>>(T_1_raw, v_kpm_raw, moments_kernel_thr[m], thrust::complex<double>(1.0, 0.0),N);
        }


    }
    
    void rec_comm_A_vec(thrust::complex<double>* data_raw_d,thrust::complex<double>* data2_raw_d, int* indices_raw_d, int len_row, int N ,thrust::device_vector<thrust::complex<double>> moments_kernel_thr,thrust::complex<double>* v_kpm_raw,int M){
        
        int n_threads = 1024;
        int n_blocks = max(1, (N + n_threads - 1) / n_threads);


        ////Start T_0 and T_1

        // T_0 this can be optimized

        thrust::device_vector<thrust::complex<double>> T_0(v_kpm_raw, v_kpm_raw+N);

        thrust::complex<double> *T_0_raw = thrust::raw_pointer_cast(T_0.data());

        // T_1

        thrust::device_vector<thrust::complex<double>> T_1(N,0);
        thrust::complex<double> *T_1_raw = thrust::raw_pointer_cast(T_1.data());

        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), v_kpm_raw, T_1_raw, data_raw_d, indices_raw_d, len_row, N);


        /////// Set the commutator
      
        // First commutator is zero
        thrust::device_vector<thrust::complex<double>> C_0(N,0);
        thrust::complex<double> *C_0_raw = thrust::raw_pointer_cast(C_0.data());

        // Second commutator commutator is zero
        thrust::device_vector<thrust::complex<double>> C_1(N,0);
        thrust::complex<double> *C_1_raw = thrust::raw_pointer_cast(C_1.data());
        
        aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(1.0, 0.0),thrust::complex<double>(0.0, 0.0), v_kpm_raw, C_1_raw, data2_raw_d, indices_raw_d, len_row, N);



        // Build the device vector that will have the result

        add_device<<<n_blocks, n_threads>>>(C_1_raw, v_kpm_raw, moments_kernel_thr[1], thrust::complex<double>(0.0, 0.0),N);

        
        // Recursion 

        for (int m = 2 ; m < M ; m++){

            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(2.0, 0.0),thrust::complex<double>(-1.0, 0.0), C_1_raw, C_0_raw, data_raw_d, indices_raw_d, len_row, N);
            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(2.0, 0.0),thrust::complex<double>(1.0, 0.0), T_1_raw, C_0_raw, data2_raw_d, indices_raw_d, len_row, N);

            // Swap the pointers
            thrust::swap(C_0,C_1);

            C_0_raw = thrust::raw_pointer_cast(C_0.data());
            C_1_raw = thrust::raw_pointer_cast(C_1.data());
            
            // recursion of the Tm
            aAxby_device<<<n_blocks, n_threads>>>(thrust::complex<double>(2.0, 0.0),thrust::complex<double>(-1.0, 0.0), T_1_raw, T_0_raw, data_raw_d, indices_raw_d, len_row, N);

            thrust::swap(T_0,T_1);

            T_0_raw = thrust::raw_pointer_cast(T_0.data());
            T_1_raw = thrust::raw_pointer_cast(T_1.data());



            // Make the vdot
            add_device<<<n_blocks, n_threads>>>(C_1_raw, v_kpm_raw, moments_kernel_thr[m], thrust::complex<double>(1.0, 0.0),N);
        }


    }

}




