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
#include <thrust/complex.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
extern "C"{
///////////////////////////////////////////////7


    ///////////////////////////

    // modifier of the circular polarized light packed
    void light_modifier_circular_packed_c(thrust::complex<double> A0,
                                        thrust::complex<double> t,
                                        thrust::complex<double> w,
                                        thrust::complex<double> Tp,
                                        thrust::complex<double> eta,
                                        thrust::device_vector<thrust::complex<double>>& data,
                                        thrust::device_vector<thrust::complex<double>>& dx_vec,
                                        thrust::device_vector<thrust::complex<double>>& dy_vec,
                                        int N)
    {
        if (eta==0){

            thrust::complex<double> alpha(0.0, 0.0);
            thrust::complex<double> beta(0.0, (A0 * thrust::sin(w * t) / thrust::cosh((t - 2 * Tp) / (0.5673 * Tp))).real());
            
            // Create the exponential functor
            exponential exp_functor(alpha, beta);
            
            thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(data.begin(), dx_vec.begin(), dy_vec.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(data.end(), dx_vec.end(), dy_vec.end())),
                data.begin(),
                exp_functor
            );

        }
        else {

            thrust::complex<double> alpha(0.0, (A0 * thrust::cos(w * t) / thrust::cosh((t - 2 * Tp) / (0.5673 * Tp))).real());
            thrust::complex<double> beta(0.0, (eta * A0 * thrust::sin(w * t) / thrust::cosh((t - 2 * Tp) / (0.5673 * Tp))).real());
            
            // Create the exponential functor
            exponential exp_functor(alpha, beta);
            
            thrust::transform(
                thrust::make_zip_iterator(thrust::make_tuple(data.begin(), dx_vec.begin(), dy_vec.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(data.end(), dx_vec.end(), dy_vec.end())),
                data.begin(),
                exp_functor
            );

        }

    }

    /// Modifier of the phonons

    // void light_modifier_phonons_packed_c(thrust::complex<double> a0,
    //                                     thrust::complex<double> w,
    //                                     thrust::complex<double> Tp,
    //                                     thrust::complex<double> b,
    //                                     thrust::complex<double> Aq,
    //                                     thrust::complex<double> t,
    //                                     thrust::device_vector<thrust::complex<double>>& data,
    //                                     thrust::device_vector<thrust::complex<double>>& dx_vec,
    //                                     thrust::device_vector<thrust::complex<double>>& dy_vec,
    //                                     thrust::device_vector<thrust::complex<double>>& e_pol_x,
    //                                     thrust::device_vector<thrust::complex<double>>& e_pol_y,
    //                                     thrust::device_vector<thrust::complex<double>>& S_x,
    //                                     thrust::device_vector<thrust::complex<double>>& S_y,
    //                                     thrust::device_vector<thrust::complex<double>>& q,
    //                                     int N)
    // {
    //     thrust::complex<double> alpha(0.0, (A0 * thrust::cos(w * t) / thrust::cosh((t - 2 * Tp) / (0.5673 * Tp))).real());
    //     thrust::complex<double> beta(0.0, (eta * A0 * thrust::sin(w * t) / thrust::cosh((t - 2 * Tp) / (0.5673 * Tp))).real());
        
    //     // Create the exponential functor
    //     exponential exp_functor(alpha, beta);
        
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(data.begin(), dx_vec.begin(), dy_vec.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(data.end(), dx_vec.end(), dy_vec.end())),
    //         data.begin(),
    //         exp_functor
    //     );
    // }


    __global__ void hopping_pos_phonons_modifier_kernel(
        thrust::complex<double>* data,
        const int* indices,
        thrust::complex<double>* dx_vec,
        thrust::complex<double>* dy_vec,
        thrust::complex<double> a0,
        thrust::complex<double> b,
        thrust::complex<double> Aq,
        const thrust::complex<double>* e_pol_x,
        const thrust::complex<double>* e_pol_y,
        thrust::complex<double> wq,
        thrust::complex<double> t,
        const thrust::complex<double>* Sx,
        const thrust::complex<double>* Sy,
        const thrust::complex<double>* q,
        int len_row,
        int size)
    {

        int idx_cu = threadIdx.x + blockIdx.x * blockDim.x;
        int n_threads = blockDim.x*gridDim.x;
        int i;
        for( i = idx_cu ; i < size ; i+=n_threads){
            if (i/len_row!=indices[i]){
                thrust::complex<double> Tp = 2.0 * M_PI / wq;

                // update dx and dy in-place
                dx_vec[i] = dx_vec[i] + Aq *(e_pol_x[i / len_row] * thrust::cos(wq * t - q[0] * Sx[i / len_row])- e_pol_x[indices[i]] * thrust::cos(wq * t - q[0] * Sx[indices[i]]))/ thrust::cosh((t - 2.0 * Tp) / (0.5673 * Tp));

                dy_vec[i] = dy_vec[i] + Aq *(e_pol_y[i / len_row] * thrust::cos(wq * t - q[1] * Sy[i / len_row])- e_pol_y[indices[i]] * thrust::cos(wq * t - q[1] * Sy[indices[i]]))/ thrust::cosh((t - 2.0 * Tp) / (0.5673 * Tp));

                // update data in-place
                double r = sqrt(thrust::norm(dx_vec[i]) + thrust::norm(dy_vec[i])) / a0.real() - 1.0;
                data[i] *=  thrust::exp(-b * r);
            }


        }


    }

    // Host wrapper
    void hopping_pos_phonons_modifier(
        thrust::complex<double>* data,
        const int* indices,
        thrust::complex<double>* dx_vec,
        thrust::complex<double>* dy_vec,
        thrust::complex<double> a0,
        thrust::complex<double> b,
        thrust::complex<double> Aq,
        const thrust::complex<double>* e_pol_x,
        const thrust::complex<double>* e_pol_y,
        thrust::complex<double> wq,
        thrust::complex<double> t,
        const thrust::complex<double>* Sx,
        const thrust::complex<double>* Sy,
        const thrust::complex<double>* q,
        int len_row,
        int size)
    {
        int threadsPerBlock = 1024;
        int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        hopping_pos_phonons_modifier_kernel<<<blocks, threadsPerBlock>>>(
            data, indices, dx_vec, dy_vec,
            a0, b, Aq, e_pol_x, e_pol_y, wq, t, Sx, Sy, q, len_row, size
        );
        cudaDeviceSynchronize();
    }


}




