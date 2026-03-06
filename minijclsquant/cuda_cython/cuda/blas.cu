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

    /////////////////////////////////////////////////////////////////////////////
    // Kernel function that y=a*x+b*y
    __global__ void add_device(thrust::complex<double> *x, thrust::complex<double> *y, thrust::complex<double> a, thrust::complex<double> b,int N) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int n_threads = blockDim.x*gridDim.x;
        for(int i = idx ; i < N ; i+=n_threads){
            y[i] = a*x[i] + b*y[i];
        }

    }
    /////////////////////////////////////////////////////////////////////////////

    // Kernel function that makes y=alpha*A*x+beta*y
    __global__ void aAxby_device( thrust::complex<double> a, thrust::complex<double> b,thrust::complex<double> *x,thrust::complex<double> *y, thrust::complex<double> *data, int *indices,int len_row,int N) {
        int idx_cu = threadIdx.x + blockIdx.x * blockDim.x;
        int n_threads = blockDim.x*gridDim.x;
        int idx,i,j;
        for( i = idx_cu ; i < N ; i+=n_threads){
            y[i]*=b;
            for(j=0 ; j < len_row ; j++ ){
                idx=i*len_row+j;    
                y[i]+=a*data[idx]*x[indices[idx]];

            }

        }

    }

    /////////////////////////////////////////////////////////////////////////////

    // Kernel function that makes the vdot c=<x|a*y>


    // struct vdot {
    //     __host__ __device__
    //     thrust::complex<double> operator()(const thrust::complex<double>& x, const thrust::complex<double>& y) const { 
    //         return thrust::conj(x) * y; 
    //     }
    // };

    // /////////////////////////////////////////////////////////////////////////////
    // // Apply transformation: y = y * exp(alpha * a + beta * b)

    // struct exponential {
    //     const thrust::complex<double> alpha;
    //     const thrust::complex<double> beta;

    //     // Constructor
    //     exponential(thrust::complex<double> _alpha, thrust::complex<double> _beta) 
    //         : alpha(_alpha), beta(_beta) {}

    //     __host__ __device__
    //     thrust::complex<double> operator()(const thrust::tuple<thrust::complex<double>, thrust::complex<double>, thrust::complex<double>>& t) const {
    //         thrust::complex<double> y = thrust::get<0>(t);
    //         thrust::complex<double> a = thrust::get<1>(t);
    //         thrust::complex<double> b = thrust::get<2>(t);
    //         return y * thrust::exp(alpha * a + beta * b);
    //     }
    // };
}




