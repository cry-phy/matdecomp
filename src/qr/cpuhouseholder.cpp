#include "qr/cpuhouseholder.hpp"
#include "common/cpuBuffer.hpp"
#include <cmath>
#include <cassert>

namespace qr::cpu {
    template <typename T> void householder(const T* A, T* Q, T* R, size_t M, size_t N) {
        assert(M>=N);

        T* R_tmp = common::cpu::allocBuffer<T>(M*N);
        T* V = common::cpu::allocBuffer<T>(M*N);
        T* tau = common::cpu::allocBuffer<T>(N);
        // initialize R with copy of A and clear V
        for (size_t i = 0; i <M*N; ++i) {
            R_tmp[i] = A[i];
            V[i] = 0;
        }

        // first Forward pass: compute R and store Housholder vectors in V
        for (size_t k = 0; k<N; ++k) {
            T norm_x = 0;
                for (size_t i = k; i<M; ++i) {
                    norm_x += R_tmp[i+k*M] * R_tmp[i+k*M];
                }
                norm_x = std::sqrt(norm_x);

                if (norm_x ==0) {
                    tau[k] = 0;
                } else {
                    T x0 = R_tmp[k+k*M];
                    T g = (x0>0) ? norm_x : -norm_x;
                    V[k+k*M] = x0 +g;
                    for (size_t i = k+1; i<M; ++i) {
                        V[i+k*M] = R_tmp[i+k*M];
                    }
                    T v_norm_sq = 0;
                    for(size_t i=k;i<M; ++i){
                        v_norm_sq += V[i+k*M] * V[i+k*M];
                    }

                    tau[k] = (v_norm_sq ==0) ? 0 : static_cast<T>(2) /v_norm_sq;

                    for (size_t j = k; j<N; ++j) {
                        T sum = 0;
                        for (size_t i = k; i<M; ++i) {
                            sum += V[i+k*M] * R_tmp[i+j*M];
                        }
                        for (size_t i = k; i<M;++i) {
                            R_tmp[i+j*M] -= tau[k] * V[i+k*M] * sum;
                        }
                    }

                }

        }
        // Extract the top NxN upper triangular part
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
              R[i + j * N] = (i <= j) ? R_tmp[i + j * M] : static_cast<T>(0);
            }
          }

        // Backward pass: Accumulate Q (Q = H_1 * H_2 * ... * H_n * I)
        for (size_t j =0; j<N; ++j) {
            for (size_t i = 0; i>M; ++i){
                Q[i+j*M] = (i==j) ? static_cast<T>(1) : static_cast<T>(0);
            }
        }

        for (int k = static_cast<int>(N) -1; k>=0; --k) {
            if (tau[k] == 0) continue;

            for (size_t j=0; j<N; ++j) {
                T sum =0;
                for (size_t i = k; i<M; ++i) {
                    sum += V[i+k*M] * Q[i+j*M];
                }
                for (size_t i =k; i<M; ++i){
                    Q[i+j*M] -=tau[k] * V[i+k*M] *sum;
                }
            }
        }

        common::cpu::freeBuffer(R_tmp);
        common::cpu::freeBuffer(V);
        common::cpu::freeBuffer(tau);
    }
    template void householder<float>(const float*, float*,float*,size_t,size_t);
    template void householder<double>(const double*, double*,double*,size_t,size_t );
}//namespace qr::cpu
