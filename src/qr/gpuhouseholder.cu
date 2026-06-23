#include "qr/gpuhouseholder.cuh"
#include <cmath>

namespace qr::gpu {
template <typename T>
__global__ void initBuffersKernel(const T* A, T* R_tmp, T* V, T* Q, uint32_t M, uint32_t N) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M * N) {
    R_tmp[idx] = A[idx];
    V[idx] = 0;
    uint32_t r = idx % M;
    uint32_t c = idx / M;
    Q[idx] = (r == c) ? static_cast<T>(1) : static_cast<T>(0);
  }
}
template <typename T>
__global__ void generateHousholderKernel(T* R_tmp, T* V, T* tau, uint32_t M, uint32_t k) {
  uint32_t tid = threadIdx.x;
  extern __shared__ char shm_raw[];
  T* sdata = reinterpret_cast<T*>(shm_raw);

  T local_sq = 0;
  for (uint32_t i = k + tid; i < M; i += blockDim.x) {
    T val = R_tmp[i + k * M];
    local_sq += val * val;
  }
  sdata[tid] = local_sq;
  __syncthreads();

  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  __shared__ T shared_g;
  __shared__ T shared_vk;

  if (tid == 0) {
    T norm_x = sqrt(sdata[0]);
    if (norm_x == 0) {
      shared_g = 0;
      shared_vk = 0;
    } else {
      T x0 = R_tmp[k + k * M];
      T g = (x0 >= 0) ? norm_x : -norm_x;
      shared_g = g;
      shared_vk = x0 + g;
    }
  }
  __syncthreads();

  if (shared_g == 0) {
    if (tid == 0)
      tau[k] = 0;
    return;
  }

  // Populate V and compute the norm squared of the final Householder vector v
  local_sq = 0;
  for (uint32_t i = k + tid; i < M; i += blockDim.x) {
    T v_val = (i == k) ? shared_vk : R_tmp[i + k * M];
    V[i + k * M] = v_val;
    local_sq += v_val * v_val;
  }
  sdata[tid] = local_sq;
  __syncthreads();

  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    tau[k] = (sdata[0] == 0) ? static_cast<T>(0) : static_cast<T>(2) / sdata[0];
  }
}

template <typename T>
__global__ void applyHousholderRKernel(T* R_tmp, T* V, const T* tau, uint32_t M, uint32_t N,
                                       uint32_t k) {
  uint32_t j = k + blockIdx.x;

  if (j >= N)
    return;

  T t_val = tau[k];

  if (t_val == 0)
    return;

  uint32_t tid = threadIdx.x;
  extern __shared__ char shm_raw[];
  T* sdata = reinterpret_cast<T*>(shm_raw);

  T local_dot = 0;
  for (uint32_t i = k + tid; i < M; i += blockDim.x) {
    local_dot += V[i + k * M] * R_tmp[i + j * M];
  }
  sdata[tid] = local_dot;
  __syncthreads();

  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  __shared__ T shared_dot;
  if (tid == 0)
    shared_dot = sdata[0];
  __syncthreads();

  for (uint32_t i = k + tid; i < M; i += blockDim.x) {
    R_tmp[i + j * M] -= t_val * V[i + k * M] * shared_dot;
  }
}

template <typename T>
__global__ void applyHousholderQKernel(T* Q, const T* V, const T* tau, uint32_t M, uint32_t N,
                                       uint32_t k) {
  uint32_t j = blockIdx.x;
  if (j >= N)
    return;

  T t_val = tau[k];
  if (t_val == 0)
    return;

  uint32_t tid = threadIdx.x;
  extern __shared__ char shm_raw[];
  T* sdata = reinterpret_cast<T*>(shm_raw);

  T local_dot = 0;
  for (uint32_t i = k + tid; i < M; i += blockDim.x) {
    local_dot += V[i + k * M] * Q[i + j * M];
  }
  sdata[tid] = local_dot;
  __syncthreads();

  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  __shared__ T shared_dot;
  if (tid == 0)
    shared_dot = sdata[0];
  __syncthreads();

  for (uint32_t i = k + tid; i < M; i += blockDim.x) {
    Q[i + j * M] -= t_val * V[i + k * M] * shared_dot;
  }
}
template <typename T> __global__ void extractRKernel(const T* R_tmp, T* R, uint32_t M, uint32_t N) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * N) {
    uint32_t i = idx % N;
    uint32_t j = idx / N;
    R[i + j * N] = (i <= j) ? R_tmp[i + j * M] : static_cast<T>(0);
  }
}

template <typename T> __global__ void correctSignsKernel(T* Q, T* R, uint32_t M, uint32_t N) {
  uint32_t k = blockIdx.x; // One block handles one diagonal element/row/column
  if (k >= N)
    return;

  __shared__ bool do_flip;
  if (threadIdx.x == 0) {
    do_flip = (R[k + k * N] < 0);
  }
  __syncthreads();

  if (!do_flip)
    return;

  // Flip the k-th row of R
  for (uint32_t j = k + threadIdx.x; j < N; j += blockDim.x) {
    R[k + j * N] = -R[k + j * N];
  }

  // Flip the k-th column of Q
  for (uint32_t i = threadIdx.x; i < M; i += blockDim.x) {
    Q[i + k * M] = -Q[i + k * M];
  }
}

} // namespace qr::gpu

namespace qr::gpu {
template <typename T>
void Householder(const T* A, T* Q, T* R, uint32_t M, uint32_t N, T* workspace) {
  T* d_R_tmp;
  T* d_V;
  T* d_tau;

  cudaMalloc(&d_R_tmp, M * N * sizeof(T));
  cudaMalloc(&d_V, M * N * sizeof(T));
  cudaMalloc(&d_tau, N * sizeof(T));

  uint32_t threadsPerBlock = 256;
  uint32_t blocksMN = (M * N + threadsPerBlock - 1) / threadsPerBlock;

  initBuffersKernel<<<blocksMN, threadsPerBlock>>>(A, d_R_tmp, d_V, Q, M, N);

  for (uint32_t k = 0; k < N; ++k) {
    generateHousholderKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(d_R_tmp, d_V,
                                                                                  d_tau, M, k);
    uint32_t remainingCols = N - k;
    applyHousholderRKernel<<<remainingCols, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(
        d_R_tmp, d_V, d_tau, M, N, k);
  }

  uint32_t blocksNN = (N * N + threadsPerBlock - 1) / threadsPerBlock;
  extractRKernel<<<blocksNN, threadsPerBlock>>>(d_R_tmp, R, M, N);

  for (int k = static_cast<int>(N) - 1; k >= 0; --k) {
    applyHousholderQKernel<<<N, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(Q, d_V, d_tau, M, N,
                                                                                k);
  }
  correctSignsKernel<<<N, threadsPerBlock>>>(Q, R, M, N);

  cudaFree(d_R_tmp);
  cudaFree(d_V);
  cudaFree(d_tau);
}
template void Householder<float>(const float*, float*, float*, uint32_t, uint32_t, float*);
template void Householder<double>(const double*, double*, double*, uint32_t, uint32_t, double*);
} // namespace qr::gpu
