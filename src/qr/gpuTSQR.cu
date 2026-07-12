#include "common/gpuLinalg.cuh"
#include "common/gpuUtils.cuh"
#include "qr/gpuTSQR.cuh"

namespace qr::gpu {

// --- KERNELS ---

// 1. Initialize buffers and pad extra rows with zeros to cleanly fit the power-of-2 block structure
template <typename T>
__global__ void initBuffersKernelPadded(const T* A, T* R_tmp, T* V, uint32_t M, uint32_t M_padded,
                                        uint32_t N) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M_padded * N) {
    uint32_t r = idx % M_padded;
    uint32_t c = idx / M_padded;
    V[idx] = 0;
    if (r < M) {
      R_tmp[idx] = A[r + c * M];
    } else {
      R_tmp[idx] = 0; // Pad trailing rows with 0
    }
  }
}

// 2. Batched Householder Vector Generation
template <typename T>
__global__ void generateHouseholderKernelBatched(T* R_tmp, T* V, T* tau, uint32_t M_stride,
                                                 uint32_t H, uint32_t N, uint32_t k) {
  uint32_t batch_idx = blockIdx.y;
  uint32_t tid = threadIdx.x;

  extern __shared__ char shm_raw[];
  T* sdata = reinterpret_cast<T*>(shm_raw);

  uint32_t row_offset = batch_idx * H;

  T local_sq = 0;
  for (uint32_t i = k + tid; i < H; i += blockDim.x) {
    T val = R_tmp[(row_offset + i) + k * M_stride];
    local_sq += val * val;
  }
  sdata[tid] = local_sq;
  __syncthreads();

  // Shared memory parallel reduction
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
      T x0 = R_tmp[(row_offset + k) + k * M_stride];
      T g = (x0 >= 0) ? norm_x : -norm_x;
      shared_g = g;
      shared_vk = x0 + g;
    }
  }
  __syncthreads();

  if (shared_g == 0) {
    if (tid == 0)
      tau[batch_idx * N + k] = 0;
    return;
  }

  local_sq = 0;
  for (uint32_t i = k + tid; i < H; i += blockDim.x) {
    T v_val = (i == k) ? shared_vk : R_tmp[(row_offset + i) + k * M_stride];
    V[(row_offset + i) + k * M_stride] = v_val;
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
    tau[batch_idx * N + k] = (sdata[0] == 0) ? static_cast<T>(0) : static_cast<T>(2) / sdata[0];
  }
}

// 3. Batched Householder Reflection Application
template <typename T>
__global__ void applyHouseholderRKernelBatched(T* R_tmp, T* V, const T* tau, uint32_t M_stride,
                                               uint32_t H, uint32_t N, uint32_t k) {
  uint32_t j = k + blockIdx.x;
  if (j >= N)
    return;

  uint32_t batch_idx = blockIdx.y;
  T t_val = tau[batch_idx * N + k];
  if (t_val == 0)
    return;

  uint32_t tid = threadIdx.x;
  extern __shared__ char shm_raw[];
  T* sdata = reinterpret_cast<T*>(shm_raw);

  uint32_t row_offset = batch_idx * H;

  T local_dot = 0;
  for (uint32_t i = k + tid; i < H; i += blockDim.x) {
    local_dot += V[(row_offset + i) + k * M_stride] * R_tmp[(row_offset + i) + j * M_stride];
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

  for (uint32_t i = k + tid; i < H; i += blockDim.x) {
    R_tmp[(row_offset + i) + j * M_stride] -=
        t_val * V[(row_offset + i) + k * M_stride] * shared_dot;
  }
}

// 4. Compact intermediate R factors to maintain contiguous layout across tree levels
template <typename T>
__global__ void compactRKernel(const T* src, T* dst, uint32_t M_stride, uint32_t H, uint32_t N,
                               uint32_t P) {
  uint32_t b = blockIdx.y; // Partition index
  uint32_t j = blockIdx.x; // Column index

  uint32_t dst_M_stride = P * N;

  for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
    uint32_t src_row = b * H + i;
    uint32_t dst_row = b * N + i;
    dst[dst_row + j * dst_M_stride] = src[src_row + j * M_stride];
  }
}

// 5. Extract final upper triangular R matrix
template <typename T> __global__ void extractFinalRKernel(const T* src, T* R, uint32_t N) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * N) {
    uint32_t i = idx % N;
    uint32_t j = idx / N;
    R[i + j * N] = (i <= j) ? src[i + j * N] : static_cast<T>(0);
  }
}

// 6. Enforce positive diagonal elements on final R
template <typename T> __global__ void correctSignsRKernel(T* R, uint32_t N) {
  uint32_t k = blockIdx.x;
  if (k >= N)
    return;

  __shared__ bool do_flip;
  if (threadIdx.x == 0) {
    do_flip = (R[k + k * N] < 0);
  }
  __syncthreads();

  if (!do_flip)
    return;

  for (uint32_t j = k + threadIdx.x; j < N; j += blockDim.x) {
    R[k + j * N] = -R[k + j * N];
  }
}

// --- HOST ALGORITHM ORCHESTRATION ---

template <typename T> void TSQR(const T* d_A, T* d_R, uint32_t M, uint32_t N) {
  uint32_t threadsPerBlock = 256;

  // 1. Calculate an ideal power-of-2 grid partitioning scheme
  uint32_t P = 1;
  while (M / P > 256) {
    P *= 2;
  }
  uint32_t H_leaf = (M + P - 1) / P;
  uint32_t M_padded = P * H_leaf;

  const uint64_t leafMN = static_cast<uint64_t>(M_padded) * N;
  T *d_R_tmp = nullptr, *d_V_leaf = nullptr, *d_tau_leaf = nullptr;
  if (cudaMalloc(&d_R_tmp, leafMN * sizeof(T)) != cudaSuccess ||
      cudaMalloc(&d_V_leaf, leafMN * sizeof(T)) != cudaSuccess ||
      cudaMalloc(&d_tau_leaf, static_cast<uint64_t>(P) * N * sizeof(T)) != cudaSuccess) {
    cudaGetLastError();
    cudaFree(d_R_tmp);
    cudaFree(d_V_leaf);
    cudaFree(d_tau_leaf);
    return;
  }

  // Initialize & Pad
  uint32_t blocksInit = static_cast<uint32_t>((leafMN + threadsPerBlock - 1) / threadsPerBlock);
  initBuffersKernelPadded<<<blocksInit, threadsPerBlock>>>(d_A, d_R_tmp, d_V_leaf, M, M_padded, N);

  // --- PHASE 1: LEAF LEVEL BATCHED QR ---
  {
    for (uint32_t k = 0; k < N; ++k) {
      dim3 gridGen(1, P);
      generateHouseholderKernelBatched<<<gridGen, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(
          d_R_tmp, d_V_leaf, d_tau_leaf, M_padded, H_leaf, N, k);

      uint32_t remainingCols = N - k;
      dim3 gridApply(remainingCols, P);
      applyHouseholderRKernelBatched<<<gridApply, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(
          d_R_tmp, d_V_leaf, d_tau_leaf, M_padded, H_leaf, N, k);
    }
  }

  const uint64_t treeSz = static_cast<uint64_t>(P) * N * N;
  T *d_R_tree1 = nullptr, *d_R_tree2 = nullptr, *d_V_tree = nullptr, *d_tau_tree = nullptr;
  if (cudaMalloc(&d_R_tree1, treeSz * sizeof(T)) != cudaSuccess ||
      cudaMalloc(&d_R_tree2, treeSz * sizeof(T)) != cudaSuccess ||
      cudaMalloc(&d_V_tree, treeSz * sizeof(T)) != cudaSuccess ||
      cudaMalloc(&d_tau_tree, static_cast<uint64_t>(P) * N * sizeof(T)) != cudaSuccess) {
    cudaGetLastError();
    cudaFree(d_R_tmp);
    cudaFree(d_V_leaf);
    cudaFree(d_tau_leaf);
    cudaFree(d_R_tree1);
    cudaFree(d_R_tree2);
    cudaFree(d_V_tree);
    cudaFree(d_tau_tree);
    return;
  }

  // Compact the initial R outputs from the leaf layer into tree buffer 1
  dim3 gridCompact(N, P);
  compactRKernel<<<gridCompact, threadsPerBlock>>>(d_R_tmp, d_R_tree1, M_padded, H_leaf, N, P);

  uint32_t current_P = P;
  T* d_R_tree_src = d_R_tree1;
  T* d_R_tree_dst = d_R_tree2;

  while (current_P > 1) {
    uint32_t next_P = current_P / 2;
    uint32_t next_H = 2 * N;
    uint32_t M_stride = current_P * N;

    // Perform Batched QR reduction on current tree layer pairs
    for (uint32_t k = 0; k < N; ++k) {
      dim3 gridGen(1, next_P);
      generateHouseholderKernelBatched<<<gridGen, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(
          d_R_tree_src, d_V_tree, d_tau_tree, M_stride, next_H, N, k);

      uint32_t remainingCols = N - k;
      dim3 gridApply(remainingCols, next_P);
      applyHouseholderRKernelBatched<<<gridApply, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(
          d_R_tree_src, d_V_tree, d_tau_tree, M_stride, next_H, N, k);
    }

    // Compact this layer's results down for the next tree step
    dim3 gridTreeCompact(N, next_P);
    compactRKernel<<<gridTreeCompact, threadsPerBlock>>>(d_R_tree_src, d_R_tree_dst, M_stride,
                                                         next_H, N, next_P);

    // Ping-pong pointers
    std::swap(d_R_tree_src, d_R_tree_dst);
    current_P = next_P;
  }

  // --- PHASE 3: EXTRACT & POST-PROCESS ---
  // Extract upper triangular matrix from the root of the tree
  uint32_t blocksExtract = (N * N + threadsPerBlock - 1) / threadsPerBlock;
  extractFinalRKernel<<<blocksExtract, threadsPerBlock>>>(d_R_tree_src, d_R, N);

  // Force diagonal signs to be positive
  correctSignsRKernel<<<N, threadsPerBlock>>>(d_R, N);

  // Free all allocated device memory
  cudaFree(d_R_tmp);
  cudaFree(d_V_leaf);
  cudaFree(d_tau_leaf);
  cudaFree(d_R_tree1);
  cudaFree(d_R_tree2);
  cudaFree(d_V_tree);
  cudaFree(d_tau_tree);
}

// Explicit template instantiations
template void TSQR<float>(const float*, float*, uint32_t, uint32_t);
template void TSQR<double>(const double*, double*, uint32_t, uint32_t);

} // namespace qr::gpu
