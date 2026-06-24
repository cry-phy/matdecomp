#include "common/cpuUtils.hpp"
#include "common/gpuUtils.cuh"
#include "qr/gpuBlockedHouseholder.cuh"
#include <gtest/gtest.h>
#include <iomanip>

TEST(detailBlockedHouseholder, singleReflectorTest) {
  const int M = 5;
  float *d_tau, *d_v;
  float v[M] = {4.0f, 1.0f, 2.0f, 3.0f, 5.0f};

  CUDACHECK(cudaMalloc(&d_tau, sizeof(float)));
  CUDACHECK(cudaMalloc(&d_v, M * sizeof(float)));
  CUDACHECK(cudaMemcpy(d_v, v, M * sizeof(float), cudaMemcpyHostToDevice));

  qr::gpu::_detail::singleReflector(d_v, d_tau, M);

  float tau;
  CUDACHECK(cudaMemcpy(&tau, d_tau, sizeof(float), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(tau, 1.5393598900f, 1e-5f);

  CUDACHECK(cudaFree(d_tau));
  CUDACHECK(cudaFree(d_v));
}

TEST(detailBlockedHouseholder, applyReflectorKernelTest) {
  const int M = 4;
  const int N = 3;
  float *d_C, *d_v, *d_tau;
  // Column-major: 12 elements = 4 rows × 3 columns
  float C[M * N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  float v[M] = {1.0f, 2.0f, 3.0f, 4.0f};
  float tau = -0.5f;

  CUDACHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_v, M * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_tau, sizeof(float)));
  CUDACHECK(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_v, v, M * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_tau, &tau, sizeof(float), cudaMemcpyHostToDevice));

  qr::gpu::_detail::applyReflectorKernel<<<N, M>>>(d_C, d_v, d_tau, M, N, M);
  cudaDeviceSynchronize();

  CUDACHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  EXPECT_NEAR(C[0], 16.0f, 1e-5f);
  EXPECT_NEAR(C[1], 32.0f, 1e-5f);
  EXPECT_NEAR(C[2], 48.0f, 1e-5f);
  EXPECT_NEAR(C[3], 64.0f, 1e-5f);

  EXPECT_NEAR(C[4], 40.0f, 1e-5f);
  EXPECT_NEAR(C[5], 76.0f, 1e-5f);
  EXPECT_NEAR(C[6], 112.0f, 1e-5f);
  EXPECT_NEAR(C[7], 148.0f, 1e-5f);

  CUDACHECK(cudaFree(d_C));
  CUDACHECK(cudaFree(d_v));
  CUDACHECK(cudaFree(d_tau));
}

TEST(detailBlockedHouseholder, unblockedPanelFullTest) {
  const int M = 7;
  const int N = 4;

  float A_host[M * N];
  for (int i = 0; i < M * N; ++i) {
    A_host[i] = static_cast<float>((i * 37 + 11) % 100) * 0.1f;
  }

  float *d_V, *d_tau, *d_Q;
  CUDACHECK(cudaMalloc(&d_V, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_tau, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_Q, M * M * sizeof(float)));

  CUDACHECK(cudaMemcpy(d_V, A_host, M * N * sizeof(float), cudaMemcpyHostToDevice));

  qr::gpu::_detail::unblockedPanel(d_V, d_tau, M /*stride*/, N, M /*M_eff*/);
  CUDACHECK(cudaDeviceSynchronize());

  float I[M * M] = {0};
  for (int i = 0; i < M; ++i)
    I[i * M + i] = 1.0f;
  CUDACHECK(cudaMemcpy(d_Q, I, M * M * sizeof(float), cudaMemcpyHostToDevice));

  for (int j = N - 1; j >= 0; --j) {
    qr::gpu::_detail::applyReflectorKernel<<<M, M>>>(d_Q + j, d_V + j * M + j, d_tau + j, M, M,
                                                     M - j);
    CUDACHECK(cudaDeviceSynchronize());
  }

  // Step 3: pull Q and the packed V back to host.
  float Q_host[M * M];
  float V_host[M * N];
  CUDACHECK(cudaMemcpy(Q_host, d_Q, M * M * sizeof(float), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(V_host, d_V, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Extract R (N×N) from the upper triangle of V, diagonal included.
  float R_host[N * N] = {0};
  for (int j = 0; j < N; ++j)
    for (int i = 0; i <= j; ++i)
      R_host[i + j * N] = V_host[i + j * M];

  // Verify Q is orthogonal: Q^T Q == I.
  double ortho_err = common::cpu::orthogonalityError(Q_host, M, M);
  EXPECT_LT(ortho_err, 1e-4) << "Q is not orthogonal, error = " << ortho_err;

  // Verify Q[:,0:N] * R reconstructs A.
  float QR_host[M * N];
  common::cpu::matMul(Q_host, M, N, R_host, N, N, QR_host);
  for (int i = 0; i < M * N; ++i) {
    EXPECT_NEAR(QR_host[i], A_host[i], 1e-4f)
        << "QR != A at index " << i << ": QR=" << QR_host[i] << " A=" << A_host[i];
  }

  CUDACHECK(cudaFree(d_V));
  CUDACHECK(cudaFree(d_tau));
  CUDACHECK(cudaFree(d_Q));
}

TEST(detailBlockedHouseholder, buildTMatrixTest) {
  const int M = 4, NB = 3;
  float V[M * NB] = {7.0f, 2.0f, -1.0f, 1.0f,  // col 0: diag, then v0 below
                     4.0f, 9.0f, 3.0f,  -2.0f, // col 1: R, diag, then v1 below
                     5.0f, 6.0f, 11.0f, 2.0f}; // col 2: R, R, diag, then v2 below
  float tau[NB] = {0.5f, 0.25f, 0.5f};

  float *d_V, *d_tau, *d_T;
  CUDACHECK(cudaMalloc(&d_V, M * NB * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_tau, NB * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_T, NB * NB * sizeof(float)));
  CUDACHECK(cudaMemcpy(d_V, V, M * NB * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_tau, tau, NB * sizeof(float), cudaMemcpyHostToDevice));

  qr::gpu::_detail::buildTMatrix(d_V, d_tau, d_T, M, NB, /*ldv=*/M);

  CUDACHECK(cudaDeviceSynchronize());

  float T_host[NB * NB];
  CUDACHECK(cudaMemcpy(T_host, d_T, NB * NB * sizeof(float), cudaMemcpyDeviceToHost));

  const float expected[NB * NB] = {0.5f, 0.0f, 0.0f, 0.375f, 0.25f, 0.0f, -0.0625f, 0.125f, 0.5f};
  for (int j = 0; j < NB; ++j)
    for (int i = 0; i <= j; ++i)
      EXPECT_NEAR(T_host[i + j * NB], expected[i + j * NB], 1e-5f)
          << "T(" << i << "," << j << ") mismatch";

  CUDACHECK(cudaFree(d_V));
  CUDACHECK(cudaFree(d_tau));
  CUDACHECK(cudaFree(d_T));
}

TEST(detailBlockedHouseholder, MatrixOrchestrationTest) {
  const uint32_t M = 48;
  const uint32_t N = 48;

  float* d_A;
  CUDACHECK(cudaMalloc(&d_A, M * N * sizeof(float)));

  float* h_A = new float[M * N];
  common::cpu::fillRandom(h_A, M * N);
  CUDACHECK(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  qr::gpu::BlockedHouseholderWorkspace<float> d_workspace1;
  qr::gpu::mallocWorkspace(d_workspace1, M, N, 16);

  qr::gpu::_detail::blockedOrchestrator(d_A, d_workspace1.res, M, N, std::move(d_workspace1));

  qr::gpu::BlockedHouseholderWorkspace<float> d_workspace2;
  qr::gpu::mallocWorkspace(d_workspace2, M, N, N);

  qr::gpu::_detail::blockedOrchestrator(d_A, d_workspace2.res, M, N, std::move(d_workspace2));

  // DEBUG print result intermediates:
  float* h_res1 = new float[M * N];
  float* h_res2 = new float[M * N];
  CUDACHECK(cudaMemcpy(h_res1, d_workspace1.res, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_res2, d_workspace2.res, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << "Result from panelWidth=16:\n";
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      std::cout << h_res1[i + j * M] << std::setprecision(2) << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\nResult from panelWidth=N:\n";
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      std::cout << h_res2[i + j * M] << std::setprecision(2) << " ";
    }
    std::cout << "\n";
  }

  // for (uint32_t i = 0; i < M * N; ++i) {
  //   EXPECT_NEAR(h_res1[i], h_res2[i], 1e-4f)
  //       << "Mismatch at index " << i << ": res1=" << h_res1[i] << ", res2=" << h_res2[i];
  // }

  qr::gpu::freeWorkspace(d_workspace1);
  qr::gpu::freeWorkspace(d_workspace2);
}
