#include "common/cpuUtils.hpp"
#include "common/gpuUtils.cuh"
#include "qr/gpuBlockedHouseholder.cuh"
#include <gtest/gtest.h>

// ── Test 1: singleReflector ──────────────────────────────────────────────────
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

  // Correct expected value: tau = (beta - v[0]) / beta
  // norm(v[1:]) = √39 ≈ 6.245
  // beta = -√(16 + 39) = -√55 ≈ -7.416
  // tau = (-7.416 - 4) / -7.416 ≈ 1.539
  EXPECT_NEAR(tau, 1.5393598900f, 1e-5f);

  CUDACHECK(cudaFree(d_tau));
  CUDACHECK(cudaFree(d_v));
}

// ── Test 2: applyReflectorKernel ─────────────────────────────────────────────
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

  // Column 0 after reflector:
  // w_c = 1*1 + 2*2 + 3*3 + 4*4 = 30
  // C[:,0] = [1,2,3,4] - (-0.5)*[1,2,3,4]*30 = [1,2,3,4] + [15,30,45,60] = [16,32,48,64]
  EXPECT_NEAR(C[0], 16.0f, 1e-5f);
  EXPECT_NEAR(C[1], 32.0f, 1e-5f);
  EXPECT_NEAR(C[2], 48.0f, 1e-5f);
  EXPECT_NEAR(C[3], 64.0f, 1e-5f);

  // Column 1 after reflector:
  // w_c = 1*5 + 2*6 + 3*7 + 4*8 = 70
  // C[:,1] = [5,6,7,8] - (-0.5)*[1,2,3,4]*70 = [5,6,7,8] + [35,70,105,140] = [40,76,112,148]
  EXPECT_NEAR(C[4], 40.0f, 1e-5f);
  EXPECT_NEAR(C[5], 76.0f, 1e-5f);
  EXPECT_NEAR(C[6], 112.0f, 1e-5f);
  EXPECT_NEAR(C[7], 148.0f, 1e-5f);

  CUDACHECK(cudaFree(d_C));
  CUDACHECK(cudaFree(d_v));
  CUDACHECK(cudaFree(d_tau));
}

// ── Test 3: unblockedPanel (full QR validation) ──────────────────────────────
// Computes reflectors, builds Q, extracts R from the packed V, and verifies both
// that Q is orthogonal and that Q*R reconstructs the original A.
TEST(detailBlockedHouseholder, unblockedPanelFullTest) {
  const int M = 7;
  const int N = 4;

  // Full-rank deterministic matrix (column-major). Avoids degenerate columns so
  // every reflector is non-trivial.
  float A_host[M * N];
  for (int i = 0; i < M * N; ++i) {
    A_host[i] = static_cast<float>((i * 37 + 11) % 100) * 0.1f;
  }

  float *d_V, *d_tau, *d_Q;
  CUDACHECK(cudaMalloc(&d_V, M * N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_tau, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_Q, M * M * sizeof(float)));

  CUDACHECK(cudaMemcpy(d_V, A_host, M * N * sizeof(float), cudaMemcpyHostToDevice));

  // Step 1: factor the panel in place. After this, V holds R in its upper triangle
  // (diagonal = beta) and the essential reflector vectors below the diagonal.
  qr::gpu::_detail::unblockedPanel(d_V, d_tau, M, N);
  CUDACHECK(cudaDeviceSynchronize());

  // Step 2: build Q = H_0 · H_1 · ... · H_{N-1} from identity.
  // applyReflectorKernel left-multiplies (C := H·C), so to land H_0 on the far left
  // we apply in REVERSE order. Reflector j acts on rows j..M-1, hence the +j offset.
  float I[M * M] = {0};
  for (int i = 0; i < M; ++i)
    I[i * M + i] = 1.0f;
  CUDACHECK(cudaMemcpy(d_Q, I, M * M * sizeof(float), cudaMemcpyHostToDevice));

  for (int j = N - 1; j >= 0; --j) {
    qr::gpu::_detail::applyReflectorKernel<<<M, M>>>(d_Q + j, d_V + j * M + j, d_tau + j,
                                                     M /*stride*/, M /*numcols*/, M - j /*M_eff*/);
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
