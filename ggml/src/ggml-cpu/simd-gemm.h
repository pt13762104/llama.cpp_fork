#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#include "simd-mappings.h"

// TODO: add support for sizeless vector types
#if defined(GGML_SIMD) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#if defined(__AVX512F__) || defined (__ARM_NEON) || defined(__ARM_NEON__)
    static constexpr int GEMM_RM = 4;
    static constexpr int GEMM_RN = 4; // 16+4+1 = 25/32
#elif defined(__AVX2__) || defined(__AVX__)
    static constexpr int GEMM_RM = 6;
    static constexpr int GEMM_RN = 2; // 12+2+1 = 15/16
#else
    static constexpr int GEMM_RM = 2;
    static constexpr int GEMM_RN = 2;
#endif

#if defined(__ARM_FEATURE_SVE)

#define load_i_r(i, r) if constexpr ((i < RM) && (r < RN)) \
                       acc##i##r = GGML_F32_VEC_LOAD(C + i * N + r * KN)
#define load_acc(i) load_i_r(i, 0); load_i_r(i, 1); load_i_r(i, 2); load_i_r(i, 3);
#define store_i_r(i, r) if constexpr ((i < RM) && (r < RN)) \
                        GGML_F32_VEC_STORE(C + i * N + r * KN, acc##i##r)
#define store_acc(i) store_i_r(i, 0); store_i_r(i, 1); store_i_r(i, 2); store_i_r(i, 3);
#define load_b(r) if constexpr (r < RN) \
                  Bv##r = GGML_F32_VEC_LOAD(B + kk * N + r * KN)
#define FMA(i, r) if constexpr ((i < RM) && (r < RN)) \
                  acc##i##r = svmla_n_f32_m(svptrue_b32(), acc##i##r, Bv##r, A[i * K + kk])
#define accum(i) FMA(i, 0); FMA(i, 1); FMA(i, 2); FMA(i, 3)

template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, int KN)
{

    GGML_F32_VEC acc00,acc01,acc02,acc03;
    GGML_F32_VEC acc10,acc11,acc12,acc13;
    GGML_F32_VEC acc20,acc21,acc22,acc23;
    GGML_F32_VEC acc30,acc31,acc32,acc33;

    load_acc(0); load_acc(1); load_acc(2); load_acc(3);

    for (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv0, Bv1, Bv2, Bv3;
        load_b(0); load_b(1); load_b(2); load_b(3);
        accum(0); accum(1); accum(2); accum(3);
    }

    store_acc(0); store_acc(1); store_acc(2); store_acc(3);

}

#else

template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, int KN)
{

    GGML_F32_VEC acc[RM][RN];
    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    for (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        for (int r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        for (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            for (int r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}
#endif
// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
#if defined(__ARM_FEATURE_SVE)
    static int KN = svcntw();
#else
    static constexpr int KN = GGML_F32_EPR;
#endif

    int64_t ii = 0;
    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N, KN);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N, KN);
        }
        for (; jj < N; jj++) {
            for (int64_t i = 0; i < GEMM_RM; i++) {
                float a = C[i * N + jj];
                for (int64_t kk = 0; kk < K; kk++) {
                    a += A[i + kk] * B[kk * N + jj];
                }
                C[i * N + jj] = a;
            }
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    for (; ii < M; ii++) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N, KN);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N, KN);
        }
        for (; jj < N; jj++) {
            float a = C[jj];
            for (int64_t kk = 0; kk < K; kk++) {
                a += A[kk] * B[kk * N + jj];
            }
            C[jj] = a;
        }

        A += K;
        C += N;
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#else // scalar path

static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = C[i * N + j];
            for (int64_t kk = 0; kk < K; kk++) {
                sum += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#endif // GGML_SIMD
