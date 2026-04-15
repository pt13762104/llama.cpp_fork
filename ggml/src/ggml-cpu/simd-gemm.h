#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#include "simd-mappings.h"

// TODO: add support for sizeless vector types
#if defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#if defined(__AVX512F__) || defined (__ARM_NEON)
    static constexpr int GEMM_RM = 4;
    static constexpr int GEMM_RN = 4; // 16+4+1 = 25/32
#elif defined(__AVX2__) || defined(__AVX__)
    static constexpr int GEMM_RM = 6;
    static constexpr int GEMM_RN = 2; // 12+2+1 = 15/16
#else
    static constexpr int GEMM_RM = 2;
    static constexpr int GEMM_RN = 2;
#endif

template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

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

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

    int64_t ii = 0;
    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
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
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
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

#elif defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)

static constexpr int GEMM_RM = 4;
static constexpr int GEMM_RN = 4; // 16+4+1 = 24/32

#define define_ACC(x) GGML_F32_VEC acc##x##0,acc##x##1,acc##x##2,acc##x##3;
#define load_i_r(i, r) if constexpr ((i < RM) && (r < RN)) \
                       acc##i##r = GGML_F32_VEC_LOAD(C + i * N + r * KN)
#define load4(i) load_i_r(i, 0); \
                 load_i_r(i, 1); \
                 load_i_r(i, 2); \
                 load_i_r(i, 3);
#define store_i_r(i, r) if constexpr ((i < RM) && (r < RN)) \
                        GGML_F32_VEC_STORE(C + i * N + r * KN, acc##i##r)
#define store4(i) store_i_r(i, 0); \
                 store_i_r(i, 1); \
                 store_i_r(i, 2); \
                 store_i_r(i, 3);
#define load_b(r) if constexpr (r < RN) \
                  Bv##r = GGML_F32_VEC_LOAD(B + kk * N + r * KN)
// GGML_F32xt_FMA is svmad_f32_m, which is somehow slower
#define FMA(i, r) if constexpr ((i < RM) && (r < RN)) \
                  acc##i##r = svmla_f32_m(DEFAULT_PG, acc##i##r, Bv##r, p)
#define accum_a(i) p = GGML_F32_VEC_SET1(A[i * K + kk]); \
                   FMA(i, 0); \
                   FMA(i, 1); \
                   FMA(i, 2); \
                   FMA(i, 3)

template <int KN, int RM, int RN>
static inline void sve_simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N)
{

    define_ACC(0);
    define_ACC(1);
    define_ACC(2);
    define_ACC(3);

    load4(0);
    load4(1);
    load4(2);
    load4(3);
    
    for (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv0, Bv1, Bv2, Bv3, p;
        
        load_b(0);
        load_b(1);
        load_b(2);
        load_b(3);

        accum_a(0);
        accum_a(1);
        accum_a(2);
        accum_a(3);
    }

    store4(0);
    store4(1);
    store4(2);
    store4(3);
    
}
template<int KN>
static void sve_simd_gemm_kernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{

    int64_t ii = 0;
    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            sve_simd_gemm_ukernel<KN, GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            sve_simd_gemm_ukernel<KN, GEMM_RM, 1>(C + jj, A, B + jj, K, N);
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
            sve_simd_gemm_ukernel<KN, 1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            sve_simd_gemm_ukernel<KN, 1, 1>(C + jj, A, B + jj, K, N);
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

static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    int sve_vector_length = svcntb();
    switch(sve_vector_length){
        case 16:
        {
            sve_simd_gemm_kernel<4>(C, A, B, M, K, N);
            break;
        }
        case 32:
        {
            sve_simd_gemm_kernel<8>(C, A, B, M, K, N);
            break;
        }
        case 64:
        {
            sve_simd_gemm_kernel<16>(C, A, B, M, K, N);
            break;
        }
        // There's no SVE implementation beyond 512-bit?
        case 128:
        {
            sve_simd_gemm_kernel<32>(C, A, B, M, K, N);
            break;
        }
        default:
        {
            sve_simd_gemm_kernel<64>(C, A, B, M, K, N);
            break;
        }
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
