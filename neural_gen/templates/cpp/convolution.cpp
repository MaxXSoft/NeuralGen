// for debugging
#ifndef GENERATED
#include "define.h"

#define LAYER_ID 0
#define PADDING_VALID
#define STRIDE 1
#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 5
#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define INPUT_DEPTH 1
#define OUTPUT_WIDTH 28
#define OUTPUT_HEIGHT 28
#define OUTPUT_DEPTH 6
#define ACTIVATION tanh
#endif  // GENERATED

DECL_LAYER(CONV_3D, LAYER_ID) {
#ifdef _OPENMP
#if defined(SIMD)
#if SIMD_ALIGN(OUTPUT_WIDTH) != 0 && SIMD_REMAIN(OUTPUT_WIDTH) != 0
#pragma omp parallel for collapse(2)
#else
#pragma omp parallel for collapse(3)
#endif
#else
#pragma omp parallel for collapse(3)
#endif
#endif  // _OPENMP
  for (size_t channel = 0; channel < OUTPUT_DEPTH; ++channel) {
    for (size_t y = 0; y < OUTPUT_HEIGHT; ++y) {
#ifdef SIMD
#if SIMD_ALIGN(OUTPUT_WIDTH) != 0
      for (size_t x = 0; x < SIMD_ALIGN(OUTPUT_WIDTH); x += SIMD_VEC_LEN) {
        // current neuron
        size_t index =
            (channel * OUTPUT_HEIGHT * OUTPUT_WIDTH) + y * OUTPUT_WIDTH + x;
        VecN mm_bias = SIMD_MM(set1_ps)(bias[channel]);
        VecN mm_cur = SIMD_MM(setzero_ps)();
        // perform convolution
        for (size_t inc = 0; inc < INPUT_DEPTH; ++inc) {
          size_t addr1 =
              GetIndex(0, 0, INPUT_DEPTH * channel + inc, KERNEL_WIDTH,
                       KERNEL_HEIGHT, OUTPUT_DEPTH * INPUT_DEPTH);
          size_t addr2 =
              GetIndex(0, 0, inc, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH);
          VecN mm_sum = SIMD_MM(setzero_ps)();
          // kernel
          const float *pw = weight + addr1;
          const float *ppw = pw;
          // input
          const float *pi = in + addr2;
          const float *ppi = pi + y * INPUT_WIDTH + x;
          for (size_t wy = 0; wy < KERNEL_HEIGHT; wy++) {
            for (size_t wx = 0; wx < KERNEL_WIDTH; wx++) {
              VecN mm_weight = SIMD_MM(set1_ps)(*ppw++);
              VecN mm_in = SIMD_MM(loadu_ps)(ppi + wy * INPUT_WIDTH + wx);
              mm_sum = SIMD_MM(add_ps)(mm_sum,
                                       SIMD_MM(mul_ps)(mm_weight, mm_in));
            }
          }
          mm_cur = SIMD_MM(add_ps)(mm_cur, mm_sum);
        }
        // add bias and perform activation
        SIMD_MM(storeu_ps)(out + index, SIMD_MM(add_ps)(mm_cur, mm_bias));
        out[index + 0] = ACT_FUNC(ACTIVATION)(out[index + 0]);
        out[index + 1] = ACT_FUNC(ACTIVATION)(out[index + 1]);
        out[index + 2] = ACT_FUNC(ACTIVATION)(out[index + 2]);
        out[index + 3] = ACT_FUNC(ACTIVATION)(out[index + 3]);
        out[index + 4] = ACT_FUNC(ACTIVATION)(out[index + 4]);
        out[index + 5] = ACT_FUNC(ACTIVATION)(out[index + 5]);
        out[index + 6] = ACT_FUNC(ACTIVATION)(out[index + 6]);
        out[index + 7] = ACT_FUNC(ACTIVATION)(out[index + 7]);
      }
#endif
#if SIMD_REMAIN(OUTPUT_WIDTH) != 0
      for (size_t x = SIMD_ALIGN(OUTPUT_WIDTH); x < OUTPUT_WIDTH; ++x) {
        // current neuron
        size_t index =
            (channel * OUTPUT_HEIGHT * OUTPUT_WIDTH) + y * OUTPUT_WIDTH + x;
        float cur = 0.0;
        // perform convolution
        for (size_t inc = 0; inc < INPUT_DEPTH; ++inc) {
          size_t addr1 =
              GetIndex(0, 0, INPUT_DEPTH * channel + inc, KERNEL_WIDTH,
                       KERNEL_HEIGHT, OUTPUT_DEPTH * INPUT_DEPTH);
          size_t addr2 =
              GetIndex(0, 0, inc, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH);
          float sum = 0.0;
          // kernel
          const float *pw = weight + addr1;
          const float *ppw = pw;
          // input
          const float *pi = in + addr2;
          const float *ppi = pi + y * INPUT_WIDTH + x;
          for (size_t wy = 0; wy < KERNEL_HEIGHT; wy++) {
            for (size_t wx = 0; wx < KERNEL_WIDTH; wx++) {
              sum += *ppw++ * ppi[wy * INPUT_WIDTH + wx];
            }
          }
          cur += sum;
        }
        // add bias and perform activation
        out[index] = ACT_FUNC(ACTIVATION)(cur + bias[channel]);
      }
#endif
#else
      for (size_t x = 0; x < OUTPUT_WIDTH; ++x) {
        // current neuron
        size_t index =
            (channel * OUTPUT_HEIGHT * OUTPUT_WIDTH) + y * OUTPUT_WIDTH + x;
        float cur = 0.0;
        // perform convolution
        for (size_t inc = 0; inc < INPUT_DEPTH; ++inc) {
          size_t addr1 =
              GetIndex(0, 0, INPUT_DEPTH * channel + inc, KERNEL_WIDTH,
                       KERNEL_HEIGHT, OUTPUT_DEPTH * INPUT_DEPTH);
          size_t addr2 =
              GetIndex(0, 0, inc, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH);
          float sum = 0.0;
          // kernel
          const float *pw = weight + addr1;
          const float *ppw = pw;
          // input
          const float *pi = in + addr2;
          const float *ppi = pi + y * INPUT_WIDTH + x;
          for (size_t wy = 0; wy < KERNEL_HEIGHT; wy++) {
            for (size_t wx = 0; wx < KERNEL_WIDTH; wx++) {
              sum += *ppw++ * ppi[wy * INPUT_WIDTH + wx];
            }
          }
          cur += sum;
        }
        // add bias and perform activation
        out[index] = ACT_FUNC(ACTIVATION)(cur + bias[channel]);
      }
#endif  // SIMD
    }
  }
}

#undef LAYER_ID
#undef PADDING_VALID
#undef STRIDE
#undef KERNEL_WIDTH
#undef KERNEL_HEIGHT
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef INPUT_DEPTH
#undef OUTPUT_WIDTH
#undef OUTPUT_HEIGHT
#undef OUTPUT_DEPTH
#undef ACTIVATION
