// for debugging
#ifndef GENERATED
#include <algorithm>
#include <cstddef>

#include "define.h"

#define LAYER_ID 0
#define FUNCTION_AVERAGE
#define PADDING_VALID
#define STRIDE 1
#define KERNEL_WIDTH 2
#define KERNEL_HEIGHT 2
#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define INPUT_DEPTH 6
#define OUTPUT_WIDTH 14
#define OUTPUT_HEIGHT 14
#define OUTPUT_DEPTH 6
#define ACTIVATION tanh
#endif

void POOLING(LAYER_ID)(float *in, float *out, float *weight, float *bias) {
  for (size_t i = 0; i < OUTPUT_DEPTH; i++) {
    size_t block = INPUT_WIDTH * INPUT_HEIGHT * i;
    for (size_t y = 0; y < OUTPUT_HEIGHT; y++) {
      for (size_t x = 0; x < OUTPUT_WIDTH; x++) {
        size_t rows = y * KERNEL_WIDTH;
        size_t cols = x * KERNEL_HEIGHT;
        size_t index =
            (i * OUTPUT_HEIGHT * OUTPUT_WIDTH) + y * OUTPUT_WIDTH + x;
#if defined(FUNCTION_AVERAGE)
        out[index] = 0.0;
        for (size_t m = 0; m < KERNEL_WIDTH; m++) {
          for (size_t n = 0; n < KERNEL_HEIGHT; n++) {
            out[index] +=
                weight[i] * in[(rows + m) * INPUT_WIDTH + cols + n + block];
          }
        }
        constexpr float kScaleFactor = 1.0 / (KERNEL_WIDTH * KERNEL_HEIGHT);
        out[index] *= kScaleFactor;
#elif defined(FUNCTION_MAX)
        out[index] = -1e9;
        for (size_t m = 0; m < KERNEL_WIDTH; m++) {
          for (size_t n = 0; n < KERNEL_HEIGHT; n++) {
            out[index] = std::max(
                out[index],
                weight[i] *
                    in[(rows + m) * INPUT_WIDTH + cols + n + block]);
          }
        }
#endif
        out[index] += bias[i];
        out[index] = ACT_FUNC(ACTIVATION)(out[index]);
      }
    }
  }
}

#undef LAYER_ID
#undef FUNCTION_AVERAGE
#undef FUNCTION_MAX
#undef PADDING_VALID
#undef PADDING_SAME
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
