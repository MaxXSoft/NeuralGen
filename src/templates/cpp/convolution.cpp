// for debugging
#ifndef GENERATED
#include <cstddef>

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
#endif

void CONV_3D(LAYER_ID)(float *in, float *out, float *weight, float *bias) {
  for (size_t channel = 0; channel < OUTPUT_DEPTH; ++channel) {
    for (size_t y = 0; y < OUTPUT_HEIGHT; ++y) {
      for (size_t x = 0; x < OUTPUT_WIDTH; ++x) {
        // current neuron
        size_t index =
            (channel * OUTPUT_HEIGHT * OUTPUT_WIDTH) + y * OUTPUT_WIDTH + x;
        out[index] = 0.0;
        // perform convolution
        for (size_t inc = 0; inc < INPUT_DEPTH; ++inc) {
          size_t addr1 =
              GetIndex(0, 0, INPUT_DEPTH * channel + inc, KERNEL_WIDTH,
                       KERNEL_HEIGHT, OUTPUT_DEPTH * INPUT_DEPTH);
          size_t addr2 =
              GetIndex(0, 0, inc, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH);
          // kernel
          const float *pw = weight + addr1;
          // input
          const float *pi = in + addr2;
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * INPUT_WIDTH + x;
          for (size_t wy = 0; wy < KERNEL_HEIGHT; wy++) {
            for (size_t wx = 0; wx < KERNEL_WIDTH; wx++) {
              sum += *ppw++ * ppi[wy * INPUT_WIDTH + wx];
            }
          }
          out[index] += sum;
        }
        // add bias and perform activation
        out[index] += bias[channel];
        out[index] = ACT_FUNC(ACTIVATION)(out[index]);
      }
    }
  }
}

#undef LAYER_ID
#undef PADDING_VALID
#undef PADDING_SAME
#undef STRIDE
#undef KERNEL_WIDTH
#undef KERNEL_HEIGHT
#undef OUTPUT_WIDTH
#undef OUTPUT_HEIGHT
#undef OUTPUT_DEPTH
#undef ACTIVATION
