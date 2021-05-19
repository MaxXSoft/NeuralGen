// for debugging
#ifndef GENERATED
#include <cstddef>

#include "define.h"

#define LAYER_ID 0
#define INPUT_SIZE 120
#define OUTPUT_SIZE 10
#define ACTIVATION tanh
#endif

void FULL_CONN(LAYER_ID)(float *in, float *out, float *weight,
                         float *bias) {
  for (size_t i = 0; i < OUTPUT_SIZE; i++) {
    out[i] = 0.0;
    for (size_t c = 0; c < INPUT_SIZE; c++) {
      out[i] += weight[c * OUTPUT_SIZE + i] * in[c];
    }
    out[i] += bias[i];
    out[i] = ACT_FUNC(ACTIVATION)(out[i]);
  }
}

#undef LAYER_ID
#undef INPUT_SIZE
#undef OUTPUT_SIZE
#undef ACTIVATION
