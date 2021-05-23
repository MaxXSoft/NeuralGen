#define SCALE_FACTOR (1.0 / (KERNEL_WIDTH * KERNEL_HEIGHT))

DECL_LAYER(POOLING, LAYER_ID) {
  size_t i = get_global_id(0);
  size_t y = get_global_id(1);
  size_t x = get_global_id(2);
  size_t block = INPUT_WIDTH * INPUT_HEIGHT * i;
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
  out[index] *= SCALE_FACTOR;
#elif defined(FUNCTION_MAX)
  out[index] = -1e9;
  for (size_t m = 0; m < KERNEL_WIDTH; m++) {
    for (size_t n = 0; n < KERNEL_HEIGHT; n++) {
      float cur = weight[i] * in[(rows + m) * INPUT_WIDTH + cols + n + block];
      if (cur > out[index]) out[index] = cur;
    }
  }
#endif
  out[index] += bias[i];
  out[index] = ACT_FUNC(ACTIVATION)(out[index]);
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
