DECL_LAYER(CONV_3D, LAYER_ID) {
  size_t channel = get_global_id(0);
  size_t y = get_global_id(1);
  size_t x = get_global_id(2);
  // current neuron
  size_t index =
      (channel * OUTPUT_HEIGHT * OUTPUT_WIDTH) + y * OUTPUT_WIDTH + x;
  float cur = 0.0;
  // perform convolution
  for (size_t inc = 0; inc < INPUT_DEPTH; ++inc) {
    size_t addr1 = GetIndex(0, 0, INPUT_DEPTH * channel + inc, KERNEL_WIDTH,
                            KERNEL_HEIGHT);
    size_t addr2 = GetIndex(0, 0, inc, INPUT_WIDTH, INPUT_HEIGHT);
    float sum = 0.0;
    // kernel
#ifdef OPT
    constant const float *pw = weight + addr1;
    constant const float *ppw = pw;
#else
    global const float *pw = weight + addr1;
    global const float *ppw = pw;
#endif  // OPT
    // input
    global const float *pi = in + addr2;
    global const float *ppi = pi + y * INPUT_WIDTH + x;
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
