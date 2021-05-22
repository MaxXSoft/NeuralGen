DECL_LAYER(FULL_CONN, LAYER_ID) {
  size_t i = get_global_id(0);
  out[i] = 0.0;
  for (size_t c = 0; c < INPUT_SIZE; c++) {
    out[i] += weight[c * OUTPUT_SIZE + i] * in[c];
  }
  out[i] += bias[i];
  out[i] = ACT_FUNC(ACTIVATION)(out[i]);
}

#undef LAYER_ID
#undef INPUT_SIZE
#undef OUTPUT_SIZE
#undef ACTIVATION
