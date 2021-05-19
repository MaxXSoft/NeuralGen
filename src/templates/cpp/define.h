#ifndef NEURALGEN_DEFINE_H_
#define NEURALGEN_DEFINE_H_

#include <algorithm>    // max pooling
#include <cassert>      // GetIndex
#include <cmath>        // ActFuncs
#include <cstddef>      // size_t
#include <cstdint>      // module file struct
#include <cstring>      // main
#include <fstream>      // main
#include <iostream>     // main
#include <memory>       // main
#include <stdexcept>    // main
#include <string_view>  // main
#include <utility>      // main
#include <vector>       // main

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

#define ACT_FUNC(id) CONCAT(ActFunc_, id)
#define CONV_3D(id) CONCAT(Conv3D_, id)
#define POOLING(id) CONCAT(Pooling_, id)
#define FULL_CONN(id) CONCAT(FullConn_, id)

#define DECL_LAYER(type, id) \
  void type(id)(float *in, float *out, float *weight, float *bias)

inline size_t GetIndex(size_t x, size_t y, size_t channel, size_t width,
                       size_t height, size_t depth) {
  assert(x >= 0 && x < width);
  assert(y >= 0 && y < height);
  assert(channel >= 0 && channel < depth);
  return (height * channel + y) * width + x;
}

inline float ACT_FUNC(tanh)(float x) {
  auto ep = std::exp(x), em = std::exp(-x);
  return (ep - em) / (ep + em);
}

#endif  // NEURALGEN_DEFINE_H_
