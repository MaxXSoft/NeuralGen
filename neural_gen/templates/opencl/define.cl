#include <cassert>
#include <cstddef>

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

#define ACT_FUNC(id) CONCAT(ActFunc_, id)
#define CONV_3D(id) CONCAT(Conv3D_, id)
#define POOLING(id) CONCAT(Pooling_, id)
#define FULL_CONN(id) CONCAT(FullConn_, id)

#define DECL_LAYER(type, id)                                    \
  kernel void type(id)(global float *in, global float *out,     \
                       global float *weight, global float *bias)

inline size_t GetIndex(size_t x, size_t y, size_t channel, size_t width,
                       size_t height, size_t depth) {
  assert(x >= 0 && x < width);
  assert(y >= 0 && y < height);
  assert(channel >= 0 && channel < depth);
  return (height * channel + y) * width + x;
}

inline float ACT_FUNC(tanh)(float x) {
  auto ep = exp(x), em = exp(-x);
  return (ep - em) / (ep + em);
}
