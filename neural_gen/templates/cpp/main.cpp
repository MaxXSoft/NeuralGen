#ifndef GENERATED
#include "define.h"
#define NETWORK_LAYERS(e) e(CONV_3D, 0, 100) e(FULL_CONN, 1, 10)
#define OUTPUT_SIZE 10
#endif  // GENERATED

// expand declarations of all layers
NETWORK_LAYERS(DECL_EXPANDER);

namespace {

/*
  Model File Format (field: bytes):

  MAGIC_NUMBER:       4
  LAYER_NUM:          4
  LAYERn_WEIGHT_SIZE: 4
  LAYERn_BIAS_SIZE:   4
  LAYERn_WEIGHT_DATA: LAYERn_WEIGHT_SIZE * sizeof(float)
  LAYERn_BIAS_DATA:   LAYERn_BIAS_SIZE * sizeof(float)
*/

// pointer to float array
using FloatArr = std::unique_ptr<float[]>;

// model data (vector of (weight, bias) pairs)
using ModelData = std::vector<std::pair<FloatArr, FloatArr>>;

// magic number: 1 go ge cal => yi gou ji calc => yi gou ji suan
constexpr uint32_t kModFileMagicNum = 0x1909eca1;

struct ModelFileHeader {
  uint32_t magic;
  uint32_t layer_num;
};

struct ModelLayerHeader {
  uint32_t weight_size;
  uint32_t bias_size;
};

// open file or fail
void OpenFile(std::ifstream &ifs, std::string_view file) {
  ifs.close();
  ifs.clear();
  ifs.open(std::string(file), std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open file!");
  }
}

// read model from file
ModelData ReadModel(std::istream &is) {
  // read file header
  ModelFileHeader mfh;
  is.read(reinterpret_cast<char *>(&mfh), sizeof(ModelFileHeader));
  if (mfh.magic != kModFileMagicNum) {
    throw std::runtime_error("Invalid model file, magic number mismatch!");
  }
  // read layers
  ModelLayerHeader mlh;
  ModelData model;
  for (size_t i = 0; i < mfh.layer_num; ++i) {
    is.read(reinterpret_cast<char *>(&mlh), sizeof(ModelLayerHeader));
    auto weight = std::make_unique<float[]>(mlh.weight_size);
    auto bias = std::make_unique<float[]>(mlh.bias_size);
    is.read(reinterpret_cast<char *>(weight.get()),
            mlh.weight_size * sizeof(float));
    is.read(reinterpret_cast<char *>(bias.get()),
            mlh.bias_size * sizeof(float));
    model.push_back({std::move(weight), std::move(bias)});
  }
  return model;
}

// read input from file
FloatArr ReadInput(std::istream &is) {
  // read input
  std::vector<float> arr;
  while (!is.fail() && !is.eof()) {
    float cur;
    is.read(reinterpret_cast<char *>(&cur), sizeof(float));
    arr.push_back(cur);
  }
  if (is.fail() && !is.eof()) throw std::runtime_error("File error!");
  // copy to float array
  auto input = std::make_unique<float[]>(arr.size());
  std::memcpy(input.get(), arr.data(), arr.size() * sizeof(float));
  return input;
}

// infer
FloatArr Infer(const ModelData &model, FloatArr input) {
  NETWORK_LAYERS(NETWORK_EXPANDER);
  return input;
}

// dump output to stderr
void DumpOutput(const FloatArr &output) {
  for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
    if (i) std::cerr << ' ';
    std::cerr << output[i];
  }
  std::cerr << std::endl;
}

// get the index of the maximum output
size_t GetMaxIndex(const FloatArr &output) {
  float max_elem = -1e9;
  size_t max_i = 0;
  for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
    if (output[i] > max_elem) {
      max_elem = output[i];
      max_i = i;
    }
  }
  return max_i;
}

}  // namespace

int main(int argc, const char *argv[]) {
  // check & parse command line arguments
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " MODEL <INPUT ...>" << std::endl;
    return 1;
  }
  std::string_view mod_file = argv[1];

  // read model data
  std::ifstream ifs;
  OpenFile(ifs, mod_file);
  auto model = ReadModel(ifs);

  // read inputs
  for (int i = 2; i < argc; ++i) {
    OpenFile(ifs, argv[i]);
    auto input = ReadInput(ifs);
    // infer
    auto output = Infer(model, std::move(input));
    DumpOutput(output);
    std::cout << GetMaxIndex(output) << std::endl;
  }
  return 0;
}

#undef NETWORK_LAYERS
#undef OUTPUT_SIZE
