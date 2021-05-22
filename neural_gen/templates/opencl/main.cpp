#ifndef GENERATED
const char *kOpenCLProgram = "";
#define NETWORK_LAYERS(e) e(CONV_3D, 0, 100) e(FULL_CONN, 1, 10)
#define OUTPUT_SIZE 10
#endif  // GENERATED

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CONCAT_IMPL(x, y) x #y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

#define CONV_3D(id) CONCAT(Conv3D_, id)
#define POOLING(id) CONCAT(Pooling_, id)
#define FULL_CONN(id) CONCAT(FullConn_, id)

#define NETWORK_EXPANDER(type, id, out_size) \
  do {                                       \
    /* TODO */                               \
  } while (0);

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

// pointer to OpenCL context
using ContextPtr = std::unique_ptr<std::remove_pointer_t<cl_context>,
                                   decltype(&clReleaseContext)>;
// pointer to command queue
using CmdQueuePtr = std::unique_ptr<std::remove_pointer_t<cl_command_queue>,
                                    decltype(&clReleaseCommandQueue)>;
// pointer to program
using ProgramPtr = std::unique_ptr<std::remove_pointer_t<cl_program>,
                                   decltype(&clReleaseProgram)>;

// OpenCL devices
std::vector<cl_device_id> devices;
// the selected OpenCL device
cl_device_id device;
// OpenCL context
ContextPtr context_;
// OpenCL command queue
CmdQueuePtr cmd_queue_;
// OpenCL program
ProgramPtr program_;

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

// initialize OpenCL device
void InitDevice(size_t platform_id, size_t device_id) {
  // initialize platform info
  cl_uint plat_num;
  if (clGetPlatformIDs(0, nullptr, &plat_num) || plat_num <= platform_id) {
    throw std::runtime_error("invalid platform configuration");
  }
  std::vector<cl_platform_id> plats;
  plats.resize(plat_num);
  if (clGetPlatformIDs(plat_num, plats.data(), nullptr)) {
    throw std::runtime_error("failed to read platform ids");
  }
  // initialize device
  cl_uint dev_num;
  if (clGetDeviceIDs(plats[platform_id], CL_DEVICE_TYPE_ALL, 0, nullptr,
                     &dev_num) ||
      dev_num <= device_id) {
    throw std::runtime_error("invalid device configuration");
  }
  devices.resize(dev_num);
  if (clGetDeviceIDs(plats[platform_id], CL_DEVICE_TYPE_ALL, dev_num,
                     devices.data(), nullptr)) {
    throw std::runtime_error("failed to read device ids");
  }
  device = devices[device_id];
}

// initialize context & command queue
void InitContext() {
  cl_int err;
  // initialize context
  context_ =
      ContextPtr(clCreateContext(nullptr, devices.size(), devices.data(),
                                 nullptr, nullptr, &err),
                 clRetainContext);
  if (err) throw std::runtime_error("failed to create context");
  // initialize command queue
  cmd_queue_ =
      CmdQueuePtr(clCreateCommandQueue(context_.get(), device, 0, &err),
                  clReleaseCommandQueue);
  if (err) throw std::runtime_error("failed to create command queue");
}

// load OpenCL program
void LoadProgram() {
  cl_int err;
  // create program
  program_ =
      ProgramPtr(clCreateProgramWithSource(context_.get(), 1,
                                           &kOpenCLProgram, nullptr, &err),
                 clReleaseProgram);
  if (err) throw std::runtime_error("failed to create program");
  // build program for devices
  if (clBuildProgram(program_.get(), devices.size(), devices.data(),
                     nullptr, nullptr, nullptr)) {
    // read compile log
    size_t log_size;
    clGetProgramBuildInfo(program_.get(), device, CL_PROGRAM_BUILD_LOG, 0,
                          nullptr, &log_size);
    std::string comp_log;
    comp_log.resize(log_size);
    clGetProgramBuildInfo(program_.get(), device, CL_PROGRAM_BUILD_LOG,
                          log_size, comp_log.data(), nullptr);
    throw std::runtime_error("failed to build program\n" + comp_log);
  }
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
  // check & parse arguments
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " PLAT_ID DEV_ID MODEL <INPUT ...>"
              << std::endl;
    return 1;
  }
  auto plat_id = std::strtoul(argv[1], nullptr, 10);
  auto dev_id = std::strtoul(argv[2], nullptr, 10);
  std::string_view mod_file = argv[3];

  // initialize OpenCL related stuffs
  InitDevice(plat_id, dev_id);
  InitContext();
  LoadProgram();

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
