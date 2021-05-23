#ifndef GENERATED
const char *kOpenCLOptions = "";
const char *kOpenCLProgram = "";
#define NETWORK_LAYERS(e) e(CONV_3D, 0, 28, 28, 6) e(FULL_CONN, 1, 10, 1, 1)
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
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CONCAT_IMPL(x, y) #x #y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

#define CONV_3D(id) CONCAT(Conv3D_, id)
#define POOLING(id) CONCAT(Pooling_, id)
#define FULL_CONN(id) CONCAT(FullConn_, id)

namespace {

// pointer to OpenCL context
using ContextPtr = std::unique_ptr<std::remove_pointer_t<cl_context>,
                                   decltype(&clReleaseContext)>;
// pointer to command queue
using CmdQueuePtr = std::unique_ptr<std::remove_pointer_t<cl_command_queue>,
                                    decltype(&clReleaseCommandQueue)>;
// pointer to program
using ProgramPtr = std::unique_ptr<std::remove_pointer_t<cl_program>,
                                   decltype(&clReleaseProgram)>;
// pointer to kernel
using KernelPtr = std::unique_ptr<std::remove_pointer_t<cl_kernel>,
                                  decltype(&clReleaseKernel)>;
// pointer to OpenCL buffer
using BufferPtr = std::unique_ptr<std::remove_pointer_t<cl_mem>,
                                  decltype(&clReleaseMemObject)>;

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

// float vector
using FloatVec = std::vector<float>;

// model data (vector of (weight, bias) pairs)
using ModelData = std::vector<std::pair<BufferPtr, BufferPtr>>;

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

// OpenCL devices
std::vector<cl_device_id> devices;
// the selected OpenCL device
cl_device_id device;
// OpenCL context
ContextPtr context = {nullptr, nullptr};
// OpenCL command queue
CmdQueuePtr cmd_queue = {nullptr, nullptr};
// OpenCL program
ProgramPtr program = {nullptr, nullptr};
// OpenCL kernels of all layers
std::unordered_map<size_t, KernelPtr> kernels;

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
  context =
      ContextPtr(clCreateContext(nullptr, devices.size(), devices.data(),
                                 nullptr, nullptr, &err),
                 clRetainContext);
  if (err) throw std::runtime_error("failed to create context");
  // initialize command queue
  cmd_queue =
      CmdQueuePtr(clCreateCommandQueue(context.get(), device, 0, &err),
                  clReleaseCommandQueue);
  if (err) throw std::runtime_error("failed to create command queue");
}

// load OpenCL program
void LoadProgram() {
  cl_int err;
  // create program
  program =
      ProgramPtr(clCreateProgramWithSource(context.get(), 1,
                                           &kOpenCLProgram, nullptr, &err),
                 clReleaseProgram);
  if (err) throw std::runtime_error("failed to create program");
  // build program for devices
  if (clBuildProgram(program.get(), devices.size(), devices.data(),
                     kOpenCLOptions, nullptr, nullptr)) {
    // read compile log
    size_t log_size;
    clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG, 0,
                          nullptr, &log_size);
    std::string comp_log;
    comp_log.resize(log_size);
    clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG,
                          log_size, comp_log.data(), nullptr);
    throw std::runtime_error("failed to build program\n" + comp_log);
  }
}

// initialize kernels of all layers
void InitKernels() {
#define NETWORK_EXPANDER(type, id, width, height, depth)              \
  do {                                                                \
    cl_int err;                                                       \
    kernels.insert(                                                   \
        {id, KernelPtr(clCreateKernel(program.get(), type(id), &err), \
                       clReleaseKernel)});                            \
    if (err) throw std::runtime_error("failed to create kernel");     \
  } while (0);

  NETWORK_LAYERS(NETWORK_EXPANDER);

#undef NETWORK_EXPANDER
}

// create a new OpenCL buffer
BufferPtr NewBuffer(size_t size, cl_mem_flags flags) {
  cl_int err;
  auto buffer =
      BufferPtr(clCreateBuffer(context.get(), flags, size, nullptr, &err),
                clReleaseMemObject);
  if (err) throw std::runtime_error("failed to create OpenCL buffer");
  return buffer;
}

// write data to the specific OpenCL buffer
void WriteBuffer(const BufferPtr &buffer, void *mem, size_t size) {
  if (clEnqueueWriteBuffer(cmd_queue.get(), buffer.get(), CL_FALSE, 0, size,
                           mem, 0, nullptr, nullptr)) {
    throw std::runtime_error("failed to write OpenCL buffer");
  }
}

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
    // read weight & bias
    is.read(reinterpret_cast<char *>(&mlh), sizeof(ModelLayerHeader));
    auto weight = std::make_unique<float[]>(mlh.weight_size);
    auto bias = std::make_unique<float[]>(mlh.bias_size);
    is.read(reinterpret_cast<char *>(weight.get()),
            mlh.weight_size * sizeof(float));
    is.read(reinterpret_cast<char *>(bias.get()),
            mlh.bias_size * sizeof(float));
    // create buffer for weight & bias
    if (!mlh.weight_size || !mlh.bias_size) {
      model.push_back(
          {BufferPtr(nullptr, nullptr), BufferPtr(nullptr, nullptr)});
    }
    else {
      auto weight_buf =
          NewBuffer(mlh.weight_size * sizeof(float), CL_MEM_READ_ONLY);
      auto bias_buf =
          NewBuffer(mlh.bias_size * sizeof(float), CL_MEM_READ_ONLY);
      WriteBuffer(weight_buf, weight.get(),
                  mlh.weight_size * sizeof(float));
      WriteBuffer(bias_buf, bias.get(), mlh.bias_size * sizeof(float));
      model.push_back({std::move(weight_buf), std::move(bias_buf)});
    }
  }
  return model;
}

// read input from file
FloatVec ReadInput(std::istream &is) {
  // read input
  FloatVec arr;
  while (!is.fail() && !is.eof()) {
    float cur;
    is.read(reinterpret_cast<char *>(&cur), sizeof(float));
    arr.push_back(cur);
  }
  if (is.fail() && !is.eof()) throw std::runtime_error("File error!");
  return arr;
}

// infer
FloatArr Infer(const ModelData &model, FloatVec input) {
#define NETWORK_EXPANDER(type, id, width, height, depth)                 \
  do {                                                                   \
    /* create output buffer */                                           \
    auto out_buf = NewBuffer(width * height * depth * sizeof(float),     \
                             CL_MEM_READ_WRITE);                         \
    /* get pointer of the current kernel */                              \
    const auto &kernel = kernels.find(id)->second;                       \
    /* set kernel arguments */                                           \
    auto in = buffer.get(), out = out_buf.get();                         \
    auto weight = model[id].first.get(), bias = model[id].second.get();  \
    if (clSetKernelArg(kernel.get(), 0, sizeof(cl_mem), &in) ||          \
        clSetKernelArg(kernel.get(), 1, sizeof(cl_mem), &out) ||         \
        clSetKernelArg(kernel.get(), 2, sizeof(cl_mem), &weight) ||      \
        clSetKernelArg(kernel.get(), 3, sizeof(cl_mem), &bias)) {        \
      throw std::runtime_error("failed to set argument");                \
    }                                                                    \
    /* run kernel */                                                     \
    cl_int ret;                                                          \
    size_t global_worksize[3] = {depth, height, width};                  \
    if ((ret = clEnqueueNDRangeKernel(cmd_queue.get(), kernel.get(), 3,  \
                                      nullptr, global_worksize, nullptr, \
                                      0, nullptr, nullptr))) {           \
      throw std::runtime_error(                                          \
          "error when executing kernel, error code: " +                  \
          std::to_string(ret));                                          \
    }                                                                    \
    clFinish(cmd_queue.get());                                           \
    /* update for next layer */                                          \
    buffer = std::move(out_buf);                                         \
    out_id = id;                                                         \
    last_out_size = width * height * depth;                              \
  } while (0);

  size_t out_id = 0, last_out_size;
  // create input buffer
  auto buffer = NewBuffer(input.size() * sizeof(float), CL_MEM_READ_ONLY);
  WriteBuffer(buffer, input.data(), input.size() * sizeof(float));
  // perform inference
  NETWORK_LAYERS(NETWORK_EXPANDER);
  // get output
  auto output = std::make_unique<float[]>(last_out_size);
  if (clEnqueueReadBuffer(cmd_queue.get(), buffer.get(), CL_TRUE, 0,
                          last_out_size * sizeof(float), output.get(), 0,
                          nullptr, nullptr)) {
    throw std::runtime_error("failed to read OpenCL buffer");
  }
  return output;

#undef NETWORK_EXPANDER
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
  InitKernels();

  // read model data
  std::ifstream ifs;
  OpenFile(ifs, mod_file);
  auto model = ReadModel(ifs);

  // read inputs
  for (int i = 4; i < argc; ++i) {
    // read input from file
    OpenFile(ifs, argv[i]);
    auto input = ReadInput(ifs);
    // infer
    auto output = Infer(model, std::move(input));
    DumpOutput(output);
    std::cout << GetMaxIndex(output) << std::endl;
  }
  return 0;
}
