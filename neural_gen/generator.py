from typing import TextIO
from neural_gen.layer import Layer, Input, Convolution, Pooling, FullConnection
from neural_gen.network import Network
from os import path


class Generator:
  '''
  Interface of neural network generator.
  '''

  '''
  Template directory.
  '''
  __TEMPLATE_DIR = path.join(path.dirname(
      path.realpath(__file__)), 'templates')

  def generate(self, network: Network) -> None:
    '''
    Generate on neural network.
    '''
    raise NotImplementedError

  def dump(self, f: TextIO) -> None:
    '''
    Dump the generated code to the specific file.
    '''
    raise NotImplementedError

  @staticmethod
  def _read_template(gen: str, name: str) -> str:
    '''
    Read template file.
    '''
    with open(path.join(Generator.__TEMPLATE_DIR, gen, name), 'r') as f:
      return f.read()


class CppGenerator(Generator):
  '''
  Generate C++ code for a nerual network.
  '''

  '''
  Type (in generated C++ code) of all layers.
  '''
  __LAYER_TYPE = {
      'input': None,
      'convolution': 'CONV_3D',
      'pooling': 'POOLING',
      'full_connection': 'FULL_CONN',
  }

  def __init__(self) -> None:
    # generated code
    self.__code = ''
    # load templates
    self.__define = Generator._read_template('cpp', 'define.h')
    self.__main = Generator._read_template('cpp', 'main.cpp')
    self.__convolution = Generator._read_template('cpp', 'convolution.cpp')
    self.__pooling = Generator._read_template('cpp', 'pooling.cpp')
    self.__fullconn = Generator._read_template('cpp', 'fullconn.cpp')

  def __gen_input(self, layer_id: int, layer: Input, last_layer: Layer) -> None:
    '''
    Generate input layer.
    '''
    # do nothing
    pass

  def __gen_conv(self, layer_id: int, layer: Convolution, last_layer: Layer) -> None:
    '''
    Generate convolution layer.
    '''
    last_width, last_height, last_depth = last_layer.get_output_shape()
    self.__code += f'#define LAYER_ID {layer_id}\n'
    self.__code += f'#define PADDING_{layer["padding"].upper()}\n'
    self.__code += f'#define STRIDE {layer["stride"]}\n'
    self.__code += f'#define KERNEL_WIDTH {layer["kernel"]["width"]}\n'
    self.__code += f'#define KERNEL_HEIGHT {layer["kernel"]["height"]}\n'
    self.__code += f'#define INPUT_WIDTH {last_width}\n'
    self.__code += f'#define INPUT_HEIGHT {last_height}\n'
    self.__code += f'#define INPUT_DEPTH {last_depth}\n'
    self.__code += f'#define OUTPUT_WIDTH {layer["output"]["width"]}\n'
    self.__code += f'#define OUTPUT_HEIGHT {layer["output"]["height"]}\n'
    self.__code += f'#define OUTPUT_DEPTH {layer["output"]["depth"]}\n'
    self.__code += f'#define ACTIVATION {layer["activation"]}\n\n'
    self.__code += f'{self.__convolution}\n'

  def __gen_pooling(self, layer_id: int, layer: Pooling, last_layer: Layer) -> None:
    '''
    Generate pooling layer.
    '''
    last_width, last_height, last_depth = last_layer.get_output_shape()
    self.__code += f'#define LAYER_ID {layer_id}\n'
    self.__code += f'#define FUNCTION_{layer["function"].upper()}\n'
    self.__code += f'#define PADDING_{layer["padding"].upper()}\n'
    self.__code += f'#define STRIDE {layer["stride"]}\n'
    self.__code += f'#define KERNEL_WIDTH {layer["kernel"]["width"]}\n'
    self.__code += f'#define KERNEL_HEIGHT {layer["kernel"]["height"]}\n'
    self.__code += f'#define INPUT_WIDTH {last_width}\n'
    self.__code += f'#define INPUT_HEIGHT {last_height}\n'
    self.__code += f'#define INPUT_DEPTH {last_depth}\n'
    self.__code += f'#define OUTPUT_WIDTH {layer["output"]["width"]}\n'
    self.__code += f'#define OUTPUT_HEIGHT {layer["output"]["height"]}\n'
    self.__code += f'#define OUTPUT_DEPTH {layer["output"]["depth"]}\n'
    self.__code += f'#define ACTIVATION {layer["activation"]}\n\n'
    self.__code += f'{self.__pooling}\n'

  def __gen_full_conn(self, layer_id: int, layer: FullConnection, last_layer: Layer) -> None:
    '''
    Generate fully connection layer.
    '''
    last_size = last_layer.get_output_size()
    size = layer["output_size"]
    self.__code += f'#define LAYER_ID {layer_id}\n'
    self.__code += f'#define INPUT_SIZE {last_size}\n'
    self.__code += f'#define OUTPUT_SIZE {size}\n'
    self.__code += f'#define ACTIVATION {layer["activation"]}\n\n'
    self.__code += f'{self.__fullconn}\n'

  def generate(self, network: Network) -> None:
    self.__code = '#define GENERATED\n\n'
    # generate the architecture of network
    layer_desc = []
    for i, layer in enumerate(network.layers):
      layer_type = CppGenerator.__LAYER_TYPE[layer.layer_type()]
      if layer_type:
        layer_desc.append(f'e({layer_type}, {i}, {layer.get_output_size()})')
    self.__code += f'#define NETWORK_LAYERS(e) {" ".join(layer_desc)}\n'
    self.__code += f'#define OUTPUT_SIZE {network.layers[-1].get_output_size()}\n\n'
    self.__code += f'{self.__define}\n'
    self.__code += f'{self.__main}\n'
    # generate all layers
    layer_gen = {
        'input': self.__gen_input,
        'convolution': self.__gen_conv,
        'pooling': self.__gen_pooling,
        'full_connection': self.__gen_full_conn,
    }
    for i, layer in enumerate(network.layers):
      layer_gen[layer.layer_type()](
          i, layer, network.layers[i - 1] if i - 1 >= 0 else None)

  def dump(self, f: TextIO) -> None:
    f.write(self.__code)


class OpenCLGenerator(Generator):
  '''
  Generate OpenCL code for a nerual network.
  '''

  '''
  Type (in generated OpenCL code) of all layers.
  '''
  __LAYER_TYPE = {
      'input': None,
      'convolution': 'CONV_3D',
      'pooling': 'POOLING',
      'full_connection': 'FULL_CONN',
  }

  def __init__(self) -> None:
    # generated code
    self.__code = ''
    # load templates
    self.__main = Generator._read_template('opencl', 'main.cpp')
    self.__define = Generator._read_template('opencl', 'define.cl')
    self.__convolution = Generator._read_template('opencl', 'convolution.cl')
    self.__pooling = Generator._read_template('opencl', 'pooling.cl')
    self.__fullconn = Generator._read_template('opencl', 'fullconn.cl')

  def __gen_input(self, layer_id: int, layer: Input, last_layer: Layer) -> None:
    '''
    Generate input layer.
    '''
    # do nothing
    pass

  def __gen_conv(self, layer_id: int, layer: Convolution, last_layer: Layer) -> None:
    '''
    Generate convolution layer.
    '''
    last_width, last_height, last_depth = last_layer.get_output_shape()
    self.__code += f'#define LAYER_ID {layer_id}\n'
    self.__code += f'#define PADDING_{layer["padding"].upper()}\n'
    self.__code += f'#define STRIDE {layer["stride"]}\n'
    self.__code += f'#define KERNEL_WIDTH {layer["kernel"]["width"]}\n'
    self.__code += f'#define KERNEL_HEIGHT {layer["kernel"]["height"]}\n'
    self.__code += f'#define INPUT_WIDTH {last_width}\n'
    self.__code += f'#define INPUT_HEIGHT {last_height}\n'
    self.__code += f'#define INPUT_DEPTH {last_depth}\n'
    self.__code += f'#define OUTPUT_WIDTH {layer["output"]["width"]}\n'
    self.__code += f'#define OUTPUT_HEIGHT {layer["output"]["height"]}\n'
    self.__code += f'#define OUTPUT_DEPTH {layer["output"]["depth"]}\n'
    self.__code += f'#define ACTIVATION {layer["activation"]}\n\n'
    self.__code += f'{self.__convolution}\n'

  def __gen_pooling(self, layer_id: int, layer: Pooling, last_layer: Layer) -> None:
    '''
    Generate pooling layer.
    '''
    last_width, last_height, last_depth = last_layer.get_output_shape()
    self.__code += f'#define LAYER_ID {layer_id}\n'
    self.__code += f'#define FUNCTION_{layer["function"].upper()}\n'
    self.__code += f'#define PADDING_{layer["padding"].upper()}\n'
    self.__code += f'#define STRIDE {layer["stride"]}\n'
    self.__code += f'#define KERNEL_WIDTH {layer["kernel"]["width"]}\n'
    self.__code += f'#define KERNEL_HEIGHT {layer["kernel"]["height"]}\n'
    self.__code += f'#define INPUT_WIDTH {last_width}\n'
    self.__code += f'#define INPUT_HEIGHT {last_height}\n'
    self.__code += f'#define INPUT_DEPTH {last_depth}\n'
    self.__code += f'#define OUTPUT_WIDTH {layer["output"]["width"]}\n'
    self.__code += f'#define OUTPUT_HEIGHT {layer["output"]["height"]}\n'
    self.__code += f'#define OUTPUT_DEPTH {layer["output"]["depth"]}\n'
    self.__code += f'#define ACTIVATION {layer["activation"]}\n\n'
    self.__code += f'{self.__pooling}\n'

  def __gen_full_conn(self, layer_id: int, layer: FullConnection, last_layer: Layer) -> None:
    '''
    Generate fully connection layer.
    '''
    last_size = last_layer.get_output_size()
    size = layer["output_size"]
    self.__code += f'#define LAYER_ID {layer_id}\n'
    self.__code += f'#define INPUT_SIZE {last_size}\n'
    self.__code += f'#define OUTPUT_SIZE {size}\n'
    self.__code += f'#define ACTIVATION {layer["activation"]}\n\n'
    self.__code += f'{self.__fullconn}\n'

  def generate(self, network: Network) -> None:
    self.__code = '#define GENERATED\n\n'
    self.__code += 'const char *kOpenCLOptions = "";\n'
    self.__code += 'const char *kOpenCLProgram = R"(\n'
    self.__code += f'{self.__define}\n'
    # generate all layers
    layer_gen = {
        'input': self.__gen_input,
        'convolution': self.__gen_conv,
        'pooling': self.__gen_pooling,
        'full_connection': self.__gen_full_conn,
    }
    for i, layer in enumerate(network.layers):
      layer_gen[layer.layer_type()](
          i, layer, network.layers[i - 1] if i - 1 >= 0 else None)
    self.__code += ')";\n\n'
    # generate the architecture of network
    layer_desc = []
    for i, layer in enumerate(network.layers):
      layer_type = OpenCLGenerator.__LAYER_TYPE[layer.layer_type()]
      if layer_type:
        w, h, d = layer.get_output_shape()
        layer_desc.append(f'e({layer_type}, {i}, {w}, {h}, {d})')
    self.__code += f'#define NETWORK_LAYERS(e) {" ".join(layer_desc)}\n'
    self.__code += f'#define OUTPUT_SIZE {network.layers[-1].get_output_size()}\n\n'
    self.__code += f'{self.__main}\n'

  def dump(self, f: TextIO) -> None:
    f.write(self.__code)
