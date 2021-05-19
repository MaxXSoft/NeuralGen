from typing import Type, Optional, TextIO
from layer import Layer, Input, Convolution, Pooling, FullConnection
from network import Network
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
  Type of all layers.
  '''
  __LAYER_TYPE = {
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

  def __gen_input(self, layer_id: int, last_layer: Optional[Type[Layer]], layer: Input) -> None:
    '''
    Generate input layer.
    '''
    # do nothing
    pass

  def __gen_conv(self, layer_id: int, last_layer: Optional[Type[Layer]], layer: Convolution) -> None:
    '''
    Generate convolution layer.
    '''
    self.__code += f'#define LAYER_ID {layer_id}\n'
    self.__code += f'#define PADDING_{layer["padding"].upper()}\n'
    self.__code += f'#define STRIDE {layer["stride"]}\n'
    self.__code += f'#define KERNEL_WIDTH {layer["kernel"]["width"]}\n'
    self.__code += f'#define KERNEL_HEIGHT {layer["kernel"]["height"]}\n'
    # TODO
    self.__code += f'#define INPUT_WIDTH ???\n'
    self.__code += f'#define INPUT_HEIGHT ???\n'
    self.__code += f'#define INPUT_DEPTH ???\n'
    self.__code += f'#define OUTPUT_WIDTH {layer["kernel"]["width"]}\n'
    self.__code += f'#define OUTPUT_HEIGHT {layer["kernel"]["height"]}\n'
    self.__code += f'#define OUTPUT_DEPTH {layer["kernel"]["depth"]}\n'
    self.__code += f'#define ACTIVATION {layer["activation"]}\n'

  def __gen_pooling(self, layer_id: int, last_layer: Optional[Type[Layer]], layer: Pooling) -> None:
    '''
    Generate pooling layer.
    '''
    # TODO
    pass

  def __gen_full_conn(self, layer_id: int, last_layer: Optional[Type[Layer]], layer: FullConnection) -> None:
    '''
    Generate fully connection layer.
    '''
    # TODO
    pass

  def generate(self, network: Network) -> None:
    self.__code = '#define GENERATED\n\n'
    # generate the architecture of network
    layer_desc = []
    for i, layer in enumerate(network.layers):
      layer_type = CppGenerator.__LAYER_TYPE[layer.layer_type()]
      layer_desc.append(f'e({layer_type}, {i}, {layer.get_output_size()})')
    self.__code += f'#define NETWORK_LAYERS {" ".join(layer_desc)}\n'
    self.__code += f'#define OUTPUT_SIZE {network.layers[-1].get_output_size()}\n\n'
    self.__code += self.__define
    self.__code += self.__main
    # generate all layers
    layer_gen = {
        'input': self.__gen_input,
        'convolution': self.__gen_conv,
        'pooling': self.__gen_pooling,
        'full_connection': self.__gen_full_conn,
    }
    for i, layer in enumerate(network.layers):
      layer_gen[layer.layer_type()](
          i, network.layers[i - 1] if i else None, layer)

  def dump(self, f: TextIO) -> None:
    f.write(self.__code)
