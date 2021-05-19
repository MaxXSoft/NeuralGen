from typing import Optional, Tuple, TextIO
from neural_gen.layer import Input, Convolution, Pooling, FullConnection
from neural_gen.network import Network
from os import path


'''
Type of (width, height, depth) tuple.
'''
__WhdTuple = Optional[Tuple[int, int, int]]


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

  def __gen_input(self, layer_id: int, layer: Input, last_whd: '__WhdTuple') -> '__WhdTuple':
    '''
    Generate input layer.
    '''
    # do nothing
    return (layer['width'], layer['height'], layer['depth'])

  def __gen_conv(self, layer_id: int, layer: Convolution, last_whd: '__WhdTuple') -> '__WhdTuple':
    '''
    Generate convolution layer.
    '''
    last_width, last_height, last_depth = last_whd
    width = layer["output"]["width"]
    height = layer["output"]["height"]
    depth = layer["output"]["depth"]
    self.__code += f'#define LAYER_ID {layer_id}\n'
    self.__code += f'#define PADDING_{layer["padding"].upper()}\n'
    self.__code += f'#define STRIDE {layer["stride"]}\n'
    self.__code += f'#define KERNEL_WIDTH {layer["kernel"]["width"]}\n'
    self.__code += f'#define KERNEL_HEIGHT {layer["kernel"]["height"]}\n'
    self.__code += f'#define INPUT_WIDTH {last_width}\n'
    self.__code += f'#define INPUT_HEIGHT {last_height}\n'
    self.__code += f'#define INPUT_DEPTH {last_depth}\n'
    self.__code += f'#define OUTPUT_WIDTH {width}\n'
    self.__code += f'#define OUTPUT_HEIGHT {height}\n'
    self.__code += f'#define OUTPUT_DEPTH {depth}\n'
    self.__code += f'#define ACTIVATION {layer["activation"]}\n\n'
    self.__code += f'{self.__convolution}\n'
    return (width, height, depth)

  def __gen_pooling(self, layer_id: int, layer: Pooling, last_whd: '__WhdTuple') -> '__WhdTuple':
    '''
    Generate pooling layer.
    '''
    last_width, last_height, last_depth = last_whd
    width = layer["output"]["width"]
    height = layer["output"]["height"]
    depth = layer["output"]["depth"]
    self.__code += f'#define LAYER_ID {layer_id}\n'
    self.__code += f'#define FUNCTION_{layer["function"].upper()}\n'
    self.__code += f'#define PADDING_{layer["padding"].upper()}\n'
    self.__code += f'#define STRIDE {layer["stride"]}\n'
    self.__code += f'#define KERNEL_WIDTH {layer["kernel"]["width"]}\n'
    self.__code += f'#define KERNEL_HEIGHT {layer["kernel"]["height"]}\n'
    self.__code += f'#define INPUT_WIDTH {last_width}\n'
    self.__code += f'#define INPUT_HEIGHT {last_height}\n'
    self.__code += f'#define INPUT_DEPTH {last_depth}\n'
    self.__code += f'#define OUTPUT_WIDTH {width}\n'
    self.__code += f'#define OUTPUT_HEIGHT {height}\n'
    self.__code += f'#define OUTPUT_DEPTH {depth}\n'
    self.__code += f'#define ACTIVATION {layer["activation"]}\n\n'
    self.__code += f'{self.__pooling}\n'
    return (width, height, depth)

  def __gen_full_conn(self, layer_id: int, layer: FullConnection, last_whd: '__WhdTuple') -> '__WhdTuple':
    '''
    Generate fully connection layer.
    '''
    _, _, last_size = last_whd
    size = layer["output_size"]
    self.__code += f'#define LAYER_ID {layer_id}\n'
    self.__code += f'#define INPUT_SIZE {last_size}\n'
    self.__code += f'#define OUTPUT_SIZE {size}\n'
    self.__code += f'#define ACTIVATION {layer["activation"]}\n\n'
    self.__code += f'{self.__fullconn}\n'
    return (1, 1, size)

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
    last_whd = None
    for i, layer in enumerate(network.layers):
      last_whd = layer_gen[layer.layer_type()](i, layer, last_whd)

  def dump(self, f: TextIO) -> None:
    f.write(self.__code)
