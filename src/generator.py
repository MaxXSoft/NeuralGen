from typing import Text, Type, TextIO
from layer import Layer
from network import Network


class Generator:
  '''
  Interface of neural network generator.
  '''

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


class CppGenerator(Generator):
  '''
  Generate C++ code for a nerual network.
  '''

  def __init__(self) -> None:
    # TODO
    pass

  def generate(self, network: Network) -> None:
    for layer in network.layers:
      # TODO
      pass

  def dump(self, f: TextIO) -> None:
    # TODO
    pass
