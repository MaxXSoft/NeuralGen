from typing import Dict, Any, Type


class Layer:
  '''
  Interface of neural network layers.
  '''

  @staticmethod
  def layer_type() -> str:
    '''
    Get the type of the current layer.
    '''
    raise NotImplementedError

  def from_dict(self, d: Dict[str, Any]) -> Type['Layer']:
    '''
    Read layer from a dictionary.
    '''
    raise NotImplementedError

  def to_dict(self) -> Dict[str, Any]:
    '''
    Convert the current layer to a dictionary.
    '''
    raise NotImplementedError

  def __getitem__(self, key: str) -> Any:
    return self.to_dict()[key]


class Input(Layer):
  '''
  Input layer.
  '''

  @staticmethod
  def layer_type() -> str:
    return 'input'

  def from_dict(self, d: Dict[str, Any]) -> Type['Layer']:
    self.__width: int = d['width']
    self.__height: int = d['height']
    self.__depth: int = d['depth']
    return self

  def to_dict(self) -> Dict[str, Any]:
    return {
        'type': Input.layer_type(),
        'width': self.__width,
        'height': self.__height,
        'depth': self.__depth,
    }


class Convolution(Layer):
  '''
  Convolution layer.
  '''

  @staticmethod
  def layer_type() -> str:
    return 'convolution'

  def from_dict(self, d: Dict[str, Any]) -> Type['Layer']:
    self.__padding: str = d['padding']
    self.__stride: int = d['stride']
    self.__kernel: Dict[str, int] = d['kernel']
    self.__output: Dict[str, int] = d['output']
    self.__activation: str = d['activation']
    return self

  def to_dict(self) -> Dict[str, Any]:
    return {
        'type': Convolution.layer_type(),
        'padding': self.__padding,
        'stride': self.__stride,
        'kernel': self.__kernel,
        'output': self.__output,
        'activation': self.__activation,
    }


class Pooling(Layer):
  '''
  Pooling layer.
  '''

  @staticmethod
  def layer_type() -> str:
    return 'pooling'

  def from_dict(self, d: Dict[str, Any]) -> Type['Layer']:
    self.__function: str = d['function']
    self.__padding: str = d['padding']
    self.__stride: int = d['stride']
    self.__kernel: Dict[str, int] = d['kernel']
    self.__output: Dict[str, int] = d['output']
    self.__activation: str = d['activation']
    return self

  def to_dict(self) -> Dict[str, Any]:
    return {
        'type': Pooling.layer_type(),
        'function': self.__function,
        'padding': self.__padding,
        'stride': self.__stride,
        'kernel': self.__kernel,
        'output': self.__output,
        'activation': self.__activation,
    }


class FullConnection(Layer):
  '''
  Fully connection layer.
  '''

  @staticmethod
  def layer_type() -> str:
    return 'full_connection'

  def from_dict(self, d: Dict[str, Any]) -> Type['Layer']:
    self.__output_size: int = d['outputSize']
    self.__activation: str = d['activation']
    return self

  def to_dict(self) -> Dict[str, Any]:
    return {
        'type': FullConnection.layer_type(),
        'output_size': self.__output_size,
        'activation': self.__activation,
    }


def from_dict(d: Dict[str, Any]) -> Type['Layer']:
  '''
  Read layer from a dictionary.
  '''
  layers = {l.layer_type(): l for l in [
      Input, Convolution, Pooling, FullConnection]}
  return layers[d['type']]().from_dict(d)
