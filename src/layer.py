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
    self.__kernel: Dict[str, int] = d['kernel']
    self.__output: Dict[str, int] = d['output']
    return self

  def to_dict(self) -> Dict[str, Any]:
    return {
        'type': Convolution.layer_type(),
        'kernel': self.__kernel,
        'output': self.__output,
    }


class Pooling(Layer):
  '''
  Pooling layer.
  '''

  @staticmethod
  def layer_type() -> str:
    return 'pooling'

  def from_dict(self, d: Dict[str, Any]) -> Type['Layer']:
    self.__kernel: Dict[str, int] = d['kernel']
    self.__output: Dict[str, int] = d['output']
    return self

  def to_dict(self) -> Dict[str, Any]:
    return {
        'type': Pooling.layer_type(),
        'kernel': self.__kernel,
        'output': self.__output,
    }


class FullConnection(Layer):
  '''
  Full connection layer.
  '''

  @staticmethod
  def layer_type() -> str:
    return 'full_connection'

  def from_dict(self, d: Dict[str, Any]) -> Type['Layer']:
    self.__output: Dict[str, int] = d['output']
    return self

  def to_dict(self) -> Dict[str, Any]:
    return {
        'type': FullConnection.layer_type(),
        'output': self.__output,
    }


class Activation(Layer):
  '''
  Activation layer.
  '''

  @staticmethod
  def layer_type() -> str:
    return 'activation'

  def from_dict(self, d: Dict[str, Any]) -> Type['Layer']:
    self.__function: str = d['function']
    return self

  def to_dict(self) -> Dict[str, Any]:
    return {
        'type': Activation.layer_type(),
        'function': self.__function,
    }


def from_dict(d: Dict[str, Any]) -> Type['Layer']:
  '''
  Read layer from a dictionary.
  '''
  layers = {l.layer_type(): l for l in [
      Input, Convolution, Pooling, FullConnection, Activation]}
  return layers[d['type']]().from_dict(d)
