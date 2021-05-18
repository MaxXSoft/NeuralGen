from typing import List, Type, Dict, Any
from layer import Layer, from_dict as layer_from_dict


'''
API version
'''
__API_VERSION = '0.0.1'


class Network:
  '''
  Represents a neural network.
  '''

  def __init__(self, name: str) -> None:
    self.__name = name
    self.__layers: List[Type[Layer]] = []

  def add_layer(self, layer: Type[Layer]) -> None:
    '''
    Add a new layer to the current network.
    '''
    self.__layers.append(layer)

  def to_dict(self) -> Dict[str, Any]:
    '''
    Convert the current network to a dictionary.
    '''
    return {
        'apiVersion': __API_VERSION,
        'name': self.__name,
        'layers': [l.to_dict() for l in self.__layers]
    }

  @property
  def name(self) -> str:
    '''
    Get the name of the current network.
    '''
    return self.__name

  @property
  def layers(self) -> List[Type[Layer]]:
    '''
    Get the list of all layers of the current network.
    '''
    return self.__layers


def from_dict(d: Dict[str, Any]) -> Network:
  '''
  Read network from a dictionary.
  '''
  network = Network(d['name'])
  for l in d['layers']:
    network.add_layer(layer_from_dict(l))
  return network
