import json
from typing import List, Type, Dict, Any
from layer import Layer, from_dict as layer_from_dict


class Network:
  '''
  A neural network.
  '''

  def __init__(self, name: str) -> None:
    self.__name = name
    self.__layers: List[Type[Layer]] = []

  def add_layer(self, layer: Type[Layer]) -> None:
    '''
    Add a new layer to the current network.
    '''
    self.__layers.append(layer)

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
