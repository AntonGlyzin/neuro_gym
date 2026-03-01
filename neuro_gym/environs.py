from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Union, Dict, Any, Callable


class Complexity(IntEnum):
    """ Сложность окружающей среды. """
    
    LOW = 2
    MEDIUM = 4
    HIGH = 8


@dataclass
class NeuroConfig:
    """ Настройки для нейронной сети. """
    
    input_size: int
    output_size: int
    complexity: Complexity
    calc_confidence: bool
    update_vector: Callable[[Union[np.ndarray, torch.Tensor]], Any]


class Environs(object):
    """ Окружающая среда. """
    
    elements: Dict[str, Environs] = {}
    
    def __init__(self, id: str, name: str, params: dict, neuro_config: NeuroConfig):
        self.id = id
        self.name = name
        self.params = params
        self.neuro_config = neuro_config
        self.elements[id] = self
    
    @classmethod
    def get(cls, key: Union[str, int]) -> Environs:
        """Получает окружение по индексу.

        Args:
            key (Union[str, int]): Индекс.

        Returns:
            Environs: Окружение.
        """        
        if key is str:
            return cls.elements.get(key)
        return list(cls.elements.values())[key]
    
    @classmethod
    def len(cls) -> int:
        """ Количество окружений. """        
        return len(cls.elements)
    
    @classmethod
    def iter(cls):
        return iter(cls.elements.values())