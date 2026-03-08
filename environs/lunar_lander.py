from typing import Dict, Union
import torch
import numpy as np
from neuro_gym.environ import Environ, Complexity


class LunarLander(Environ):
    
    @property
    def id(self) -> str:
        return 'LunarLander-v3'
    
    @property
    def name(self) -> str:
        return 'Лунный посадочный модуль'
    
    @property
    def params(self) -> Dict:
        return {
            'id': self.id,
            'continuous': False, 
            'gravity': -10.0, # 0 до -12.0
            'enable_wind': True,  # случайным образом в диапазоне от -9999 до 9999
            'wind_power': 20.0,  #  от 0 до 20.0
            'turbulence_power': 2.0 #  от 0 до 2.0
        }
    
    @property
    def complexity(self) -> int:
        return Complexity.LOW
    
    @property
    def number_input_neurons(self) -> int: 
        return 8
    
    @property
    def number_output_neurons(self) -> int: 
        return 4
    
    @property
    def calc_confidence(self) -> bool: 
        return True
    
    def update_vector(self, output_vector: Union[np.ndarray, torch.Tensor]) -> int:
        return torch.argmax(output_vector, dim=-1).item()