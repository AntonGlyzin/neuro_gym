from typing import Dict, Union
import torch
import numpy as np
from neuro_gym.environ import Environ, Complexity


class LunarLander(Environ):
    
    @property
    def id(self) -> str:
        return 'BipedalWalker-v3'
    
    @property
    def name(self) -> str:
        return 'Двуногий шагающий робот'
    
    @property
    def params(self) -> Dict:
        return {
            'id': self.id,
            'hardcore': True
        }
    
    @property
    def complexity(self) -> int:
        return Complexity.MEDIUM
    
    @property
    def number_input_neurons(self) -> int: 
        return 24
    
    @property
    def number_output_neurons(self) -> int: 
        return 4
    
    @property
    def calc_confidence(self) -> bool: 
        return False
    
    def update_vector(self, output_vector: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return torch.tanh(output_vector).numpy()