from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Union, Tuple, List, Any

from neuro_gym.environs import Environs
from settings import ROOT_DIR


class NetworkAgent(nn.Module):
    """ Нейронная сеть агента. """
    
    def __init__(self, environ: Environs):
        super(NetworkAgent, self).__init__()
        self._environ = environ
        self._neuro_config = environ.neuro_config
        hidden_size = self._hidden_size()
        self._path_agent = ROOT_DIR / self._environ.id / 'Agent'
        self._network = nn.Sequential(
            nn.Linear(self._neuro_config.input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, self._neuro_config.output_size)
        )
        self._init_weights()
        self.eval()
    
    @property
    def count_weights(self) -> int:
        """ Общее количество весов в сетию. """
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self):
        """ Инициализация весов с использованием Xavier. """
        for module in self._network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def predict(self, x: Union[np.ndarray, list, torch.Tensor], 
                confidence: bool = False,
                ensure_batch: bool = False) -> Union[Any, Tuple]:
        """ Предсказание действия. """
        tensor_x = self._prepare_input(x, ensure_batch)
        with torch.no_grad():
            outputs = self._network(tensor_x)
        if outputs.dim() == 2:
            outputs = tensor_x.squeeze()
        res = self._neuro_config.update_vector(outputs)
        if not confidence:
            return res
        return res, F.softmax(outputs, dim=-1).max()
    
    def get_weights_as_vector(self) -> np.ndarray:
        """ Получение весов модели. """
        weights_vector = []
        for _, param in self.named_parameters():
            weights_vector.extend(param.data.numpy().flatten())
        return np.array(weights_vector, dtype=np.float32)
    
    def set_weights_from_vector(self, chromosome: List[float]) -> None:
        """ Установка весов из плоского вектора. """
        if len(chromosome) != self.count_weights:
            raise ValueError(f"Ожидается {self.count_weights} весов, получено {len(chromosome)}")
        start_idx = 0
        with torch.no_grad():
            for param in self.parameters():
                num_elements = param.numel()
                end_idx = start_idx + num_elements
                layer_weights = chromosome[start_idx:end_idx]
                weights_tensor = torch.tensor(layer_weights, dtype=param.dtype).reshape(param.shape)
                param.copy_(weights_tensor)
                start_idx = end_idx
    
    def save_modal(self):
        """ Сохранение модели в файл. """
        torch.save(self.state_dict(), str(self._path_agent))
    
    def load_modal(self):
        """ Загрузить модель из файла. """
        if self._path_agent.exists():
            self.load_state_dict(torch.load(str(self._path_agent)))
        self.eval()
    
    def _prepare_input(self, x: Union[np.ndarray, list, torch.Tensor], 
                    ensure_batch: bool = False) -> torch.Tensor:
        """ Подготовка входных данных для нейросети. """
        if isinstance(x, torch.Tensor):
            tensor_x = x.float()
        elif isinstance(x, np.ndarray):
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            tensor_x = torch.from_numpy(x)
        else:
            tensor_x = torch.tensor(x, dtype=torch.float32)
        if ensure_batch and tensor_x.dim() == 1:
            tensor_x = tensor_x.unsqueeze(0)
        return tensor_x
    
    def _hidden_size(self) -> int:
        """ Размера скрытого слоя. """
        size = self._neuro_config.complexity * self._neuro_config.input_size * 2
        powers = [32, 64, 128, 256, 512]
        return min(powers, key=lambda x: abs(x - size))