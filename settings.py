from pathlib import Path
from neuro_gym.environs import Environs, Complexity, torch, NeuroConfig

ROOT_DIR = Path(__file__).parent

Environs(
    id='LunarLander-v3',
    name='Лунный посадочный модуль',
    params={
        'id': 'LunarLander-v3',
        'continuous': False, 
        'gravity': -10.0, # 0 до -12.0
        'enable_wind': True,  # случайным образом в диапазоне от -9999 до 9999
        'wind_power': 20.0,  #  от 0 до 20.0
        'turbulence_power': 2.0 #  от 0 до 2.0
    },
    neuro_config=NeuroConfig(
        input_size=8, 
        output_size=4, 
        complexity=Complexity.LOW,
        calc_confidence=True,
        update_vector=lambda y: torch.argmax(y, dim=-1).item()
    )
)

Environs(
    id='BipedalWalker-v3',
    name='Двуногий шагающий робот',
    params={
        'id': 'BipedalWalker-v3',
        'hardcore': True
    },
    neuro_config=NeuroConfig(
        input_size=24, 
        output_size=4, 
        complexity=Complexity.MEDIUM, 
        calc_confidence=False,
        update_vector=lambda y: torch.tanh(y).numpy()
    )
)