import torch
import torch.nn as nn
import numpy as np
import os
import time
import torch.nn.functional as F
from deap import base, algorithms
from deap import creator
from deap import tools

import gymnasium as gym
from typing import Union, Tuple, List

import random
import matplotlib.pyplot as plt


RANDOM_SEED = 33
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class NetworkAgent(nn.Module):
    """ Нейронная сеть агента. """
    
    INPUT_SIZE = 8
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 4
    
    FILENAME = 'agent'
    
    def __init__(self):
        super(NetworkAgent, self).__init__()
        self._network = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, self.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.HIDDEN_SIZE // 2, self.OUTPUT_SIZE)
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
                return_probabilities: bool = False) -> Union[int, Tuple[int, np.ndarray]]:
        """ Предсказание действия. """
        tensor_x = self._prepare_input(x)
        with torch.no_grad():
            outputs = self._network(tensor_x)
            if return_probabilities:
                probabilities = F.softmax(outputs, dim=-1)
                action = int(torch.argmax(outputs, dim=-1))
                return action, probabilities[0].numpy()
            else:
                return int(torch.argmax(outputs, dim=-1))
    
    def _prepare_input(self, x: Union[np.ndarray, list, torch.Tensor], 
                    ensure_batch: bool = False) -> torch.Tensor:
        """ Подготовка входных данных для нейросети. """
        if isinstance(x, torch.Tensor):
            tensor_x = x.float()
        elif isinstance(x, np.ndarray):
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            tensor_x = torch.from_numpy(x)
        else:  # list
            tensor_x = torch.tensor(x, dtype=torch.float32)
        # Добавляем batch dimension если нужно
        if ensure_batch and tensor_x.dim() == 1:
            tensor_x = tensor_x.unsqueeze(0)
        return tensor_x
    
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
        torch.save(self.state_dict(), self.FILENAME)
    
    def load_modal(self):
        """ Загрузить модель из файла. """
        try:
            self.load_state_dict(torch.load(self.FILENAME))
        except FileNotFoundError:
            pass
        self.eval()


class AlgorithmEvaluate:
    
    POPULATION_SIZE = 30
    CROSSOVER_RATE = 0.9
    MUTATION_RATE = 0.1
    MAX_GENERATIONS = 10
    
    LAMBDA = 60
    
    HALL_OF_FAME_SIZE = 4
    
    EVALUATION_TRIALS = 3
    
    LOW = -1.0
    UP = 1.0
    ETA = 15

    def __init__(self, model: NetworkAgent, environ: gym.Env):
        self._model = model
        self._environ = environ
        self._chromosome_length = self._model.count_weights
        self._hof = tools.HallOfFame(self.HALL_OF_FAME_SIZE)
        self._toolbox = base.Toolbox()
        
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self._toolbox.register("random_weight", np.random.uniform, -1.0, 1.0)
        self._toolbox.register("individual_creator", tools.initRepeat, creator.Individual, 
                            self._toolbox.random_weight, self._model.count_weights)
        self._toolbox.register("population_creator", tools.initRepeat, list, self._toolbox.individual_creator)
        
        self._population = self._toolbox.population_creator(n=self.POPULATION_SIZE)
        
        self._toolbox.register("evaluate", self._evaluate)
        self._toolbox.register("select", tools.selTournament, tournsize=3)
        self._toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.LOW, 
                        up=self.UP, eta=self.ETA)
        self._toolbox.register("mutate", tools.mutPolynomialBounded, low=self.LOW, 
                        up=self.UP, eta=self.ETA, indpb=1.0/model.count_weights)
        
        self._stats = tools.Statistics(lambda ind: ind.fitness.values)
        self._stats.register("max", np.max)
        self._stats.register("avg", np.mean)
        self._stats.register("min", np.min)
        self._stats.register("std", np.std)
        
        self.max_values = None
        self.mean_values = None
        self.min_values = None
        self.std_values = None
    
    def evaluate(self):
        """ Запуск эволюции. """
        population, logbook = algorithms.eaMuPlusLambda(
                self._population, 
                self._toolbox,
                mu=self.POPULATION_SIZE,
                cxpb=self.CROSSOVER_RATE,
                mutpb=self.MUTATION_RATE,
                ngen=self.MAX_GENERATIONS,
                lambda_= self.LAMBDA,
                halloffame=self._hof,
                stats=self._stats,
                verbose=False
            )
        self.max_values = logbook.select("max")
        self.mean_values = logbook.select("avg")
        self.min_values = logbook.select("min")
        self.std_values = logbook.select("std")
        return self._hof.items[0]
    
    def _evaluate(self, individual: List[float]):
        """ Фитнес функция. """
        total_rewards = []
        self._model.set_weights_from_vector(individual)
        for _ in range(self.EVALUATION_TRIALS):
            observation, info = self._environ.reset()
            terminated = False
            truncated = False
            total_reward = 0
            while not (terminated or truncated):
                action = self._model.predict(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            
            x, y, x_vel, y_vel, angle, ang_vel, left_leg, right_leg = observation
            custom_bonus = 0
            if abs(y_vel) < 0.5:  # Маленькая вертикальная скорость
                custom_bonus += 50
            # Бонус за центр площадки
            if abs(x) < 0.1:
                custom_bonus += 25
            # Штраф за большой угол
            if abs(angle) > 0.5:
                custom_bonus -= 30
            total_rewards.append(total_reward + custom_bonus)
        return np.mean(total_rewards),
    
    def best_individuals(self) -> List[List[float]]:
        """Получение лучших особей. """
        return self._hof.items.copy()
    
    def save_best_individuals(self):
        """ Сохранение лучших особей. """
        np.save('best_individ.npy', np.array(self._hof.items))
    
    def load_best_population(self):
        """ Загружает лучших особей в популяцию. """
        items = []
        try:
            items = np.load('best_individ.npy').tolist()
        except FileNotFoundError:
            pass
        for i, _ in enumerate(items):
            ind = self._toolbox.individual_creator()
            ind[:] = items[i][:]
            self._population[i] = ind


def plot_evolution(max_values: list, mean_values: list, min_values: list, 
                std_values: list, best_values: List[list]):
        """ Визуализация результата. """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        generations = range(len(max_values))
        plt.plot(generations, max_values, 'b-', label='Max', linewidth=2)
        plt.plot(generations, mean_values, 'g-', label='Mean', linewidth=2)
        plt.plot(generations, min_values, 'r-', label='Min', linewidth=2)
        plt.fill_between(generations, 
                        np.array(mean_values) - np.array(std_values),
                        np.array(mean_values) + np.array(std_values),
                        alpha=0.2, color='g')
        plt.xlabel('Поколение')
        plt.ylabel('Награда')
        plt.title('Эволюция популяции')
        plt.legend()
        plt.grid(True)
        
        ax = plt.subplot(1, 2, 2)
        best1 = []
        best2 = []
        best3 = []
        for i in best_values:
            best1.append(i[0])
            best2.append(i[1])
            best3.append(i[2])
        x = np.arange(len(best1))
        width = 0.3
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(1, len(best1)+1)])
        ax.bar(x - width, best1, width=width, label='Лучший 1', alpha=0.8)
        ax.bar(x, best2, width=width, label='Лучший 2', alpha=0.8)
        ax.bar(x + width, best3, width=width, label='Лучший 3', alpha=0.8)
        plt.xlabel('Количество десятилетий')
        plt.ylabel('Максимальная награда за десятилетие')
        plt.title('Лучший результат за десятилетие')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show(block=True)


def input_user_number(text: str, min: int, max: int) -> int:
    env_num = -1
    while True:
        env_num = input(text)
        try:
            env_num = int(env_num)
            if min <= env_num <= max:
                return env_num
        except:
            continue

if __name__ == '__main__':
    # random.uniform(0, 20.0)
    
    environs = {
        'Лунный посадочный модуль': {
            'id': 'LunarLander-v3',
            'continuous': False, 
            'gravity': -10.0, # 0 до -12.0
            'enable_wind': True,  # случайным образом в диапазоне от -9999 до 9999
            'wind_power': 20.0,  #  от 0 до 20.0
            'turbulence_power': 2.0 #  от 0 до 2.0
        }
    }
    
    max_values = []
    mean_values = []
    min_values = []
    std_values = []
    
    best_values = []
    best_individuals: List[List[list]] = []
    
    model = NetworkAgent()
    model.load_modal()
    
    print(
        """ 
            ╔══════════════════════════════════════════════╗
            ║               Нейроэволюция                  ║
            ╚══════════════════════════════════════════════╝
        """
    )
    print('Список окружающей среды: ')
    for i, val in enumerate(environs.keys()):
        print('{}.{}'.format(i+1, val))
    print('')
    
    env_num = input_user_number(
        'Выберите номер окружения: ',
        1, len(environs)
    )
    params = list(environs.values())[env_num-1]
    
    act_ev = input_user_number(
        'Активировать режим нейроэволюции? 0 - Нет, 1 - Да: ',
        0, 1
    )
    
    if act_ev:
        num_ten = input_user_number(
            'Количество десятилетий от 1 до бесконечности: ',
            1, 99999999
        )
        LEN_BAR = 40
        ONE_PART = LEN_BAR / num_ten
        total_sum = 0
        progress = "[" + " " * LEN_BAR + "]"
        print(f"\r{progress} 0%", end="", flush=True)
        for i in range(1, num_ten + 1):
            env = gym.make(**params)
            gen_alg = AlgorithmEvaluate(model, env)
            gen_alg.load_best_population()
            best_individual = gen_alg.evaluate()
            gen_alg.save_best_individuals()
            for i in gen_alg.best_individuals():
                model.set_weights_from_vector(i)
                break
            model.save_modal()
            env.close()
            
            max_values.extend(gen_alg.max_values.copy())
            mean_values.extend(gen_alg.mean_values.copy())
            min_values.extend(gen_alg.min_values.copy())
            std_values.extend(gen_alg.std_values.copy())
            best = [
                i.fitness.values[0]
                for i in gen_alg.best_individuals()
            ]
            best_individuals.append(gen_alg.best_individuals())
            best_values.extend([best])
            
            total_sum += ONE_PART
            end = int(LEN_BAR - total_sum)
            progress = "[" + "█" * int(total_sum) + " " * end + "]"
            percent = int((100/LEN_BAR)*int(total_sum))
            print(f"\r{progress} {percent}%", end="", flush=True)
            
        plot_evolution(max_values, mean_values, 
                    min_values, std_values, best_values)
        print('\nНейроэволюция завершена.')
        print('')
        print('Сохранение обученной модели.')
        user_ten = input_user_number(
            'Выберите десятилетие от 1 до {}: '.format(num_ten),
            1, num_ten
        )
        user_individ = input_user_number(
            'Выберите модель от 1 до 3: ',
            1, 3
        )
        selected_individ = best_individuals[user_ten-1][user_individ-1]
        model.set_weights_from_vector(selected_individ)
        model.save_modal()
    
    env = gym.make(render_mode='human', **params)
    observation, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    while not (terminated or truncated):
        action = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(.03)
        total_reward += reward
    print(f"{observation}, {terminated}, {total_reward}")
    env.close()
    
    