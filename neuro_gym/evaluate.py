from __future__ import annotations
import numpy as np
from deap import base, algorithms
from deap import creator
from deap import tools
import gymnasium as gym
from typing import List, Tuple

from neuro_gym.environs import Environs
from neuro_gym.agent import NetworkAgent

from settings import ROOT_DIR


class AlgorithmEvaluate:
    """ Генетический алгоритм для эволюции модели. """
    
    POPULATION_SIZE = 30
    CROSSOVER_RATE = 0.9
    MUTATION_RATE = 0.1
    MAX_GENERATIONS = 10
    
    LAMBDA = 60
    
    HALL_OF_FAME_SIZE = 3
    
    EVALUATION_TRIALS = 3
    
    LOW = -1.0
    UP = 1.0
    ETA = 20

    def __init__(self, model: NetworkAgent, environ: gym.Env, individ: Environs):
        """
        
        Args:
            model (NetworkAgent): Нейронная модель.
            environ (gym.Env): Сцена.
            individ (Environs): Окружение и настройки.
        """        
        self._individ = individ
        self._model = model
        self._environ = environ
        self._chromosome_length = self._model.count_weights
        self._hof = tools.HallOfFame(self.HALL_OF_FAME_SIZE)
        self._toolbox = base.Toolbox()
        self._path_best_individs = ROOT_DIR / self._individ.id / 'BestIndivids.npy'
        
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
        _, logbook = algorithms.eaMuPlusLambda(
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
    
    def best_individuals(self) -> List[List[float]]:
        """Получение лучших особей.

        Returns:
            List[List[float]]: Особи.
        """        
        return self._hof.items.copy()
    
    def save_best_individuals(self):
        """ Сохранение лучших особей. """
        np.save(str(self._path_best_individs), np.array(self._hof.items))
    
    def load_best_population(self):
        """ Загружает лучших особей в популяцию. """
        items = []
        if self._path_best_individs.exists():
            items = np.load(str(self._path_best_individs)).tolist()
        for i, _ in enumerate(items):
            ind = self._toolbox.individual_creator()
            ind[:] = items[i][:]
            self._population[i] = ind
    
    def _evaluate(self, individual: List[float]) -> Tuple[float]:
        """Фитнес функция.

        Args:
            individual (List[float]): Выбранная особь.

        Returns:
            Tuple[float]: Результат взаимодействия со средой.
        """        
        all_rewards = []
        all_confidences = 0
        self._model.set_weights_from_vector(individual)
        for _ in range(self.EVALUATION_TRIALS):
            observation, _ = self._environ.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            episode_confidence = 0
            steps = 0
            while not (terminated or truncated):
                action, confidence = self._model.predict(observation, True)
                observation, reward, terminated, truncated, _ = self._environ.step(action)
                episode_reward += reward
                if not self._individ.neuro_config.calc_confidence:
                    continue
                episode_confidence += confidence
                steps += 1
            step_confidence = (
                episode_confidence / steps 
                if steps else 0
            )
            all_confidences += step_confidence
            all_rewards.append(episode_reward)
        all_confidences /= self.EVALUATION_TRIALS
        mean_reward = np.mean(all_rewards)
        weight = 30 if mean_reward > 0 else 10
        return mean_reward + all_confidences*weight,
