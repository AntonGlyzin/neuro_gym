from __future__ import annotations
import numpy as np
import time
from pathlib import Path
import gymnasium as gym
from typing import List, Optional
import matplotlib.pyplot as plt
from neuro_gym.environs import Environs
from neuro_gym.agent import NetworkAgent
from neuro_gym.evaluate import AlgorithmEvaluate

from settings import ROOT_DIR


class NeuroGym(object):
    """ Алгоритм нейроэволюции моделей. """
    
    def __init__(self):
        self._model: Optional[NetworkAgent] = None
        self._env: Optional[Environs] = None
        self._env_dir: Optional[Path] = None
        self.max_values = []
        self.mean_values = []
        self.min_values = []
        self.std_values = []
        self.best_values = []
        self.best_individuals: List[List[list]] = []
    
    def start(self):
        """ Запуск алгоритма. """
        print(
            """ 
                ╔══════════════════════════════════════════════╗
                ║            Алгоритм нейроэволюции            ║
                ╚══════════════════════════════════════════════╝
            """
        )
        print('Список окружающих сред: ')
        for i, val in enumerate(Environs.iter()):
            print('{}. {}.'.format(i+1, val.name))
        user_action = 0
        try:
            while True:
                print('')
                env_num = self._input_user_number(
                    'Выберите номер окружения: ',
                    1, Environs.len()
                )
                self._env = Environs.get(env_num-1)
                self._env_dir = ROOT_DIR.joinpath(self._env.id)
                if not self._env_dir.exists():
                    self._env_dir.mkdir(parents=True, exist_ok=True)
                self._load_statistics()
                self._model = NetworkAgent(self._env)
                self._model.load_modal()
                user_action = self._input_user_number(
                    '1. Активировать режим нейроэволюции.\n'
                    '2. Статистика нейроэволюция моделей.\n'
                    '3. Тестовый запуск.\n'
                    '4. Выбрать другую модель.\n'
                    '5. Моя текущая модель.\n'
                    'Ваше действие: ',
                    1, 5
                )
                if user_action == 1:
                    num_ten = self._input_user_number(
                        'Количество десятилетий от 1 до бесконечности: ',
                        1, 99999999
                    )
                    self._run_train_agent(num_ten)
                    print('\nНейроэволюция завершена.', end='\n\n')
                    print('Выберите модель для сохранения.')
                    self._plot_evolution()
                    self._select_save_model()
                elif user_action == 2:
                    self._plot_evolution()
                elif user_action == 3:
                    self._run_test_agent()
                elif user_action == 4:
                    self._plot_evolution()
                    self._select_save_model()
                elif user_action == 5:
                    self._my_model()
        except KeyboardInterrupt:
            if user_action == 1:
                print('\n\nВыберите модель для сохранения.')
                self._plot_evolution()
                self._select_save_model()
            print('Завершение работы.')
    
    def _my_model(self):
        current = self._model.get_weights_as_vector()
        num_ten = 0
        num_ind = 0
        for i, epoch in enumerate(self.best_individuals, 1):
            for j, ind in enumerate(epoch, 1):
                if np.array_equal(np.array(ind, dtype=np.float32), current):
                    num_ind = j
                    num_ten = i
        if num_ten and num_ind:
            year_ind = num_ten * 10
            print(
                f'Эта модель из {num_ten} десятилетия. '
                f'Особь под номером {num_ind}. '
                f'Возраст генов особи {year_ind} лет.'
            )
    
    def _select_save_model(self):
        """ Сохранение выбранной модели. """
        num_ten = len(self.best_individuals)
        if not num_ten:
            print('Нет обученных моделей.')
            return
        user_ten = self._input_user_number(
            'Выберите десятилетие от 1 до {}: '.format(num_ten), 1, num_ten
        )
        user_individ = self._input_user_number(
            'Выберите модель от 1 до 3: ', 1, 3
        )
        selected_individ = self.best_individuals[user_ten-1][user_individ-1]
        self._model.set_weights_from_vector(selected_individ)
        self._model.save_modal()
        print('Сохранение обученной модели.')
    
    def _save_statistics(self):
        """ Сохранение статистики. """
        np.save(str(self._env_dir / 'Max'), np.array(self.max_values))
        np.save(str(self._env_dir / 'Mean'), np.array(self.mean_values))
        np.save(str(self._env_dir / 'Min'), np.array(self.min_values))
        np.save(str(self._env_dir / 'Std'), np.array(self.std_values))
        np.save(str(self._env_dir / 'BestEvalVals'), np.array(self.best_values))
        np.save(str(self._env_dir / 'BestEvalIndivids'), np.array(self.best_individuals))
    
    def _load_statistics(self):
        """ Загрузка статистики. """
        try:
            self.max_values = np.load(str(self._env_dir / 'Max.npy')).tolist()
            self.mean_values = np.load(str(self._env_dir / 'Mean.npy')).tolist()
            self.min_values = np.load(str(self._env_dir / 'Min.npy')).tolist()
            self.std_values = np.load(str(self._env_dir / 'Std.npy')).tolist()
            self.best_values = np.load(str(self._env_dir / 'BestEvalVals.npy')).tolist()
            self.best_individuals: List[List[list]] = np.load(str(self._env_dir / 'BestEvalIndivids.npy')).tolist()
        except Exception:
            pass
    
    def _plot_evolution(self):
        """ Визуализация результата. """
        plt.figure(figsize=(12, 5))
        axes = plt.subplot(2, 1, 1)
        generations = range(len(self.max_values))
        plt.plot(generations, self.max_values, 'b-', label='Максимальное', linewidth=2, alpha=0.8)
        plt.plot(generations, self.mean_values, 'g-', label='Среднее', linewidth=2, alpha=0.8)
        plt.plot(generations, self.min_values, 'r-', label='Минимальное', linewidth=2, alpha=0.8)
        axes.axhline(color='gray', linestyle='--', alpha=0.5)
        axes.axvline(color='gray', linestyle='--', alpha=0.5)
        plt.fill_between(generations, 
                        np.array(self.mean_values) - np.array(self.std_values),
                        np.array(self.mean_values) + np.array(self.std_values),
                        alpha=0.2, color='g')
        plt.xlabel('Поколения')
        plt.ylabel('Награда')
        plt.title('Эволюция популяции')
        plt.legend()
        plt.grid(True, alpha=0.3)
        ax = plt.subplot(2, 1, 2)
        best1 = []
        best2 = []
        best3 = []
        for i in self.best_values:
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
        plt.title('Лучший результат за десятилетия')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.suptitle('Нейроэволюция моделей', fontsize=16)
        plt.tight_layout()
        plt.show(block=True)
    
    def _run_test_agent(self):
        """ Запуск агента в окружающей среде. """
        env_gym = gym.make(render_mode='human', **self._env.params)
        observation, _ = env_gym.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps_confidence = 0
        steps = 0
        while not (terminated or truncated):
            action, confidence = self._model.predict(observation, True)
            observation, reward, terminated, truncated, _ = env_gym.step(action)
            time.sleep(.03)
            total_reward += reward
            if not self._env.neuro_config.calc_confidence:
                continue
            steps_confidence += confidence
            steps += 1
        if self._env.neuro_config.calc_confidence and (total_reward > 0):
            step_confidence = steps_confidence / steps if steps else 0
            print(f"Средняя уверенность шага: {step_confidence:.1%}")
        print(f"Общая награда: {total_reward:.1f}")
        env_gym.close()
    
    def _run_train_agent(self, num_ten: int):
        """Тренировка моделей в окружение среды.

        Args:
            num_ten (int): Количество десятилетий.
        """        
        LEN_BAR = 40
        ONE_PART = LEN_BAR / num_ten
        total_sum = 0
        progress = "[" + " " * LEN_BAR + "]"
        print(f"\r{progress} 0%", end="", flush=True)
        for i in range(1, num_ten + 1):
            env_gym = gym.make(**self._env.params)
            gen_alg = AlgorithmEvaluate(self._model, env_gym, self._env)
            gen_alg.load_best_population()
            gen_alg.evaluate()
            gen_alg.save_best_individuals()
            for i in gen_alg.best_individuals():
                self._model.set_weights_from_vector(i)
                break
            self._model.save_modal()
            env_gym.close()
            
            self.max_values.extend(gen_alg.max_values.copy())
            self.mean_values.extend(gen_alg.mean_values.copy())
            self.min_values.extend(gen_alg.min_values.copy())
            self.std_values.extend(gen_alg.std_values.copy())
            best = [
                i.fitness.values[0]
                for i in gen_alg.best_individuals()
            ]
            self.best_individuals.append(gen_alg.best_individuals())
            self.best_values.extend([best])
            self._save_statistics()
            total_sum += ONE_PART
            end = int(LEN_BAR - total_sum)
            progress = "[" + "█" * int(total_sum) + " " * end + "]"
            percent = int((100/LEN_BAR)*int(total_sum))
            print(f"\r{progress} {percent}%", end="", flush=True)

    def _input_user_number(self, text: str, min: int, max: int) -> int:
        """Получение числа от пользователя в выбранном пределе. 

        Args:
            text (str): Название вывода.
            min (int): Минимальное число.
            max (int): Максимальное число.

        Returns:
            int: Пользовательский выбор.
        """        
        env_num = -1
        while True:
            env_num = input(text)
            try:
                env_num = int(env_num)
                if min <= env_num <= max:
                    return env_num
            except Exception:
                continue