import pandas as pd
import numpy as np

class GradientDescentMse:
    """
    Базовый класс для реализации градиентного спуска в задаче линейной МНК регрессии 
    """

    def __init__(self, samples: pd.DataFrame, targets: pd.DataFrame,
                 learning_rate: float = 1e-3, threshold = 1e-6, copy: bool = True):
        """
        self.samples - матрица признаков
        self.targets - вектор таргетов
        self.beta - вектор с изначальными весами модели == коэффициентами бета (состоит из единиц)
        self.learning_rate - параметр *learning_rate* для корректировки нормы градиента
        self.threshold - величина, меньше которой изменение в loss-функции означает остановку градиентного спуска
        iteration_loss_dict - словарь, который будет хранить номер итерации и соответствующую MSE
        copy: копирование матрицы признаков или создание изменения in-place
        """
        
        if copy == True:
            self.samples = samples.copy()
        elif copy == False:
            self.samples = samples
        self.targets = targets
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.iteration_loss_dict = {}
        self.beta = np.ones(self.samples.shape[1])


    def add_constant_feature(self):
        """
        Метод для создания константной фичи в матрице объектов samples
        Метод создает колонку с константным признаком (interсept) в матрице признаков.
        Hint: так как количество признаков увеличилось на одну, не забудьте дополнить вектор с изначальными весами модели!
        """

        self.samples["constant"] = 1
        self.beta = np.ones(self.samples.shape[1])       
        

    def calculate_mse_loss(self) -> float:
        """
        Метод для расчета среднеквадратической ошибки
        
        :return: среднеквадратическая ошибка при текущих весах модели : float
        """

        self.prediction = np.dot(self.samples, self.beta)
        mse = ((self.prediction - self.targets)**2).mean()
        return mse


    def calculate_gradient(self) -> np.ndarray:
        """
        Метод для вычисления вектора-градиента
        Метод возвращает вектор-градиент, содержащий производные по каждому признаку.
        Сначала матрица признаков скалярно перемножается на вектор self.beta, и из каждой колонки
        полученной матрицы вычитается вектор таргетов. Затем полученная матрица скалярно умножается на матрицу признаков.
        Наконец, итоговая матрица умножается на 2 и усредняется по каждому признаку.
        
        :return: вектор-градиент, т.е. массив, содержащий соответствующее количество производных по каждой переменной : np.ndarray
        """
        
        self.scalar_value = np.dot(self.samples,self.beta)
        self.minus_target = self.scalar_value - self.targets
        self.scalar_on_samples = np.dot(self.minus_target,self.samples)
        self.gradient = self.scalar_on_samples/self.samples.shape[0] * 2  
        return self.gradient

    
    def iteration(self):
        """
        Обновляем веса модели в соответствии с текущим вектором-градиентом
        """
        
        self.beta = self.beta - self.learning_rate * self.calculate_gradient()
         

    def learn(self):
        """
        Итеративное обучение весов модели до срабатывания критерия останова
        Запись mse и номера итерации в iteration_loss_dict
                
        Описание алгоритма работы для изменения функции потерь:
            Фиксируем текущие mse -> previous_mse
            Делаем шаг градиентного спуска
            Записываем новые mse -> next_mse
            Пока |(previous_mse) - (next_mse)| < threshold:
                Повторяем первые 3 шага
        """
        
        self.previous_mse = self.calculate_mse_loss()
        self.iteration()
        self.next_mse = self.calculate_mse_loss()

        while abs(self.previous_mse - self.next_mse) > self.threshold:        
            
            self.previous_mse = self.calculate_mse_loss()
            self.iteration()
            self.next_mse = self.calculate_mse_loss()
            self.iteration_loss_dict.update({len(self.iteration_loss_dict)+1 : self.previous_mse})
            