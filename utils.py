import os
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """
    Фиксирует все генераторы псевдослучайных чисел для 100% воспроизводимости.
    Вызывать в ПЕРВОЙ ячейке каждого Jupyter Notebook'а.
    """
    # 1. Фиксация на уровне Python
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    
    # 2. Фиксация на уровне NumPy
    np.random.seed(seed)
    
    # 3. Фиксация на уровне TensorFlow / Keras (для GRU, LSTM, Hybrid)
    tf.random.set_seed(seed)
    
    # Отключаем возможную многопоточную недетерминированность TF (опционально, но надежно)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    print(f" Random seed зафиксирован: {seed}")

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Рассчитывает и выводит основные метрики регрессии.
    Возвращает словарь с метриками для сохранения.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- Метрики {model_name} ---")
    print(f"MAE:  {mae:.2f} кВт")
    print(f"RMSE: {rmse:.2f} кВт")
    print(f"R2:   {r2:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def plot_predictions(y_true, y_pred, dates, model_name="Model", days=3):
    """
    Строит красивый график сравнения предсказаний и реальности на заданное количество дней.
    """
    # Берем срез данных (24 часа * кол-во дней)
    points = 24 * days
    
    plt.figure(figsize=(15, 6))
    plt.plot(dates[:points], y_true[:points], label='Фактическое потребление', color='blue', linewidth=2)
    plt.plot(dates[:points], y_pred[:points], label=f'Прогноз ({model_name})', color='orange', linestyle='--', linewidth=2)
    
    plt.title(f"Сравнение прогноза и факта: {model_name} (Первые {days} дня)", fontsize=14)
    plt.xlabel("Время")
    plt.ylabel("Потребление (кВт)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
