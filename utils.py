import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from tensorflow.keras import layers, models, regularizers


def set_seed(seed=42):
    """Фиксирует все сиды для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    print(f"Сид установлен на {seed} для Python, NumPy, TensorFlow и PYTHONHASHSEED.")




@tf.keras.utils.register_keras_serializable()
def asymmetric_profit_loss(y_true, y_pred):
    """
    Кастомная функция потерь (MSE-based).
    Штрафует сильнее за недопрогноз (когда факт > прогноза).
    Заставляет модель предсказывать пики с запасом.
    Улучшенная версия для повышения R2 и экономики.
    """
    error = y_true - y_pred
    # Оптимальные веса для баланса R2 и прибыли:
    penalty_under = 3.0 
    penalty_over = 1.0
    
    loss = tf.where(error > 0, 
                    tf.square(error) * penalty_under, 
                    tf.square(error) * penalty_over)
    
    return tf.reduce_mean(loss)

def evaluate_and_plot_predictions(y_test_real, y_pred_real, dates_test, model_name="Модель", n_hours=168, save_path=None):
    """
    Рассчитывает метрики и строит непрерывный график  
    на заданное число часов (n_hours) с выделением недопрогноза.
    """
    mae_overall = mean_absolute_error(y_test_real, y_pred_real)
    rmse_overall = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    epsilon = 1e-10 
    mape_overall = np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + epsilon))) * 100
    r2_overall = r2_score(y_test_real, y_pred_real)

    print(f"{f'Итоговые метрики: {model_name}':^55}")
    print(f"MAE:      {mae_overall:>10.2f} Вт")
    print(f"RMSE:     {rmse_overall:>10.2f} Вт")
    print(f"MAPE:     {mape_overall:>10.2f} %")
    print(f"R2:       {r2_overall:>10.4f}")

    step_idx = 0 
    plt.figure(figsize=(18, 6))
    plt.plot(dates_test[:n_hours], y_test_real[:n_hours, step_idx], 
             label='Фактическое потребление', color='black', linewidth=2, alpha=0.7)
    plt.plot(dates_test[:n_hours], y_pred_real[:n_hours, step_idx], 
             label=f'Прогноз {model_name} ', color='darkorange', linewidth=2, linestyle='--')

    plt.fill_between(dates_test[:n_hours], y_test_real[:n_hours, step_idx], y_pred_real[:n_hours, step_idx], 
                     where=(y_test_real[:n_hours, step_idx] > y_pred_real[:n_hours, step_idx]),
                     color='red', alpha=0.15, label='Недопрогноз ')

    plt.title(f'Сравнение прогноза {model_name} и факта на тестовой неделе', fontsize=14)
    plt.xlabel('Дата и время', fontsize=12)
    plt.ylabel('Потребление (Вт)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    
    return {'mae': mae_overall, 'rmse': rmse_overall, 'mape': mape_overall, 'r2': r2_overall}

def plot_training_history_with_lr(history, start_epoch=5, save_path=None):
    """
    Визуализация истории обучения (Loss и MAE) строго начиная с заданной эпохи.
    Отмечает вертикальными линиями моменты снижения Learning Rate.
    """
    hist = history.history
    loss = hist.get('loss', [])
    val_loss = hist.get('val_loss', [])
    mae = hist.get('mae', [])
    val_mae = hist.get('val_mae', [])
    
    lrs = hist.get('learning_rate') or hist.get('lr', [])
    
    if len(loss) < start_epoch:
        start_epoch = 1 
        
    s = start_epoch - 1
    epochs = range(start_epoch, len(loss) + 1)
    
    # Находим эпохи, где LR уменьшился по сравнению с предыдущей
    lr_drop_epochs = []
    if len(lrs) > 0:
        for i in range(1, len(lrs)):
            if lrs[i] < lrs[i-1]:
                lr_drop_epochs.append(i + 1)
                
    # Оставляем только те отметки, которые входят в диапазон графика
    visible_drops = [e for e in lr_drop_epochs if e >= start_epoch]

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    # LOSS 
    ax[0].plot(epochs, loss[s:], label='Train Loss', color='steelblue', linewidth=2)
    if len(val_loss) > 0:
        ax[0].plot(epochs, val_loss[s:], label='Val Loss', color='darkorange', linewidth=2, linestyle='--')
    
    # Отметки снижения LR
    for drop_e in visible_drops:
        ax[0].axvline(x=drop_e, color='grey', linestyle=':', alpha=0.8)
    if visible_drops:
        ax[0].plot([], [], color='grey', linestyle=':', label='LR Drop (x0.75)')

    ax[0].set_title(f'История Loss (с {start_epoch} эпохи)', fontsize=13)
    ax[0].set_xlabel('Эпоха', fontsize=11)
    ax[0].set_ylabel('Asymmetric Loss', fontsize=11)
    ax[0].set_xlim(left=start_epoch) 
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # MAE
    if len(mae) > 0:
        ax[1].plot(epochs, mae[s:], label='Train MAE', color='forestgreen', linewidth=2)
    if len(val_mae) > 0:
        ax[1].plot(epochs, val_mae[s:], label='Val MAE', color='crimson', linewidth=2, linestyle='--')
    
    # Отметки снижения LR
    for drop_e in visible_drops:
        ax[1].axvline(x=drop_e, color='grey', linestyle=':', alpha=0.8)
    if visible_drops:
        ax[1].plot([], [], color='grey', linestyle=':', label='LR Drop (x0.75)')

    ax[1].set_title(f'История MAE (с {start_epoch} эпохи)', fontsize=13)
    ax[1].set_xlabel('Эпоха', fontsize=11)
    ax[1].set_ylabel('MAE', fontsize=11)
    ax[1].set_xlim(left=start_epoch)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')


def create_sequences(X_data, y_data, dates_array, seq_len, horizon):
    X, y, dates = [], [], []
    for i in range(len(X_data) - seq_len - horizon + 1):
        X.append(X_data[i : i + seq_len])
        y_window = y_data[i + seq_len : i + seq_len + horizon]
        # Убрали np.log1p для повышения стабильности R2 и работы в реальных Ваттах
        y.append(y_window) 
        dates.append(dates_array[i + seq_len]) 
    return np.array(X), np.array(y), np.array(dates)


def build_legacy_hybrid_model(input_shape=(168, 57)):
    """
    Архитектура для старых весов из Kaggle, чтобы избежать ошибки загрузки Lambda слоев
    и несовпадения форм (kernel_size=3, Mish activation и т.д.).
    """
    inputs = layers.Input(shape=input_shape, name="input_layer_2")
    
    # CNN
    x_cnn = layers.Conv1D(64, kernel_size=3, strides=2, padding='same', activation='mish',
                          kernel_regularizer=regularizers.l2(0.0001), name="conv1d_6")(inputs)
    x_cnn = layers.BatchNormalization(name="batch_normalization_4")(x_cnn)
    x_cnn = layers.SpatialDropout1D(0.2, name="spatial_dropout1d_2")(x_cnn)
    
    x_cnn = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='mish',
                          name="conv1d_7")(x_cnn)
    x_cnn = layers.MaxPooling1D(pool_size=2, strides=2, name="max_pooling1d_2")(x_cnn)
    
    # LSTM
    x_rnn = layers.LSTM(128, return_sequences=True, dropout=0.2, name="lstm_4")(x_cnn)
    
    # Attention
    attn = layers.MultiHeadAttention(num_heads=8, key_dim=32, value_dim=32, name="multi_head_attention_2")(x_rnn, x_rnn)
    x_rnn = layers.Add(name="add_4")([x_rnn, attn])
    x_rnn = layers.LayerNormalization(name="layer_normalization_6")(x_rnn)
    
    # Pooling
    x_pool_avg = layers.GlobalAveragePooling1D(name="global_average_pooling1d_2")(x_rnn)
    x_pool_max = layers.GlobalMaxPooling1D(name="global_max_pooling1d_2")(x_rnn)
    
    # Lambda replacement (extract last timestep)
    x_last = layers.Lambda(lambda x: x[:, -1, :], name="lambda")(inputs)
    
    # Dense from last step
    x_dense_4 = layers.Dense(32, activation='mish', name="dense_4")(x_last)
    
    # Concat
    x = layers.Concatenate(name="concatenate_4")([x_pool_avg, x_pool_max, x_dense_4])
    
    # Final Dense
    x = layers.Dense(256, activation='mish', kernel_regularizer=regularizers.l2(0.0001), name="dense_5")(x)
    x = layers.Dropout(0.3, name="dropout_7")(x)
    x = layers.Dense(128, activation='mish', kernel_regularizer=regularizers.l2(0.0001), name="dense_6")(x)
    x = layers.Dense(24, activation='linear', name="dense_7")(x)
    
    return models.Model(inputs=inputs, outputs=x, name="Hybrid_Model")
    