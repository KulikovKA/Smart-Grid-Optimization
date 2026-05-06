import traceback
import sys
import tensorflow as tf
from utils import asymmetric_profit_loss

dl_models = {
    'GRU (Custom Loss)': 'gru_model.keras',
    'LSTM (Custom Loss)': 'lstm_model.keras'
}

for name, filename in dl_models.items():
    print(f"Testing {name}...")
    sys.stdout.flush()
    try:
        model = tf.keras.models.load_model(
            f'model_weights/{filename}', 
            custom_objects={'asymmetric_profit_loss': asymmetric_profit_loss}, 
            safe_mode=False
        )
        print(f"Successfully loaded {name}")
        sys.stdout.flush()
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
