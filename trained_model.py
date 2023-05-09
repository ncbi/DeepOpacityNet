import tensorflow as tf

from tensorflow.keras.models import load_model

def load(model_path):
    
    model = load_model(model_path)
        
    return model