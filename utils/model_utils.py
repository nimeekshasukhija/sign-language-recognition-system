
import tensorflow as tf
import os

def load_model(path="model/asl_cnn_best.h5"):
    if os.path.exists(path):
        print(f"✅ Loaded model from {path}")
        return tf.keras.models.load_model(path)
    else:
        raise FileNotFoundError("Model file not found.")

def save_model(model, path="model/asl_cnn_new.h5"):
    model.save(path)
    print(f"📦 Model saved to {path}")
