# api/utils.py
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from django.conf import settings
import os

# Load model once at startup
MODEL_PATH = os.path.join(settings.BASE_DIR, "api", "model", "cassava_model_v3.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Wrap prediction in a tf.function for faster CPU inference
@tf.function
def predict_fn(img_array):
    return model(img_array, training=False)

# Map model output to disease names
LABEL_MAP = {
    0: "Cassava Bacterial Blight",
    1: "Cassava Brown Streak",
    2: "Cassava Green Mottle",
    3: "Cassava Mosaic",
    4: "Healthy"
}

def preprocess_leaf_image(file_obj):
    """Convert uploaded file to 224x224 MobileNetV2-ready array"""
    img = Image.open(file_obj).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict_disease(file_obj):
    """Return predicted disease name"""
    img_array = preprocess_leaf_image(file_obj)
    preds = predict_fn(img_array).numpy()
    class_idx = int(np.argmax(preds, axis=1)[0])
    return LABEL_MAP[class_idx]
