from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.conf import settings
import tensorflow as tf
import numpy as np
import os

# Load model once 
# MODEL_PATH = "C:/Users/RA_CT/Desktop/Tek_Juice/Backend/cassava_backend/api/assets/cassava_model_v3.h5"
MODEL_PATH = os.path.join(settings.BASE_DIR, "api", "model", "cassava_model.h5")
model = load_model(MODEL_PATH)

# Map model output to disease names
label_map = {
    0: "Cassava Bacterial Blight",
    1: "Cassava Brown Streak",
    2: "Cassava Green Mottle",
    3: "Cassava Mosaic",
    4: "Healthy"
}

def preprocess_leaf_image(img_path):
    """
    Preprocess image for inference:
    - Resize to 224x224
    - Apply MobileNetV2 preprocessing
    """
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

class PredictLeafDisease(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        file_obj = request.FILES.get('image')
        if not file_obj:
            return Response({"error": "No image provided"}, status=400)

        # Save temporary image
        img_path = f"temp_{file_obj.name}"
        with open(img_path, "wb") as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        # Preprocess for inference
        img_array = preprocess_leaf_image(img_path)

        # Predict
        preds = model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        prediction = label_map[class_idx]

        # Clean up
        os.remove(img_path)

        return Response({"prediction": prediction})


@csrf_exempt  # only for testing in dev; in production, handle CSRF properly
def predict_form_view(request):
    prediction = None
    if request.method == "POST" and request.FILES.get('image'):
        file_obj = request.FILES['image']
        img_path = f"temp_{file_obj.name}"
        with open(img_path, "wb") as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        # Preprocess and predict
        img_array = preprocess_leaf_image(img_path)
        preds = model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        prediction = label_map[class_idx]

        os.remove(img_path)

    return render(request, "upload.html", {"prediction": prediction})