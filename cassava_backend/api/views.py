# api/views.py
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .utils import predict_disease
from .services.api_service import generate_disease_advice

# Optional: simple in-memory cache for advice
ADVICE_CACHE = {}

class PredictLeafDisease(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        file_obj = request.FILES.get("image")
        if not file_obj:
            return Response({"error": "No image provided"}, status=400)

        try:
            prediction = predict_disease(file_obj)
        except Exception as e:
            return Response({"error": "Prediction failed", "details": str(e)}, status=500)

        return Response({"prediction": prediction})


@csrf_exempt
def predict_form_view(request):
    prediction = None
    if request.method == "POST" and request.FILES.get("image"):
        file_obj = request.FILES["image"]
        try:
            prediction = predict_disease(file_obj)
        except Exception as e:
            prediction = f"Error: {e}"

    return render(request, "upload.html", {"prediction": prediction})


@api_view(["POST"])
def disease_advice(request):
    disease = request.data.get("disease")
    if not disease:
        return Response({"error": "Disease name is required"}, status=status.HTTP_400_BAD_REQUEST)

    # Check cache first
    if disease in ADVICE_CACHE:
        return Response({"disease": disease, "advice": ADVICE_CACHE[disease]}, status=200)

    try:
        advice = generate_disease_advice(disease)
        ADVICE_CACHE[disease] = advice  # store in cache
    except Exception as e:
        return Response({"error": "Failed to generate advice", "details": str(e)}, status=500)

    return Response({"disease": disease, "advice": advice}, status=200)
