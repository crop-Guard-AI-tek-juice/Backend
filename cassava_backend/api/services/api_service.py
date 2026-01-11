import google.generativeai as genai
from django.conf import settings

genai.configure(api_key=settings.NLP_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

def generate_disease_advice(disease_name):
    prompt = f"""
    You are an agricultural assistant.

    Disease detected: {disease_name}

    Explain:
    1. What this disease is
    2. How it affects cassava plants
    3. General prevention and control measures
    4. Common categories of pesticides or insecticides used (no dosages)
    5. Safety and environmental precautions

    Keep the explanation simple and farmer-friendly.
    """

    response = model.generate_content(prompt)
    return response.text
