# api/gemini.py
from google import genai
from django.conf import settings

client = genai.Client(api_key=settings.NLP_API_KEY)

def generate_disease_advice(disease_name: str) -> str:
    prompt = f"""
You are an agricultural assistant helping small-scale cassava farmers.

Disease detected: {disease_name}

Explain clearly:
1. What this disease is
2. How it affects cassava plants
3. Common causes and how it spreads
4. General prevention and control measures
5. Common categories of pesticides or insecticides used (NO brand names, NO dosages)
6. Safety and environmental precautions

Use simple, practical, farmer-friendly language.
Avoid medical or chemical dosages.
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )

    return response.text.strip()
