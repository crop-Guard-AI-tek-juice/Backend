from django.urls import path
from .views import PredictLeafDisease, predict_form_view

urlpatterns = [
    path('predict/', PredictLeafDisease.as_view(), name='predict-leaf-disease'),
    path('predict-form/', predict_form_view, name='predict-form'), 
]
