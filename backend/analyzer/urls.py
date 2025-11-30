from django.urls import path
from .views import PredictView, MitigateView, GeminiAnalysisView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('mitigate/', MitigateView.as_view(), name='mitigate'),
    path('gemini-analysis/', GeminiAnalysisView.as_view(), name='gemini-analysis'),
]
