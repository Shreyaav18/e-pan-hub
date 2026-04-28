from django.urls import path
from . import views

app_name = 'document_verification'

urlpatterns = [
    path('upload/', views.upload_view, name='upload'),
    path('result/<int:pk>/', views.result_view, name='result'),
    path('api/verify/', views.api_verify_view, name='api_verify'),
]
