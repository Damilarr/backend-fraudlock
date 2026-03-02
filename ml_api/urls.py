"""
URL configuration for ML API — Fraudlock
"""
from django.urls import path
from . import views

urlpatterns = [
    # Home & health
    path('', views.api_home, name='api_home'),
    path('health/', views.health_check, name='health_check'),
    path('numbers/lookup/',    views.lookup_number,    name='lookup_number'),
    path('numbers/directory/', views.number_directory, name='number_directory'),
    # Core detection
    path('check-message/', views.check_message, name='check_message'),
    path('predict/', views.predict_sms, name='predict_sms'),  # Legacy

    # Community features
    path('report/', views.report_number, name='report_number'),
    path('feedback/', views.submit_feedback, name='submit_feedback'),

    # Audio (TTS)
    path('audio/', views.generate_audio, name='generate_audio'),

    # Admin dashboard
    path('admin/stats/', views.admin_stats, name='admin_stats'),
    path('admin/logs/', views.admin_logs, name='admin_logs'),
    path('admin/numbers/', views.admin_numbers, name='admin_numbers'),
    path('admin/feedback/', views.admin_feedback, name='admin_feedback'),
    path('admin/export/', views.admin_export, name='admin_export'),
]