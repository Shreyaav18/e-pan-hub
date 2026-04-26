"""Project URL configuration.

This repo uses a single-file Django settings module (settings.py) and expects
ROOT_URLCONF = 'urls'.

We expose:
- /verify/...  (existing document_verification app URLs)
- /aml/monitor/ (JSON AML scoring endpoint used by Streamlit)
"""

from django.contrib import admin
from django.urls import include, path
from django.http import JsonResponse

from aml_monitoring.views import aml_monitor_view

def root_view(request):
    return JsonResponse({
        "status": "ok",
        "message": "KYC/AML Backend API",
        "endpoints": {
            "admin": "/admin/",
            "verify": "/verify/",
            "aml_monitor": "/aml/monitor/"
        }
    })

urlpatterns = [
    path("", root_view),
    path("admin/", admin.site.urls),
    path("verify/", include("document_verification.urls")),
    path("aml/monitor/", aml_monitor_view),
]
