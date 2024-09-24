from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from smily import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("image",views.imageapi.as_view()),
    # path("home",views.home)
]
