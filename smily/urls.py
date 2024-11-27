from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from smily import views
from .views import imageapi

urlpatterns = [
    path('admin/', admin.site.urls),
    # path("image",views.imageapi.as_view()),
    # # path("upload",views.upload_view()),s
    # path('upload/',views.upload_view, name='upload_file'),
    # # path("home",views.home)
    # path('upload/', upload, name='upload_view'),
    path('image/', imageapi.as_view(), name='image_process'),
    # path('upload/',views.upload_view),
]

