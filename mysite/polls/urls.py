from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('polls/imageSubmit',views.imageSubmit, name='imageSubmit'),
]
