# detection/urls.py

from django.urls import path
from .views import video_feed, index

urlpatterns = [
    path('', index, name='index'),
    path('live_feed/', video_feed, name='video_feed'),
]
