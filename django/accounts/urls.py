from django.urls import path
from .views import *

urlpatterns = [
    path('dashboard/', dashboard_view, name='dashboard'),
    path('stop_webcam/', stop_webcam, name='stop_webcam'),
    path('webcam_feed/', webcam_feed, name='webcam_feed'),
    path('webcam/', webcam_view, name='webcam'),
    path('board/', board, name='board'),
    path('signup/', signup, name='signup'),
    path('login/', login, name='login'),
    path('logout/', logout, name='logout'),
    path('', home, name='home'),
]