# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 22:50:19 2023

@author: math
"""

from django.urls import path
from . import views
urlpatterns = [
    path('', views.post_list, name='post_list'),
]