from django.urls import path
from .views import (
    main_menu,
    create_database,
    details,
)

app_name = 'predictor'

urlpatterns = [
    path('', main_menu, name='main-menu'),
    path('create_database/', create_database, name='create-database'),
    path('details/<str:hause>/', details, name='details'),
]