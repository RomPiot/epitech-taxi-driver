from django.urls import path
from t_aia_902 import views


urlpatterns = [
    path("", views.algo_choice, name="home"),
    path("algo_params", views.algo_params, name="home"),
]
