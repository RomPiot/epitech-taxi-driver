from django.urls import path
from t_aia_902 import views


urlpatterns = [
    path("", views.page_home, name="home"),
]
