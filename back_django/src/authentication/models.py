from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    birthday = models.DateField(null=True)
    more_informations = models.CharField(null=True, blank=True, max_length=255)
