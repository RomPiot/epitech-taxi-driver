from django.shortcuts import render

from .forms.user_input import UserInputForm

# from .forms import InputForm


def page_home(request):
    context = {}
    context["form"] = UserInputForm()

    return render(request, "page/home.html", context)
