from pprint import pprint
from django.shortcuts import render

from .forms.qlearning_params import QLearningParamsForm
from .forms.select_algo import SelectAlgoForm
from .services.q_learning import QLearning

# from .forms import InputForm


def algo_choice(request):
    context = {}

    if request.method == "POST":
        form = SelectAlgoForm(request.POST)

        if form.is_valid():
            pprint(form.data)
            if form.data["algo"] == "q-learning":
                context["form"] = QLearningParamsForm()
            elif form.data["algo"] == "deep-q-learning":
                context["form"] = QLearningParamsForm()
            elif form.data["algo"] == "sarsa":
                context["form"] = QLearningParamsForm()

            return render(request, "page/home.html", context)

    else:
        context["form"] = SelectAlgoForm()

    return render(request, "page/home.html", context)


def algo_params(request):
    context = {}

    if request.method == "POST":
        form = QLearningParamsForm(request.POST)

        if form.is_valid():
            pprint(form.data["decay_rate"])
            result = QLearning().taxi(
                nb_episodes=int(form.data["nb_episodes"]),
                epsilon_rate=float(form.data["epsilon_rate"]),
                epsilon_min=float(form.data["epsilon_min"]),
                epsilon_max=float(form.data["epsilon_max"]),
                learning_rate=float(form.data["learning_rate"]),
                reward_discount_rate=float(form.data["reward_discount_rate"]),
                decay_rate=float(form.data["decay_rate"]),
            )
            pprint(result)

            return render(request, "page/home.html", context)

    else:
        context["form"] = QLearningParamsForm()

    return render(request, "page/home.html", context)
