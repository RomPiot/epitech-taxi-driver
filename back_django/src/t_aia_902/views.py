from pprint import pprint
from django.shortcuts import render, redirect

from .forms.deep_qlearning_params import DeepQLearningParamsForm
from .services.deep_q_learning import DeepQLearning

from .forms.qlearning_params import QLearningParamsForm
from .forms.select_algo import SelectAlgoForm
from .services.q_learning import QLearning
import pandas as pd
import os


def algo_choice(request):
    context = {}

    if request.method == "POST":
        form = SelectAlgoForm(request.POST)

        if form.is_valid():
            # context["algo_selected"] = form.data["algo"]
            return redirect("algo_params", form.data["algo"])

    else:
        context["form"] = SelectAlgoForm()

    return render(request, "page/home.html", context)


def algo_params(request, algo_selected):
    context = {}
    module_dir = os.path.dirname(__file__)  # get current directory

    file_to_load_1 = os.path.join(module_dir, "data/saved_model/trained_500000.h5")
    deep_q_learning_best_row_2 = DeepQLearning().load_DeepRl(file_to_load_1, 50)
    pprint(deep_q_learning_best_row_2)

    file_to_load_2 = os.path.join(module_dir, "data/saved_model/trained_1000000.h5")
    deep_q_learning_best_row_2 = DeepQLearning().load_DeepRl(file_to_load_2, 50)
    pprint(deep_q_learning_best_row_2)

    return

    if algo_selected == "q-learning":
        form = QLearningParamsForm(request.POST)
    elif algo_selected == "deep-q-learning":
        form = DeepQLearningParamsForm(request.POST)
    elif algo_selected == "sarsa":
        form = QLearningParamsForm(request.POST)

    if request.method == "POST":

        if form.is_valid():
            file_path = os.path.join(module_dir, "data/q-learning.csv")
            qlearning_data = pd.read_csv(filepath_or_buffer=file_path, delimiter=",", encoding="utf-8", header=0)
            qlearning_data = qlearning_data.sort_values(by=["avg_rewards"], ascending=[False]).reset_index(drop=True)
            q_learning_best_row = qlearning_data.iloc[0]
            q_learning_best_row["duration"] = 7.5

            q_learning_best_result = {
                "avg_rewards": q_learning_best_row["avg_rewards"],
                "avg_steps": q_learning_best_row["avg_steps"],
                "duration": q_learning_best_row["duration"],
                "nb_episodes": q_learning_best_row["nb_episodes"],
                "epsilon_rate": q_learning_best_row["epsilon_rate"],
                "epsilon_min": q_learning_best_row["epsilon_min"],
                "epsilon_max": q_learning_best_row["epsilon_max"],
                "learning_rate": q_learning_best_row["learning_rate"],
                "reward_discount_rate": q_learning_best_row["reward_discount_rate"],
                "decay_rate": q_learning_best_row["decay_rate"],
            }

            # SARSA
            # TODO : change path file
            file_path = os.path.join(module_dir, "data/q-learning.csv")
            sarsa_data = pd.read_csv(filepath_or_buffer=file_path, delimiter=",", encoding="utf-8", header=0)
            sarsa_data = sarsa_data.sort_values(by=["avg_rewards"], ascending=[False]).reset_index(drop=True)
            sarsa_best_row = sarsa_data.iloc[0]
            sarsa_best_row["duration"] = 7.5

            sarsa_best_result = {
                "avg_rewards": sarsa_best_row["avg_rewards"],
                "avg_steps": sarsa_best_row["avg_steps"],
                "duration": sarsa_best_row["duration"],
                "nb_episodes": sarsa_best_row["nb_episodes"],
                "epsilon_rate": sarsa_best_row["epsilon_rate"],
                "epsilon_min": sarsa_best_row["epsilon_min"],
                "epsilon_max": sarsa_best_row["epsilon_max"],
                "learning_rate": sarsa_best_row["learning_rate"],
                "reward_discount_rate": sarsa_best_row["reward_discount_rate"],
                "decay_rate": sarsa_best_row["decay_rate"],
            }

            file_to_load = os.path.join(module_dir, "data/saved_model/trained_1000000.h5")
            deep_q_learning_best_row = DeepQLearning().load_DeepRl(file_to_load)

            # deep q learning
            deep_q_learning_best_result = {
                "avg_rewards": deep_q_learning_best_row["avg_rewards"],
                "avg_steps": deep_q_learning_best_row["avg_steps"],
                "duration": round(float(deep_q_learning_best_row["duration"]), 2),
                "nb_episodes": "1000000",
            }

            # user results
            if algo_selected == "q-learning":
                algo_user_params = QLearning().taxi(
                    nb_episodes=int(form.data["nb_episodes"]),
                    epsilon_rate=float(form.data["epsilon_rate"]),
                    epsilon_min=float(form.data["epsilon_min"]),
                    epsilon_max=float(form.data["epsilon_max"]),
                    learning_rate=float(form.data["learning_rate"]),
                    reward_discount_rate=float(form.data["reward_discount_rate"]),
                    decay_rate=float(form.data["decay_rate"]),
                )
                # pprint(algo_user_params)
                # pprint(q_learning_best_row)

                algo_user_result = {
                    "avg_rewards": algo_user_params["avg_rewards"],
                    "avg_steps": algo_user_params["avg_steps"],
                    "duration": algo_user_params["duration"],
                    "nb_episodes": form.data["nb_episodes"],
                    "epsilon_rate": form.data["epsilon_rate"],
                    "epsilon_min": form.data["epsilon_min"],
                    "epsilon_max": form.data["epsilon_max"],
                    "learning_rate": form.data["learning_rate"],
                    "reward_discount_rate": form.data["reward_discount_rate"],
                    "decay_rate": form.data["decay_rate"],
                }
            elif algo_selected == "deep-q-learning":
                nb_episodes = form.data["nb_episodes"]
                file_to_load = os.path.join(module_dir, f"data/saved_model/trained_{nb_episodes}.h5")
                algo_user_params = DeepQLearning().load_DeepRl(file_to_load)

                algo_user_result = {
                    "avg_rewards": algo_user_params["avg_rewards"],
                    "avg_steps": algo_user_params["avg_steps"],
                    "duration": round(float(algo_user_params["duration"]), 2),
                    "nb_episodes": form.data["nb_episodes"],
                }
            elif algo_selected == "sarsa":
                # TODO : change class to use
                algo_user_params = QLearning().taxi(
                    nb_episodes=int(form.data["nb_episodes"]),
                    epsilon_rate=float(form.data["epsilon_rate"]),
                    epsilon_min=float(form.data["epsilon_min"]),
                    epsilon_max=float(form.data["epsilon_max"]),
                    learning_rate=float(form.data["learning_rate"]),
                    reward_discount_rate=float(form.data["reward_discount_rate"]),
                    decay_rate=float(form.data["decay_rate"]),
                )
                # pprint(algo_user_params)
                # pprint(q_learning_best_row)

                algo_user_result = {
                    "avg_rewards": algo_user_params["avg_rewards"],
                    "avg_steps": algo_user_params["avg_steps"],
                    "duration": algo_user_params["duration"],
                    "nb_episodes": form.data["nb_episodes"],
                    "epsilon_rate": form.data["epsilon_rate"],
                    "epsilon_min": form.data["epsilon_min"],
                    "epsilon_max": form.data["epsilon_max"],
                    "learning_rate": form.data["learning_rate"],
                    "reward_discount_rate": form.data["reward_discount_rate"],
                    "decay_rate": form.data["decay_rate"],
                }

        context["algo_user_result"] = algo_user_result
        context["q_learning_best_result"] = q_learning_best_result
        context["sarsa_best_result"] = sarsa_best_result
        context["deep_q_learning_best_result"] = deep_q_learning_best_result
        context["algo_selected"] = algo_selected

        return render(request, "page/home.html", context)

    else:
        if algo_selected == "q-learning":
            form = QLearningParamsForm()
        elif algo_selected == "deep-q-learning":
            form = DeepQLearningParamsForm()
        elif algo_selected == "sarsa":
            form = QLearningParamsForm()

        context["form"] = form

    return render(request, "page/home.html", context)
