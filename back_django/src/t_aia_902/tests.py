import os
from copy import copy
from django.test import TestCase
import numpy as np
from app.settings.base import BASE_DIR
from selenium import webdriver
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
import time
from t_aia_902.services.q_learning import QLearning
from t_aia_902.services.sarsa import Sarsa
from t_aia_902.services.deep_q_learning import DeepQLearning


class FunctionalTestCase(StaticLiveServerTestCase):
    def test_open_chrome_window(self):
        self.browser = webdriver.Chrome("tests/functional_tests/chromedriver")
        self.browser.get(self.live_server_url)
        time.sleep(30)
        self.browser.close()


# class QlearningTestCase(TestCase):
#     def test_qlearning_without_episode(self):
#         qlearning = QLearning()
#         qlearning_empty_qtable = copy(QLearning().q_table)
#         qlearning_data = qlearning.play_to_taxi(
#             environment=QLearning.env, q_table=QLearning.q_table, episodes=0, show_print=False
#         )
#         self.assertEqual(type(qlearning_data), dict)
#         self.assertEqual(qlearning_data["avg_rewards"], 0)
#         self.assertEqual(qlearning_data["avg_steps"], 0)
#         self.assertEqual(qlearning_data["duration"], 0)
#         self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), True)

#     def test_qlearning_without_training(self):
#         qlearning = QLearning()
#         qlearning_empty_qtable = np.zeros((QLearning().states, QLearning().actions))
#         qlearning_data = qlearning.play_to_taxi(
#             environment=QLearning.env,
#             q_table=QLearning.q_table,
#             episodes=100,
#             training=False,
#             show_print=False,
#             reset=True,
#         )
#         self.assertEqual(type(qlearning_data), dict)
#         self.assertEqual(qlearning_data["avg_rewards"], 0)
#         self.assertEqual(qlearning_data["avg_steps"], 99)
#         self.assertEqual(qlearning_data["duration"] < 0.5, True)
#         self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), True)

#     def test_qlearning_private_training(self):
#         qlearning = QLearning()
#         qlearning_empty_qtable = np.zeros((QLearning().states, QLearning().actions))
#         qlearning_data = qlearning.play_to_taxi(
#             environment=QLearning.env,
#             q_table=QLearning.q_table,
#             episodes=500,
#             show_print=False,
#             reset=True,
#         )
#         self.assertEqual(type(qlearning_data), dict)
#         self.assertEqual(qlearning_data["avg_rewards"] < 0, True)
#         self.assertEqual(qlearning_data["avg_steps"] != 99, True)
#         self.assertEqual(qlearning_data["duration"] > 0.2, True)
#         self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), False)

#     def test_qlearning_bad_training(self):
#         qlearning = QLearning()
#         qlearning_empty_qtable = np.zeros((QLearning().states, QLearning().actions))
#         nb_episode = 100

#         result_training = qlearning.play_to_taxi(
#             environment=qlearning.env,
#             q_table=qlearning.q_table,
#             episodes=nb_episode,
#             training=True,
#             reset=True,
#             show_print=False,
#         )

#         qlearning_data = qlearning.play_to_taxi(
#             environment=qlearning.env,
#             q_table=result_training["q_table"],
#             episodes=nb_episode,
#             training=False,
#             show_print=False,
#         )

#         self.assertEqual(type(qlearning_data), dict)
#         self.assertEqual(qlearning_data["avg_rewards"] == 0, True)
#         self.assertEqual(qlearning_data["avg_steps"] == 99, True)
#         self.assertEqual(qlearning_data["duration"] < 0.5, True)
#         self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), False)

#     def test_qlearning_good_training(self):
#         qlearning = QLearning()
#         qlearning_empty_qtable = np.zeros((QLearning().states, QLearning().actions))
#         nb_episode = 10000

#         result_training = qlearning.play_to_taxi(
#             environment=qlearning.env,
#             q_table=qlearning.q_table,
#             episodes=nb_episode,
#             training=True,
#             reset=True,
#             show_print=False,
#         )

#         qlearning_data = qlearning.play_to_taxi(
#             environment=qlearning.env,
#             q_table=result_training["q_table"],
#             episodes=nb_episode,
#             training=False,
#             show_print=False,
#         )

#         self.assertEqual(type(qlearning_data), dict)
#         self.assertEqual(qlearning_data["avg_rewards"] > 0, True)
#         self.assertEqual(qlearning_data["avg_steps"] < 80, True)
#         self.assertEqual(qlearning_data["duration"] < 30, True)
#         self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), False)

#     def test_qlearning_taxi_user_method_good_training(self):
#         qlearning = QLearning()
#         qlearning_empty_qtable = np.zeros((QLearning().states, QLearning().actions))
#         qlearning_data = qlearning.taxi(
#             nb_episodes=10000,
#         )
#         print(qlearning_data["avg_rewards"])
#         self.assertEqual(type(qlearning_data), dict)
#         self.assertEqual(qlearning_data["avg_rewards"] > 0, True)
#         self.assertEqual(qlearning_data["avg_steps"] != 99, True)
#         self.assertEqual(qlearning_data["duration"] > 0.2, True)
#         self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), False)


# class SarsaTestCase(TestCase):
#     def test_sarsa_without_episode(self):
#         sarsa = Sarsa()
#         sarsa_empty_qtable = np.zeros((Sarsa().states, Sarsa().actions))
#         sarsa_data = sarsa.take_taxi_with_sarsa(
#             environment=Sarsa().env, q_table=Sarsa().q_table, episodes=0, show_print=False
#         )
#         self.assertEqual(type(sarsa_data), dict)
#         self.assertEqual(sarsa_data["avg_rewards"], 0)
#         self.assertEqual(sarsa_data["avg_steps"], 0)
#         self.assertEqual(sarsa_data["duration"], 0)
#         self.assertEqual(np.array_equal(sarsa_data["q_table"], sarsa_empty_qtable), True)

#     def test_sarsa_without_training(self):
#         sarsa = Sarsa()
#         sarsa_empty_qtable = np.zeros((Sarsa().states, Sarsa().actions))
#         sarsa_data = sarsa.take_taxi_with_sarsa(
#             environment=Sarsa.env,
#             q_table=Sarsa.q_table,
#             episodes=100,
#             training=False,
#             show_print=False,
#             reset=True,
#         )
#         self.assertEqual(type(sarsa_data), dict)
#         self.assertEqual(sarsa_data["avg_rewards"], 0)
#         self.assertEqual(sarsa_data["avg_steps"], 99)
#         self.assertEqual(sarsa_data["duration"] < 0.5, True)
#         self.assertEqual(np.array_equal(sarsa_data["q_table"], sarsa_empty_qtable), True)

#     def test_sarsa_private_training(self):
#         sarsa = Sarsa()
#         sarsa_empty_qtable = np.zeros((Sarsa().states, Sarsa().actions))
#         sarsa_data = sarsa.take_taxi_with_sarsa(
#             environment=Sarsa.env,
#             q_table=Sarsa.q_table,
#             episodes=500,
#             show_print=False,
#             reset=True,
#         )
#         self.assertEqual(type(sarsa_data), dict)
#         self.assertEqual(sarsa_data["avg_rewards"] < 0, True)
#         self.assertEqual(sarsa_data["avg_steps"] != 99, True)
#         self.assertEqual(sarsa_data["duration"] > 0.2, True)
#         self.assertEqual(np.array_equal(sarsa_data["q_table"], sarsa_empty_qtable), False)

#     def test_sarsa_bad_training(self):
#         sarsa = Sarsa()
#         sarsa_empty_qtable = np.zeros((Sarsa().states, Sarsa().actions))
#         nb_episode = 100

#         result_training = sarsa.take_taxi_with_sarsa(
#             environment=sarsa.env,
#             q_table=sarsa.q_table,
#             episodes=nb_episode,
#             training=True,
#             reset=True,
#             show_print=False,
#         )

#         sarsa_data = sarsa.take_taxi_with_sarsa(
#             environment=sarsa.env,
#             q_table=result_training["q_table"],
#             episodes=nb_episode,
#             training=False,
#             show_print=False,
#         )

#         self.assertEqual(type(sarsa_data), dict)
#         self.assertEqual(sarsa_data["avg_rewards"] == 0, True)
#         self.assertEqual(sarsa_data["avg_steps"] == 99, True)
#         self.assertEqual(sarsa_data["duration"] < 0.5, True)
#         self.assertEqual(np.array_equal(sarsa_data["q_table"], sarsa_empty_qtable), False)

#     def test_sarsa_good_training(self):
#         sarsa = Sarsa()
#         sarsa_empty_qtable = np.zeros((Sarsa().states, Sarsa().actions))
#         nb_episode = 10000

#         result_training = sarsa.take_taxi_with_sarsa(
#             environment=sarsa.env,
#             q_table=sarsa.q_table,
#             episodes=nb_episode,
#             training=True,
#             reset=True,
#             show_print=False,
#         )

#         sarsa_data = sarsa.take_taxi_with_sarsa(
#             environment=sarsa.env,
#             q_table=result_training["q_table"],
#             episodes=nb_episode,
#             training=False,
#             show_print=False,
#         )

#         self.assertEqual(type(sarsa_data), dict)
#         self.assertEqual(sarsa_data["avg_rewards"] > 0, True)
#         self.assertEqual(sarsa_data["avg_steps"] < 80, True)
#         self.assertEqual(sarsa_data["duration"] < 30, True)
#         self.assertEqual(np.array_equal(sarsa_data["q_table"], sarsa_empty_qtable), False)


# class DeepQlearningTestCase(TestCase):
#     def test_deepqlearning_with_1_000_000_episodes(self):
#         file_to_load = os.path.join(BASE_DIR, "src/t_aia_902/data/saved_model/trained_1000000.h5")
#         deep_q_learning = DeepQLearning().load_DeepRl(file_to_load)
#         print(deep_q_learning)
#         self.assertEqual(type(deep_q_learning), dict)
#         self.assertEqual(deep_q_learning["avg_rewards"] > 5, True)
#         self.assertEqual(deep_q_learning["avg_steps"] < 20, True)
#         self.assertEqual(deep_q_learning["duration"] < 5, True)

#     def test_deepqlearning_with_100_000_episodes(self):
#         file_to_load = os.path.join(BASE_DIR, "src/t_aia_902/data/saved_model/trained_100000.h5")
#         deep_q_learning = DeepQLearning().load_DeepRl(file_to_load)
#         print(deep_q_learning)
#         self.assertEqual(type(deep_q_learning), dict)
#         self.assertEqual(deep_q_learning["avg_rewards"] < 0, True)
#         self.assertEqual(deep_q_learning["avg_steps"] > 50, True)
#         self.assertEqual(deep_q_learning["duration"] > 5, True)

#     def test_deepqlearning_with_400_000_episodes(self):
#         file_to_load = os.path.join(BASE_DIR, "src/t_aia_902/data/saved_model/trained_400000.h5")
#         deep_q_learning = DeepQLearning().load_DeepRl(file_to_load)
#         print(deep_q_learning)
#         self.assertEqual(type(deep_q_learning), dict)
#         self.assertEqual(deep_q_learning["avg_rewards"] < 0, True)
#         self.assertEqual(deep_q_learning["avg_steps"] > 50, True)
#         self.assertEqual(deep_q_learning["duration"] > 5, True)

#     def test_deepqlearning_with_600_000_episodes(self):
#         file_to_load = os.path.join(BASE_DIR, "src/t_aia_902/data/saved_model/trained_600000.h5")
#         deep_q_learning = DeepQLearning().load_DeepRl(file_to_load)
#         print(deep_q_learning)
#         self.assertEqual(type(deep_q_learning), dict)
#         self.assertEqual(deep_q_learning["avg_rewards"] > 0, True)
#         self.assertEqual(deep_q_learning["avg_steps"] < 20, True)
#         self.assertEqual(deep_q_learning["duration"] < 5, True)
