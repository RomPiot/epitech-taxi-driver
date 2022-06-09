from copy import copy
from django.test import TestCase
from t_aia_902.services.q_learning import QLearning
import numpy as np


class QlearningTestCase(TestCase):
    def test_qlearning_without_episode(self):
        qlearning = QLearning()
        qlearning_empty_qtable = copy(QLearning().q_table)
        qlearning_data = qlearning.play_to_taxi(
            environment=QLearning.env, q_table=QLearning.q_table, episodes=0, show_print=False
        )
        self.assertEqual(type(qlearning_data), dict)
        self.assertEqual(qlearning_data["avg_rewards"], 0)
        self.assertEqual(qlearning_data["avg_steps"], 0)
        self.assertEqual(qlearning_data["duration"], 0)
        self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), True)

    def test_qlearning_without_training(self):
        qlearning = QLearning()
        qlearning_empty_qtable = copy(QLearning().q_table)
        qlearning_data = qlearning.play_to_taxi(
            environment=QLearning.env,
            q_table=QLearning.q_table,
            episodes=100,
            training=False,
            show_print=False,
            reset=True,
        )
        self.assertEqual(type(qlearning_data), dict)
        self.assertEqual(qlearning_data["avg_rewards"], 0)
        self.assertEqual(qlearning_data["avg_steps"], 99)
        self.assertEqual(qlearning_data["duration"] < 0.2, True)
        self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), True)

    def test_qlearning_private_training(self):
        qlearning = QLearning()
        qlearning_empty_qtable = copy(QLearning().q_table)
        qlearning_data = qlearning.play_to_taxi(
            environment=QLearning.env,
            q_table=QLearning.q_table,
            episodes=500,
            show_print=False,
            reset=True,
        )
        self.assertEqual(type(qlearning_data), dict)
        self.assertEqual(qlearning_data["avg_rewards"] < 0, True)
        self.assertEqual(qlearning_data["avg_steps"] != 99, True)
        self.assertEqual(qlearning_data["duration"] > 0.2, True)
        self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), False)

    def test_qlearning_bad_training(self):
        qlearning = QLearning()
        qlearning_empty_qtable = copy(QLearning().q_table)
        nb_episode = 100

        result_training = qlearning.play_to_taxi(
            environment=qlearning.env,
            q_table=qlearning.q_table,
            episodes=nb_episode,
            training=True,
            reset=True,
            show_print=False,
        )

        qlearning_data = qlearning.play_to_taxi(
            environment=qlearning.env,
            q_table=result_training["q_table"],
            episodes=nb_episode,
            training=False,
            show_print=False,
        )

        self.assertEqual(type(qlearning_data), dict)
        self.assertEqual(qlearning_data["avg_rewards"] == 0, True)
        self.assertEqual(qlearning_data["avg_steps"] == 99, True)
        self.assertEqual(qlearning_data["duration"] < 0.2, True)
        self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), False)

    def test_qlearning_good_training(self):
        qlearning = QLearning()
        qlearning_empty_qtable = copy(QLearning().q_table)
        nb_episode = 10000

        result_training = qlearning.play_to_taxi(
            environment=qlearning.env,
            q_table=qlearning.q_table,
            episodes=nb_episode,
            training=True,
            reset=True,
            show_print=False,
        )

        qlearning_data = qlearning.play_to_taxi(
            environment=qlearning.env,
            q_table=result_training["q_table"],
            episodes=nb_episode,
            training=False,
            show_print=False,
        )

        self.assertEqual(type(qlearning_data), dict)
        self.assertEqual(qlearning_data["avg_rewards"] > 0, True)
        self.assertEqual(qlearning_data["avg_steps"] < 80, True)
        self.assertEqual(qlearning_data["duration"] < 15, True)
        self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), False)

    def test_qlearning_taxi_user_method_good_training(self):
        qlearning = QLearning()
        qlearning_empty_qtable = copy(QLearning().q_table)
        qlearning_data = qlearning.taxi(
            nb_episodes=10000,
        )
        print(qlearning_data["avg_rewards"])
        self.assertEqual(type(qlearning_data), dict)
        self.assertEqual(qlearning_data["avg_rewards"] > 0, True)
        self.assertEqual(qlearning_data["avg_steps"] != 99, True)
        self.assertEqual(qlearning_data["duration"] > 0.2, True)
        self.assertEqual(np.array_equal(qlearning_data["q_table"], qlearning_empty_qtable), False)
