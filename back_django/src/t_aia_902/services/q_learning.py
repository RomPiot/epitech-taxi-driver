import gym
import numpy as np
import pandas as pd
from random import random
import time

# import csv


class QLearning:
    env = gym.make("Taxi-v3")
    env.reset()

    actions = env.action_space.n
    states = env.observation_space.n

    # Create Q table of rewards defined to 0, for each actions on each step
    q_table = np.zeros((states, actions))

    def open_dataset(self, file_name):
        return pd.read_csv(filepath_or_buffer=file_name, delimiter=",", encoding="utf-8", header=0)

    def play_to_taxi(
        self,
        environment,
        q_table,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_max=1.0,
        learning_rate=0.02,
        reward_discount_rate=0.618,
        decay_rate=0.01,
        episodes=10000,
        max_steps=99,
        training=True,
        reset=False,
        show_print=True,
    ):
        t = time.process_time()
        duration = 0
        won_episode = 0
        total_steps = []
        total_rewards = []

        if episodes > 0:

            if reset:
                q_table = np.zeros((self.states, self.actions))

            done = False

            if show_print:
                if training:
                    print("")
                    print("---------------------- TRAINING ----------------------")
                else:
                    print("")
                    print("---------------------- TESTING ----------------------")

            for episode in range(episodes):
                environment.reset()
                state = 0
                total_episode_rewards = 0
                total_episode_steps = 0
                done = False

                for step in range(max_steps):
                    # epsilon-greedy
                    if training and random() < epsilon:  # Exploration
                        action = environment.action_space.sample()  # get random action

                    else:  # Exploitation
                        possibilities = q_table[state, :]  # possibilities from current state
                        action = np.argmax(possibilities)  # get the best direction depending on the reward value

                    # Move to direction
                    next_state, reward, done, info = environment.step(action)

                    if training:
                        # Update Q table with value function
                        # V(s) = V(s) + (lr x (V(s') - V(s)))
                        # state_value = state_value + alpha x (reward + gamma x next_state_value - state_value)
                        q_table[state, action] = q_table[state, action] + learning_rate * (
                            reward + reward_discount_rate * np.max(q_table[next_state, :]) - q_table[state, action]
                        )

                    state = next_state
                    total_episode_steps = step + 1

                    # Update statistics
                    total_episode_rewards += reward

                    if done:
                        total_rewards.append(total_episode_rewards)
                        won_episode += 1
                        break

                # game is ended
                total_steps.append(total_episode_steps)

                # epsilon decay to maintain trade-off between exploration-exploitation
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * episode)

            total_time = time.process_time() - t
            avg_rewards = round(sum(total_rewards) / len(total_rewards) if len(total_rewards) > 0 else 0, 2)
            avg_steps = round(sum(total_steps) / len(total_steps) if len(total_steps) > 0 else 0, 2)
            won_rate = round(won_episode / episodes * 100 if won_episode > 0 else 0, 2)
            duration = round(total_time if total_time > 0 else 0, 2)

            if show_print:
                print(f"Total episodes : {episodes}")
                print(f"Average rewards : {avg_rewards}")
                print(f"Average steps : {avg_steps}")
                print(f"Won episode : {won_episode} ({won_rate}%)")

                if training:
                    print(f"Training time : {duration} seconds")

        avg_rewards = avg_rewards if "avg_rewards" in locals() else 0
        avg_steps = avg_steps if "avg_steps" in locals() else 0
        duration = duration if "duration" in locals() else 0
        won_rate = won_rate if "won_rate" in locals() else 0

        return {
            "q_table": q_table,
            "avg_rewards": avg_rewards,
            "avg_steps": avg_steps,
            "duration": duration,
        }

    # qlearning_data = open_dataset("q-learning.csv")
    # qlearning_data = qlearning_data.sort_values(by=["avg_rewards"], ascending=[False]).reset_index(drop=True)
    # qlearning_data.head(5)

    def taxi(
        self,
        nb_episodes=10000,
        epsilon_rate=1.0,
        epsilon_min=0.01,
        epsilon_max=1.0,
        learning_rate=0.8,
        reward_discount_rate=0.786,
        decay_rate=0.07,
    ):
        # nb_episodes = int(input("Episode Iterations : (default = 10 000)") or 10000)
        # epsilon_rate = float(input("Epsilon Rate : (default = 1.0)") or 1.0)
        # epsilon_min = float(input("Epsilon Min :(default = 0.01)") or 0.01)
        # epsilon_max = float(input("Epsilon Max : (default = 1.0)") or 1.0)
        # learning_rate = float(input("Learning Rate : (default = 0.8)") or 0.8)
        # reward_discount_rate = float(input("Reward Discount Rate : (default = 0.786)") or 0.786)
        # decay_rate = float(input("Decay Rate : (default = 0.07)") or 0.07)

        print("---------------------- Parameters ----------------------")
        print(f"Episode iterations : {nb_episodes}")
        print(f"Epsilon rate : {epsilon_rate}")
        print(f"Epsilon min : {epsilon_min}")
        print(f"Epsilon max : {epsilon_max}")
        print(f"Learning rate : {learning_rate}")
        print(f"Reward discount rate : {reward_discount_rate}")
        print(f"Decay rate : {decay_rate}")

        result_training = self.play_to_taxi(
            environment=self.env,
            q_table=self.q_table,
            epsilon=epsilon_rate,
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
            learning_rate=learning_rate,
            episodes=nb_episodes,
            reward_discount_rate=reward_discount_rate,
            decay_rate=decay_rate,
            training=True,
            reset=True,
        )

        result_testing = self.play_to_taxi(
            environment=self.env,
            q_table=result_training["q_table"],
            epsilon=epsilon_rate,
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
            learning_rate=learning_rate,
            episodes=nb_episodes,
            reward_discount_rate=reward_discount_rate,
            decay_rate=decay_rate,
            training=False,
        )

        return result_testing


ql = QLearning()
training = ql.play_to_taxi(q_table=QLearning.q_table, environment=QLearning.env, show_print=True)
result = ql.play_to_taxi(q_table=training["q_table"], environment=QLearning.env, show_print=True)

print(result)
