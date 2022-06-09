import time

import numpy as np
import gym
import pygame
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Reshape, Embedding
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def build_model(actions):
    model = Sequential()
    model.add(Embedding(500, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def launch_DeepRl(nb_episodes):
    env = gym.make("Taxi-v3")
    env.reset()
    states = env.observation_space.n
    print(states)
    actions = env.action_space.n
    print(actions)
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy()
    model = build_model(actions)
    dqn = DQNAgent(model=model, nb_actions=actions, memory=memory, nb_steps_warmup=500,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    dqn.fit(env, nb_steps=nb_episodes, visualize=False, verbose=1, nb_max_episode_steps=99, log_interval=nb_episodes/10)

    dqn.save_weights(filepath="saved_model/custom_number_of_epoch.h5", overwrite=True)




def load_DeepRl(path, testnb=100):
    start = time.time()
    env = gym.make("Taxi-v3")
    env.reset()
    states = env.observation_space.n
    actions = env.action_space.n

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy()
    model = build_model(actions)
    dqn = DQNAgent(model=model, nb_actions=actions, memory=memory, nb_steps_warmup=500,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.load_weights(path)

    history = dqn.test(env, nb_episodes=testnb, visualize=False, nb_max_episode_steps=200)
    total_reward = 0
    total_step = 0

    for value in history.history["episode_reward"]:
        total_reward = total_reward + float(value)
    average_reward = total_reward / testnb
    print("average reward:")
    print(average_reward)

    for step in history.history["nb_steps"]:
        total_step = total_step + float(step)
    average_step = total_step/testnb
    print("average step:")
    print (average_step)

    #duration, average reward, average step
    return (time.time() - start, average_reward, average_step)





# NB_CUSTOM = 100000
# Lancer un nombre custom:
# launch_DeepRl(NB_CUSTOM)
load_DeepRl("saved_model/custom_number_of_epoch.h5")

# Lancer le benchmark de 50 000 epoch
load_DeepRl("saved_model/benchmark.h5")

# Lancer le model entrainé de 1M epoch
load_DeepRl("saved_model/trained_100000.h5")
