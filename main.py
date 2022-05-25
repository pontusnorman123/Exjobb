import random

import numpy as np
import Environment
import ShowerEnv
import os
os.add_dll_directory("C:/Program/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin")
from keras.models import Sequential
from keras.layers import Dense, Flatten,Input, Embedding,Reshape
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import tensorflow as tf
from matplotlib import pyplot as plt
import gym
import keras.optimizer_v2.learning_rate_schedule
import keras.callbacks
import keras.losses
import collections
import keras.callbacks_v1
import time
import datetime



ENV = [

    "S ",
    " A"

]

ENV2 = [

    "S A",
    "   ",
    "B C"

]

ENV3 = [

    "S A B",
    "C D E",
    "F G H",
    "I J K",
    "L M N"

]

ENV4 = [

    "S   A",
    "     ",
    "B   C",
    "     ",
    "D   E"

]


if __name__ == '__main__':



    #chose environment ENV,ENV2,ENV3 or ENV4
    env = Environment.Warehouse(ENV)



    #Action space and observation space to initialize Q-table
    action_space_size = env.action_space.n
    observation_space_size = env.size
    q_table = np.zeros((observation_space_size,action_space_size))

    # Number of episodes that the reinforcement learning model should train for
    num_episodes = 10000
    # Max number of training steps per episode
    max_steps_per_episodes = 100
    # Max number of steps for testing model
    max_steps_testing = 6

    # Q-learnig algorithm paramaters
    learning_rate = 0.1
    discount_rate = 0.0000001
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.1
    exploration_decay_rate = 0.001

    # Lists to save data
    reward_all_episodes_training = []
    steps = []
    times = []
    reward_all_episodes_testing = []

    #Training_episodes is how many times the model is retrained
    training_episodes = 0
    for i in range(training_episodes):

        ###### Manually implemented Q-learning algorithm ###### START
        start = time.time()
        for episode in range(num_episodes):

            state = env.reset()
            done = False
            rewards_current_episode_training = 0

            for step in range(max_steps_per_episodes):
                exploration_rate_threshold = random.uniform(0,1)
                if exploration_rate_threshold > exploration_rate:
                    action = np.argmax(q_table[state,:])
                else:
                    action = env.action_space.sample()
                new_state, reward, done, info = env.step(action)


                q_table[state,action] = q_table[state,action] * (1 - learning_rate) + \
                    learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

                state = new_state
                rewards_current_episode_training += reward
                if done:
                    break

            exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)\
                * np.exp(-exploration_decay_rate * episode)

            reward_all_episodes_training.append(rewards_current_episode_training)

        ###### Manually implemented Q-learning algorithm ###### END

        end = time.time()
        dif= end-start
        times.append(dif)

        ########## Testing agent ########

        for episode in range(num_episodes):


            state = env.reset()
            done = False
            rewards_current_episode_testing = 0


            for step in range(max_steps_testing):


                action = np.argmax(q_table[state, :])
                new_state, reward, done, info = env.step(action)
                state = new_state
                rewards_current_episode_testing += reward
                if done:
                    break



            reward_all_episodes_testing.append(rewards_current_episode_testing)



    counter = collections.Counter(reward_all_episodes_testing)
    rewards_per_thousand_episodes = np.array_split(np.array(reward_all_episodes_training), num_episodes/100)


    ###### saves the avrage reward per hundred epiosdes and plots ####
    count = 100
    rewards_mean=[]
    print(counter)
    for r in rewards_per_thousand_episodes:
         rewards_mean.append(sum(r / len(r)))
         #print(count, ": ", str(sum(r/100)))
         #count += 100

    plt.xlabel('Episodes in hundreds')
    plt.ylabel('Reward')
    plt.plot(rewards_mean)
    plt.savefig("graph.png")
    plt.show()

    ############## TensorFlow implementation ############

    # Action space and observation space
    actions = env.action_space.n
    inputs = env.observation_space.shape


    # Neural network model
    model = Sequential()
    model.add(Dense(units = 16,input_shape=inputs, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=actions, activation='linear'))
    model.summary()
    print(model.output_shape)


    ### Neural network paramaters
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=500000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory ,policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    opt = tf.keras.optimizers.Adam(learning_rate = 0.1)
    dqn.compile(opt, metrics=['accuracy'])

    #Training_episodes is how many times the model is retrained
    training_episodes = 1
    #Number of steps for each episode
    nb_steps = 10000

    test_rewards = []
    nb_episodes = []
    times = []

    for i in range(training_episodes):

        #Training
        start = time.time()
        history = dqn.fit(env,nb_steps=nb_steps, visualize=False, verbose=1)
        end = time.time()
        dif=end - start
        times.append(dif)


        train_rewards = history.history['episode_reward']
        nb_episodes.append(len(train_rewards))

        rewards = np.array_split(train_rewards, len(train_rewards)/100)

        rewards_mean = []
        for r in rewards:
            rewards_mean.append(sum(r/len(r)))

        plt.plot(rewards_mean)
        plt.xlabel('Episodes in thousands')
        plt.ylabel('Reward')

        plt.show()

        ## Test model
        test_history = dqn.test(env, nb_episodes=1, visualize=False,nb_max_episode_steps=2,verbose=1)


    counter = collections.Counter(history.history['episode_reward'])
    print(counter)
    print("nb_steps",nb_steps)
    print("nb_episodes_mean",np.mean(nb_episodes))
    print("time: ", np.mean(times))