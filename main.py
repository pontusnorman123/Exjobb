import random

import numpy as np
import ENV2
import ShowerEnv
import os
os.add_dll_directory("C:/Program/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin")
from keras.models import Sequential
from keras.layers import Dense, Flatten,Input
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import tensorflow as tf
from matplotlib import pyplot as plt
import gym

ENV = [

    "S ",
    " A"

]

ENV3 = [

    "S   A",
    "B   C"

]

ENV4 = [

    "S   A    B",
    "C   D    E",
    "F   G    H"

]

ORDER_SIZE = 20

if __name__ == '__main__':



    env2 = gym.make("FrozenLake-v1")
    print(env2.action_space)
    print(env2.observation_space)

    env = ENV2.Warehouse(ENV4, ORDER_SIZE)
    envo = ShowerEnv.ShowerEnv()


    ######################################################


    action_space_size = env.action_space.n
    state_space_size = env.size
    q_table = np.zeros((state_space_size,action_space_size))

    num_episodes = 80000
    max_steps_per_episodes = 300
    learning_rate = 0.1
    discount_rate = 0.99
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.1
    exploration_decay_rate = 0.01

    reward_all_episodes = []

    for episode in range(num_episodes):

        state = env.reset()
        done = False
        rewards_current_episode = 0

        for step in range(max_steps_per_episodes):
        #while not done:
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state,:])
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            q_table[state,action] = q_table[state,action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done:
                break

        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)\
            * np.exp(-exploration_decay_rate * episode)

        reward_all_episodes.append(rewards_current_episode)

    rewards_per_thousand_episodes = np.split(np.array(reward_all_episodes), num_episodes/1000)
    count = 1000
    print("***********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    print(q_table)
    ######################################################


    actions = env.action_space.n
    print("Number of Actions:", actions) #S: Denna borde vara 4. för det är 4 actions.

    inputs = env.observation_space.shape
    print(env.observation_space)
    print("Inputs:", inputs) #S: Osäker kring denna... Den borde vara (95) eller (5, 19) eller (7, 21) eller nått annat beroende på hur din observerbara 'karta' ser ut...



    # model = Sequential()
    # model.add(Dense(units=32, input_shape=inputs, activation='relu'))
    # model.add(Dense(units=32, activation='relu'))
    # model.add(Dense(units=32, activation='relu'))
    # model.add(Dense(units=actions, activation='linear'))
    # model.summary()
    # print(model.output_shape)
    #
    # ######
    # policy = BoltzmannQPolicy()
    # memory = SequentialMemory(limit=500000, window_length=1)
    # dqn = DQNAgent(model=model,gamma=0.99, memory=memory ,policy=policy, nb_actions=actions, nb_steps_warmup=100, target_model_update=0.2)
    #
    # opt = tf.keras.optimizers.Adam(learning_rate = 0.1)
    # dqn.compile(opt, metrics=['mae'])
    #
    #
    # #dqn.load_weights("test")
    # history = dqn.fit(env, nb_max_episode_steps=50, nb_steps=500000, visualize=False, verbose=2 )
    #
    # print(history.history.keys())
    #
    # train_rewards = history.history['episode_reward']
    # #dqn.save_weights("test")
    #
    #
    # rewards = np.array_split(train_rewards, len(train_rewards)/100)
    # #print(rewards)
    # count = 100
    # x=[]
    # print("***********Average reward per thousand episodes********\n")
    # for r in rewards:
    #     x.append(sum(r/len(r)))
    #     print(count, ": ", str(sum(r/len(r))))
    #     count += 100
    #
    # #plt.xlim([0,1000])
    # plt.plot(x)
    # plt.show()
    #
    #
    #
    # #x = dqn.test(env, nb_episodes=5, visualize=False)
    #
