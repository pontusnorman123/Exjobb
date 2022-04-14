import keras
import numpy as np

import Environment
import ENV2
import random
import sys,os

os.add_dll_directory("C:/Program/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin")

from keras.models import Sequential
from keras.layers import Dense, Flatten,Input
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import tensorflow as tf
from matplotlib import pyplot as plt
import datetime



ENV3 = [
    "+-------------------+",
    "|S                  |",
    "|    A   B    C     |",
    "|    D   E    F     |",
    "|    G   H    I     |",
    "|    J   K    L     |",
    "+-------------------+"
]

ENV = [
    "+-------------------+",
    "|S                  |",
    "|    A   B    C     |",
    "|    D   E    F     |",
    "|    G   H    I     |",
    "|    J   K    L     |",
    "+-------------------+"
]

ENV = [

    "S                  ",
    "    A   B    C     ",
    "    D   E    F     ",
    "    G   H    I     ",
    "    J   K    L     ",
]

ORDER_SIZE = 20

if __name__ == '__main__':


    x = np.array([0,0])
    print(x)



    env = ENV2.Warehouse(ENV, ORDER_SIZE)

    # print(env.rows,env.columns)
    # # print(env.observation_space)
    # #
    episodes = 0
    for episodes in range(1, episodes + 1):
        agent_pos = env.reset()
        #test = env.reset(ORDER_SIZE)
        done = False
        score = 0
        print(env.order)


        while not done:

            print("Agent_pos:",env.agent_pos)
            env.render()
            action = env.action_space.sample()
            print("Action:", action)
            agent_pos, reward, done, info = env.step(action)

            score += reward

        print('Episodes: {} Score: {}'.format(episodes, score))




    actions = env.action_space.n
    print("Number of Actions:", actions) #S: Denna borde vara 4. för det är 4 actions.

    inputs = env.observation_space.shape
    print("Inputs:", inputs) #S: Osäker kring denna... Den borde vara (95) eller (5, 19) eller (7, 21) eller nått annat beroende på hur din observerbara 'karta' ser ut...



    model = Sequential()
    model.add(Dense(units=24, input_shape=inputs, activation='relu'))
    #model.add(Flatten())
    model.add(Dense(units=24, activation='relu'))
    model.add(Dense(units=24, activation='relu'))
    #model.add(Flatten())
    model.add(Dense(units=actions, activation='linear'))
    model.summary()
    print(model.output_shape)

    ######
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)

    opt = tf.keras.optimizers.Adam(learning_rate = 0.1)
    dqn.compile(opt, metrics=['mae'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    history.