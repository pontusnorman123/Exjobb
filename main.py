import numpy as np
import Environment
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


ENV1 = [
    ["+-------------------+"],
    ["|S                  |"],
    ["| A : : : : : : : : |"],
    ["| : : : : : : : : : |"],
    ["| B : : : : : : : : |"],
    ["| : : : : : : : : : |"],
    ["| C : : : : : : : : |"],
    ["+-------------------+"],
]

ENV2 = [
    "+-------------------+",
    "|S                  |",
    "|                   |",
    "|  A                |",
    "|                   |",
    "|  B                |",
    "|  C                |",
    "+-------------------+"
]

ENV3 = [
    "+-------------------+",
    "|S                  |",
    "|    A   B    C     |",
    "|    D   E    F     |",
    "|    G   H    I     |",
    "|    J   K    L     |",
    "+-------------------+"
]

ORDER_SIZE = 20

if __name__ == '__main__':



    #print(len(ENV2))
    #print(len(ENV2[0]))


    env = Environment.Warehouse(ENV3)


    episodes = 1000
    for episodes in range(1, episodes + 1):
        col_pos, row_pos = env.reset(ORDER_SIZE)
        done = False
        score = 0
        #print(env.order)


        while not done:

            #env.render()
            action = env.action_space.sample()
            #print(action)
            col_pos, row_pos, reward, done, info = env.step(action)
            score += reward

        print('Episodes: {} Score: {}'.format(episodes, score))