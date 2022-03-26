import numpy as np
import Environment
import random

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


if __name__ == '__main__':



    #print(len(ENV2))
    #print(len(ENV2[0]))


    env = Environment.Warehouse(ENV2)
    env.set_order(5)



    episodes = 10
    for episodes in range(1, episodes + 1):
        col_pos, row_pos = env.reset()
        done = False
        score = 0

        while not done:

            env.render()
            action = env.action_space.sample()
            col_pos, row_pos, reward, done, info = env.step(action)
            score += reward

        print('Episodes: {} Score: {}'.format(episodes, score))