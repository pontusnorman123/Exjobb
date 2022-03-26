import string

import numpy as np
from gym import Env
from gym.spaces import Discrete,Box
import random
from enum import Enum
from pandas import *

class Actions(Enum):
    UPP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3



class Warehouse(Env):

    def __init__(self, layout):


        self.layout = layout
        self.rows = len(self.layout) - 2
        self.columns = len(self.layout[0]) - 2

        # Antal actions som kan tas
        self.action_space = Discrete(4)
        #self.observation_space = Box()

        #start state
        self.col_pos = 1
        self.row_pos = 1

        #Olika typer av varor där en karaktär representerar en vara
        self.wares = "ABCDEFGHIJKL"

        #Varorna läggs i denna array
        self.order = []

        #Antal steg som agenten tagit
        self.steps = 0

    def set_order(self, order_size):

        for i in range(order_size):
            ware = random.choice(self.wares)
            self.order.append(ware)

    def step(self, action):

        self.steps += 1

        if action == Actions.UPP.value:
            if self.row_pos == 1:
                pass
            else:
                self.row_pos -= 1

        if action == Actions.RIGHT.value:
            if self.col_pos == self.columns:
                pass
            else:
                self.col_pos += 1

        if action == Actions.DOWN.value:
            if self.row_pos == self.rows:
                pass
            else:
                self.row_pos += 1

        if action == Actions.LEFT.value:
            if self.col_pos == 1:
                pass
            else:
                self.col_pos -= 1


        ######### REWARD ###########

        current_pos = self.layout[self.row_pos][self.col_pos]
        if current_pos in self.order:
            reward = self.get_reward()

            #current_row = self.layout[self.row_pos]
            #new_row = current_row[:self.col_pos] + '*' + current_row[self.col_pos + 1:]
            #self.layout[self.row_pos] = new_row

            self.order.remove(current_pos)

        else:
            reward = 0

        ######## CHECK IF DONE ########

        if not self.order:
            finished_status = True
        else:
            finished_status = False

        info = {}

        return self.row_pos, self.col_pos, reward, finished_status, info

    def reset(self, order_size):

        self.row_pos = 1
        self.col_pos = 1
        self.set_order(order_size)
        self.steps = 0

        return self.row_pos, self.col_pos

    def render(self):

        #Agent representeras med '*'

        current_row = self.layout[self.row_pos]
        new_row = current_row[:self.col_pos] + '*' + current_row[self.col_pos + 1:]
        self.layout[self.row_pos] = new_row

        self.print()

        self.layout[self.row_pos] = current_row


    def print(self):

        for i in range(self.rows + 2):

            print(self.layout[i])


    def get_reward(self):

        return 1/self.steps
