import string

import numpy
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

    def __init__(self, layout, order_size):


        self.layout = layout
        self.rows = len(self.layout)
        self.columns = len(self.layout[0])
        self.grid = np.zeros((self.rows,self.columns))
        self.size = self.rows*self.columns

        # Antal actions som kan tas
        self.action_space = Discrete(4)
        self.obs = [i for i in range(self.rows*self.columns)]

        self.observation_space = Box(low=0,high=self.rows*self.columns, shape=(1,))
        #self.observation_space = np.array((self.rows,self.columns))
        #self.observation_space = self.columns, self.rows


        #start state
        self.col_pos = 0
        self.row_pos = 0
        self.agent_pos = 0

        #Olika typer av varor där en karaktär representerar en vara
        self.wares = "ABCDEFGHIJKL"

        #Varorna läggs i denna array
        self.order = []
        self.order_size = order_size

        #Antal steg som agenten tagit
        self.steps = 0

    def set_order(self, order_size):

        for i in range(order_size):
            ware = random.choice(self.wares)
            self.order.append(ware)

    def get_agent_pos(self):
        col_pos = self.agent_pos // self.columns
        row_pos = self.agent_pos % self.rows
        return row_pos*col_pos

    def step(self, action):

        self.steps += 1

        if action == Actions.UPP.value:
            if self.row_pos == 0:
                pass
            else:
                self.row_pos -= 1
                self.agent_pos += -self.columns
        if action == Actions.RIGHT.value:
            if self.col_pos == self.columns -1 :
                pass
            else:
                self.col_pos += 1
                self.agent_pos += 1

        if action == Actions.DOWN.value:
            if self.row_pos == self.rows - 1:
                pass
            else:
                self.row_pos += 1
                self.agent_pos += self.columns

        if action == Actions.LEFT.value:
            if self.col_pos == 0:
                pass
            else:
                self.col_pos -= 1
                self.agent_pos -=1

        ######### REWARD ###########

        current_pos = self.layout[self.row_pos][self.col_pos]
        if current_pos in self.order:
            reward = self.get_reward()


            self.order.remove(current_pos)

        else:
            reward = 0

        ######## CHECK IF DONE ########

        if not self.order:
            finished_status = True
        else:
            finished_status = False

        info = {}

        #state = self.get_agent_pos()
        return self.agent_pos, reward, finished_status, info

    def reset(self):

        self.row_pos = 0
        self.col_pos = 0
        self.agent_pos = 0
        self.set_order(self.order_size)
        self.steps = 0

        return self.agent_pos

    def render(self):


        #Agent representeras med '*'

        current_row = self.layout[self.row_pos]
        new_row = current_row[:self.col_pos] + '*' + current_row[self.col_pos + 1:]
        self.layout[self.row_pos] = new_row

        self.print()

        self.layout[self.row_pos] = current_row


    def print(self):

        print("+-------------+")


        for i in range(self.rows):

            print(self.layout[i])


        print("+-------------+")


    def get_reward(self):

        return 1/self.steps
