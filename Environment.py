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
        self.actionSpace = {0: -self.columns, 1: self.columns, 2: -1, 3: 1}


        #self.observation_space = [i for i in range(self.rows*self.columns)]
        #self.observation_space = np.array((self.rows,self.columns))
        #self.observation_space = self.columns, self.rows


        #start state
        self.col_pos = 1
        self.row_pos = 1
        self.agent_pos = 0

        #Olika typer av varor där en karaktär representerar en vara
        self.wares = "ABCDEFGHIJKL"

        #Varorna läggs i denna array
        #self.order = ['A','B','C','D','E','F','G','H']
        self.order_size = order_size

        #Antal steg som agenten tagit
        self.steps = 0

    def set_order(self, order_size):

        for i in range(order_size):
            ware = random.choice(self.wares)
            self.order.append(ware)



    def get_agent_row_and_column(self):

        x = self.agent_pos // self.columns
        y = self.agent_pos % self.rows
        return x,y

    def get_agent_pos(self):
        return self.row_pos * self.rows + self.col_pos

    def off_grid_move(self, new_state, old_state):
        if new_state not in self.observation_space:
            return True
        elif old_state % self.columns and new_state % self.columns == self.columns-1:
            return True
        elif old_state % self.columns == self.columns-1 and new_state % self.columns == 0:
            return True
        else:
            return False

    def step(self, action):

        self.steps += 1
        x,y = self.get_agent_row_and_column()
        action_res = self.agent_pos + self.actionSpace[action]

        if not self.off_grid_move(action_res, self.agent_pos):
            self.agent_pos = action_res

        ######### REWARD ###########


        current_pos = self.layout[y][x]
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

        #return self.row_pos, self.col_pos, reward, finished_status, info
        return self.agent_pos, reward, finished_status, info

    def reset(self):

        #self.row_pos = 1
        #self.col_pos = 1
        self.agent_pos = 0
        self.set_order()
        self.steps = 0

        return self.agent_pos
        #return self.get_agent_pos()

    def render(self):


        #Agent representeras med '*'

        x,y = self.get_agent_row_and_column()
        #current_row = self.layout[self.row_pos]
        current_row = self.layout[y]
        #new_row = current_row[:self.col_pos] + '*' + current_row[self.col_pos + 1:]
        new_row = current_row[:x] + '*' + current_row[x + 1:]
        #self.layout[self.row_pos] = new_row
        self.layout[y] = new_row

        self.print()

        #self.layout[self.row_pos] = current_row
        self.layout[y] = current_row


    def print(self):

        print("+-------------+")


        for i in range(self.rows):

            print(self.layout[i])


        print("+-------------+")


    def get_reward(self):

        return 1/self.steps
