
import numpy as np
from gym import Env
from gym.spaces import Discrete,Box
import random
from enum import Enum


class Actions(Enum):
    UPP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3



class Warehouse(Env):

    def __init__(self, layout):


        self.layout = layout
        self.rows = len(self.layout)
        self.columns = len(self.layout[0])
        self.size = self.rows*self.columns


        self.action_space = Discrete(4)

        self.observation_space = Box(low=np.array([0]),high=np.array([self.rows*self.columns]),dtype=int)

        #start state
        self.col_pos = 0
        self.row_pos = 0
        self.agent_pos = 0


        #Objects that should be picked up, they should match the objects in the chosen environment

        #self.order = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
        #self.order = ['A','B','C']
        #self.order = ['A', 'B', 'C', 'D', 'E']
        self.order = ['A']



    def step(self, action):


        self.steps += 1

        if action == Actions.UPP.value:
            if self.row_pos == 0:
                pass

            else:
                self.row_pos = self.row_pos - 1
                self.agent_pos = self.agent_pos - self.columns
        if action == Actions.RIGHT.value:
            if self.col_pos == self.columns -1 :
                pass
            else:
                self.col_pos = self.col_pos + 1
                self.agent_pos = self.agent_pos + 1

        if action == Actions.DOWN.value:
            if self.row_pos == self.rows - 1:
                pass
            else:
                self.row_pos = self.row_pos + 1
                self.agent_pos = self.agent_pos + self.columns

        if action == Actions.LEFT.value:
            if self.col_pos == 0:
                pass
            else:
                self.col_pos = self.col_pos - 1
                self.agent_pos = self.agent_pos - 1

        current_pos = self.layout[self.row_pos][self.col_pos]

        ######### REWARD ###########



        if current_pos in self.order:
            reward = 1
            self.order.remove(current_pos)
        else:
            reward = 0

        ######## CHECK IF DONE ########

        if not self.order:
            finished_status = True
        else:
            finished_status = False

        info = {}

        return self.agent_pos, reward, finished_status, info

    def reset(self):


        self.row_pos = 0
        self.col_pos = 0
        self.agent_pos = 0
        #self.order = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
        #self.order = ['A', 'B', 'C', 'D', 'E']
        #self.order = ['A','B','C']
        self.order = ['A']
        self.steps = 0

        return self.agent_pos

    def render(self):


        #Agent represented with '*'

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

