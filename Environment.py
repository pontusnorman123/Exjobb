import string

import numpy as np
from gym import Env
from gym.spaces import Discrete,Box
import random
from enum import Enum

class Actions(Enum):
    UPP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class Order:


    def __init__(self, order_size):
        self.order = np.chararray(order_size)
        self.order_types = "abcdefgh"
        for i in self.order:
            self.order[i] = np.random.choice(self.order_types)


class Warehouse(Env):

    def __init__(self, width, height, order_size):

        self.width = width
        self.height = height

        # Antal actions som kan tas
        self.action_space = Discrete(4)

        #start state
        self.long_pos = 0
        self.lat_pos = 0

        #delivery size




    def step(self, action):

        if action == Actions.UPP:
            if self.lat_pos == 0:
                pass
            else:
                self.lat_pos -= 1

        if action == Actions.RIGHT:
            if self.long_pos == self.width:
                pass
            else:
                self.long_pos += 1

        if action == Actions.DOWN:
            if self.lat_pos == self.height:
                pass
            else:
                self.lat_pos += 1

        if action == Actions.LEFT:
            if self.long_pos == 0:
                pass
            else:
                self.long_pos -= 1


        ######### REWARD ###########

