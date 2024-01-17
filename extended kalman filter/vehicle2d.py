import numpy as np
from util import *

class vehicle2d():

    def __init__(self):
        self.x = 0
        self.y = 0
        self.v = 0
        self.yaw = 0

    def init_state(self, x, y, yaw, v):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v 

    def update(self, dt, acc, psi_dot):
        self.v += acc*dt
        self.yaw = wrap_ang(self.yaw + psi_dot*dt)
        self.x += self.v*np.cos(self.yaw)*dt
        self.y += self.v*np.sin(self.yaw)*dt

    def get_state(self):
        s = np.array([self.x, self.y, self.yaw, self.v])
        s.shape = (4, 1)
        return s