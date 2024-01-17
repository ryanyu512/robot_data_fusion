import numpy as np

class gps_sensor():
    def __init__(self):
        self.x = None
        self.y = None
        self.meas_rate = 1
        self.pos_std = 3

class beacon_point():
    def __init__(self):
        self.x = None
        self.y = None

class lidar_sensor():
    def __init__(self, max_range = 20):
        self.range = None
        self.theta = None
        self.max_range = max_range
        self.meas_rate = 10
        self.range_std = 3.0
        self.theta_std = 0.02