import numpy as np
from util import *

class ekf():
    def __init__(self, 
                 gyro_std = 0.01/180.0*np.pi, 
                 acc_std = 1.0,
                 gps_pos_std = 3.0,
                 init_vel_std = 10.0, 
                 init_psi_std = np.deg2rad(45.),
                 lidar_range_std = 3.0,
                 lidar_theta_std = 0.02,
                 reset_gps_threshold = 5.0,
                 reset_lidar_threshold = 10.,
                 gps_meas_rate = 1.):
        
        self.state = None
        self.cov   = None

        self.gyro_std = gyro_std
        self.gps_pos_std = gps_pos_std
        self.lidar_range_std = lidar_range_std
        self.lidar_theta_std = lidar_theta_std

        self.acc_std  = acc_std
        self.init_vel_std = init_vel_std
        self.init_psi_std = init_psi_std

        self.receive_correct_gps = True
        self.incorrect_gps_cnt = 0
        self.reset_gps_threshold = reset_gps_threshold

        self.prev_pos = None
        self.gps_meas_rate = gps_meas_rate

        self.speed_list = []
        self.avg_speed = None

        self.prev_psi = None


    def get_state(self):
        return self.state
        
    def get_cov(self):
        return self.cov

    def set_state(self, state):
        self.state = state

    def set_cov(self, cov):
        self.cov = cov

    def prediction_step(self, psi_dot, dt):

        if self.state is not None:
            #get current state and covariance matrix
            state = self.get_state()
            cov   = self.get_cov()

            #update state
            x = state[0, 0]
            y = state[1, 0]
            psi = state[2, 0]
            v = state[3, 0]

            #assume acceleration noise mean  = 0 
            state = np.array([x + dt*v*np.cos(psi), 
                              y + dt*v*np.sin(psi), 
                              wrap_ang(psi + dt*psi_dot),
                              v]) 
            state.shape = (4, 1)

            #generate dF matrix (jacobian matrix)
            dF = np.array([ [1, 0, -dt*v*np.sin(psi), dt*np.cos(psi)],
                            [0, 1,  dt*v*np.cos(psi), dt*np.sin(psi)],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

            #generate process noise covariance matrix
            Q = np.zeros((4, 4))
            Q[2, 2] = (dt*self.gyro_std)**2
            Q[3, 3] = (dt*self.acc_std)**2

            #update prior covariance matrix
            cov = np.matmul(np.matmul(dF, cov), np.transpose(dF)) + Q

            self.set_state(state)
            self.set_cov(cov)

    def gps_update_step(self, gps):

        if self.state is not None and self.cov is not None:

            #get current state and covariance matrix
            state = self.get_state()
            cov = self.get_cov()

            #initialise z, H and R matrix
            z = np.array([gps.x, gps.y])
            z.shape = (2, 1)
            H = np.zeros((2, 4))
            R = np.zeros((2, 2))

            H[0, 0] = H[1, 1] = 1.
            R[0, 0] = R[1, 1] = self.gps_pos_std**2

            #compute innovation
            z_h = np.matmul(H, state)
            inno = z - z_h
            inno.shape = (2, 1)

            #compute innovation covariance
            S = np.matmul(np.matmul(H, cov), np.transpose(H)) + R

            #check if the data is good enough for fusion
            e = np.matmul(np.matmul(np.transpose(inno), np.linalg.inv(S)), inno)[0, 0]
            if e < 5.99: #5.99 => chi - squared value for 2 dimensional vector
                #compute optimised kalman filter gain
                K = np.matmul(np.matmul(cov, np.transpose(H)), np.linalg.inv(S))

                #update state and covariance matrix
                state += np.matmul(K, inno)
                cov = np.matmul((np.eye(4) - np.matmul(K, H)), cov)

                #estimate forward speed via subsequence gps signal
                #used for reinitialisation
                if self.prev_pos is not None:
                    dx = state[0, 0] - self.prev_pos[0]
                    dy = state[1, 0] - self.prev_pos[1]
                    dt = 1./self.gps_meas_rate
                    sp = np.sqrt(dx**2 + dy**2)/dt
                    if len(self.speed_list) >= 2:
                        self.speed_list.pop(0)
                    self.speed_list.append(sp)

                    if len(self.speed_list) >= 2:
                        self.avg_speed = np.mean(self.speed_list)

                self.prev_pos = [state[0, 0], state[1, 0]]

                self.receive_correct_gps = True
                self.incorrect_gps_cnt = 0
            else:
                self.receive_correct_gps = False
                self.prev_pos = None
                self.incorrect_gps_cnt += 1
                
                if self.incorrect_gps_cnt >= self.reset_gps_threshold:
                    state = None
                    cov = None
                    self.incorrect_gps_cnt = 0
        else:

            #initialise kalman filter based on gps measurement and average 
            #speed estimation
            
            self.prev_pos = None
            self.speed_list = []

            state = np.zeros((4, 1))
            cov   = np.zeros((4, 4))

            state[0, 0] = gps.x
            state[1, 0] = gps.y
            if self.prev_psi is None:
                state[2, 0] = 0.
            else:
                state[2, 0] = self.prev_psi
            if self.avg_speed is None:
                state[3, 0] = 0.
            else:
                state[3, 0] = self.avg_speed

            cov[0, 0] = self.gps_pos_std**2
            cov[1, 1] = self.gps_pos_std**2
            cov[2, 2] = self.init_psi_std**2
            cov[3, 3] = self.init_vel_std**2

        self.set_state(state)
        self.set_cov(cov)

    def lidar_update_step(self, lidar, beacon):
        
        if self.state is not None and self.cov is not None:
            state = self.get_state()
            cov = self.get_cov()

            x = state[0, 0]
            y = state[1, 0]
            psi = state[2, 0]
            v = state[3, 0]

            #compute z
            z = np.array([lidar.range, lidar.theta])
            z.shape = (2, 1)

            #compute z_hat
            #assume we know which beacon is scanned => a bit cheating here
            dx = beacon.x - x
            dy = beacon.y - y
            est_r = np.sqrt(dx**2 + dy**2)
            est_a = np.arctan2(dy, dx) - psi
            z_hat = np.array([est_r, wrap_ang(est_a)])
            z_hat.shape = (2, 1)

            #compute measurement jacobian matrix        
            dH = np.array([[  -dx/est_r,   -dy/est_r,  0, 0], 
                           [dy/est_r**2,-dx/est_r**2, -1, 0],
                          ])
        
            #compute innovation 
            inno = z - z_hat
            inno.shape = (2, 1)
            
            #compute innovation covariance matrix
            R = np.array([[self.lidar_range_std**2, 0], 
                          [0, self.lidar_theta_std**2]])
            S = np.matmul(np.matmul(dH, cov), np.transpose(dH)) + R
 
            #compute optimised kalman filter gain
            K = np.matmul(np.matmul(cov, np.transpose(dH)), np.linalg.inv(S))

            inno[1] = wrap_ang(inno[1])

            state += np.matmul(K, inno)
            cov = np.matmul((np.eye(4) - np.matmul(K, dH)), cov)

            self.set_state(state)
            self.set_cov(cov)