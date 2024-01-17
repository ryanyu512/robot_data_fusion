from sim import *

#initialise time setting
dt = 0.01
end_t = 200

#initialise state
init_speed = 2
init_x = np.random.randn()
init_y = np.random.randn()
init_psi = np.random.uniform(low = np.deg2rad(-180), high = np.deg2rad(180))

#initialise initial uncertainty
init_vel_std = 3
init_psi_std = np.deg2rad(45)

#initialise acceleration uncertainy
acc_std = 1.

#initialise sensor noise standard deviation
gyro_std = np.deg2rad(0.01)
gps_pos_std = 3.

#initialise beacon number
beacon_num = 1500

#initialise gps sensor
gps = gps_sensor()

#initialise lidar sensor
lidar = lidar_sensor()

sim(dt, 
    end_t, 
    init_x, 
    init_y, 
    init_psi, 
    init_speed, 
    init_vel_std,
    init_psi_std,
    acc_std,
    gyro_std,
    gps,
    lidar,
    beacon_num, )