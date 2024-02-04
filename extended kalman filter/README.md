![demo](https://github.com/ryanyu512/robot_data_fusion/assets/19774686/7c5f6808-74b0-4991-a02d-83507a01c32a)

This project explores extended kalman filter (EKF) to fuse noisy information from GPS, gyroscope and lidar to track the true state (x, y, heading and forward velocity) of 2D vehicle. In the above demo, "green dot" represents true robot, "red dot" represents states estimated by EKF, "+" indicates noisy GPS and "square" represents beacon location scanned by lidar. 

1. demo.py: used for testing different parameters of kalman filter
2. sensor.py: define the properties of different sensors
3. vehicle2d.py: define the properties of 2d vehicle
4. sim.py: custom function for running simulation
5. util.py: provide custom tools
