
![demo](https://github.com/ryanyu512/robot_data_fusion/assets/19774686/4a430c5a-55ef-46d7-9b0f-1b54039524a6)

This project explores extended kalman filter to fuse noisy information from GPS, gyroscope and lidar to track the true state (x, y, heading and forward velocity) of 2D vehicle. In the above demo, "+" indicates noisy GPS while sq shape represents beacon location scanned by lidar. 

1. demo.py: used for testing different parameters of kalman filter
2. sensor.py: define the properties of different sensors
3. vehicle2d.py: define the properties of 2d vehicle
4. sim.py: custom function for running simulation
5. util.py: provide custom tools
