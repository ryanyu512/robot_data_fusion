import numpy as np

from vehicle2d import *
from sensor import *
from ekf import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def sim(dt, 
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
        beacon_num, 
        area_h = 400,
        area_w = 400,
        is_animate = True,
        is_save_gif = False,
):

    #initialise simulation step
    sim_steps = np.ceil(end_t/dt).astype(int)
    gps_meas_steps = np.ceil(1/gps.meas_rate/dt).astype(int)
    lidar_meas_steps = np.ceil(1/lidar.meas_rate/dt).astype(int)

    #initialise vehicle model
    vehicle_model = vehicle2d()
    vehicle_model.init_state(init_x, init_y, init_psi, init_speed)

    #ramdomly generate beacon points
    beacon_list = [] 
    bx = np.random.uniform(low = -area_w, high = area_w, size = beacon_num)
    by = np.random.uniform(low = -area_h, high = area_h, size = beacon_num)
    for i in range(beacon_num):
        b = beacon_point()
        b.x = bx[i]
        b.y = by[i]
        beacon_list.append(b)

    #initialise kalman filter
    kf_model = ekf( init_vel_std = init_vel_std,
                    init_psi_std = init_psi_std,
                    acc_std = acc_std, 
                    gyro_std = gyro_std, 
                    gps_pos_std = gps.pos_std,
                    lidar_range_std = lidar.range_std, 
                    lidar_theta_std = lidar.theta_std,
                    gps_meas_rate= gps.meas_rate)

    #initialise innovation history
    inno_hist = []

    #initialise estimation error history
    est_err_hist = []

    #initialise kf estimated state history
    est_state_hist = []

    #initialise kf estimated covariance history
    est_cov_hist = []

    #initialise true state history
    true_state_hist = []

    #initialise gps history
    gps_history = []

    #initialise state history
    state_history = []

    #initialise beacon location history
    beacon_loc_history = []
    is_gen_beacon = False

    #initialise motion type
    motion_type = None
    curve_type = None
    motion_cnt = 0

    for i in range(1, sim_steps + 1):

        #================== randomly choose motion type ==================#
        if motion_type is None:
            motion_type = np.random.binomial(1, 0.5, 1)[0]
            curve_type = np.random.binomial(1, 0.5, 1)[0]

        if  motion_type == 0:
            if curve_type == 0:
                psi_dot = np.deg2rad( 3)
            else:
                psi_dot = np.deg2rad(-3)
        else:
            psi_dot = np.deg2rad(0)

        motion_cnt += 1

        if motion_cnt > 2000:
            motion_cnt = 0
            motion_type = None
            curve_type = None

        #================== update vehicle state ==================#
        vehicle_model.update(dt, 0., psi_dot)
        state = vehicle_model.get_state()

        #================== kf prediction ==================#
        #assume measurement rate of gyroscope is the same as simulation rate
        meas_psi_dot = psi_dot + np.random.normal(loc = 0, scale = gyro_std)
        kf_model.prediction_step(psi_dot = meas_psi_dot, dt = dt)

        #================== gps measurement ==================#
        gps_measurement = None
        if (i % gps_meas_steps) == 0:
            gps.x = state[0, 0] + np.random.normal(loc = 0, scale = gps.pos_std)
            gps.y = state[1, 0] + np.random.normal(loc = 0, scale = gps.pos_std)

            gps_measurement = [gps.x, gps.y]
            kf_model.gps_update_step(gps)
        gps_history.append(gps_measurement)

        #================== lidar measurement ==================#
        detected_beacon = [] 
        if (i % lidar_meas_steps) == 0:
            for b in beacon_list:  

                #check if any beacon is detected
                dx, dy = b.x - state[0, 0], b.y - state[1, 0]
                beacon_range = np.sqrt(dx**2 + dy**2)
                if beacon_range > lidar.max_range:
                    continue
                
                #simulate lidar measurement
                beacon_theta = wrap_ang(np.arctan2(dy, dx) - state[2, 0]) 
                detected_beacon.append([b.x, b.y])

                lidar.range = beacon_range + np.random.normal(loc = 0, scale = lidar.range_std)
                lidar.theta = wrap_ang(beacon_theta + np.random.normal(loc = 0, scale = lidar.theta_std))

                kf_model.lidar_update_step(lidar, b)    
        else:
            #check if any beacon is still within detection range
            if i > 1 and len(beacon_loc_history[-1]) > 0:
                for b in beacon_loc_history[-1]:
                    dx, dy = b[0] - state[0, 0], b[1] - state[1, 0]
                    beacon_range = np.sqrt(dx**2 + dy**2)
                    if beacon_range <= lidar.max_range:
                        detected_beacon.append([b[0], b[1]])

        beacon_loc_history.append(detected_beacon)

        #================== record history for display ==================#
        true_state_hist.append(state)
        est_state_hist.append(kf_model.get_state())
        est_cov_hist.append(kf_model.get_cov())

        state_diff = None
        if est_state_hist[-1] is not None:
            state_diff = (est_state_hist[-1] - state)[:, 0]
            state_diff[2] = wrap_ang(state_diff[2])
        est_err_hist.append(state_diff)


    #================== error plot ==================#
    time_history = np.linspace(0.0, dt*sim_steps, sim_steps+1)

    est_err_hist_plot = np.array([_ for _ in est_err_hist if _ is not None])
    est_cov_hist_plot = np.array([_ for _ in est_cov_hist if _ is not None])
    time_plot = np.array([t for (t, e) in zip(time_history, est_err_hist) if e is not None])

    fig, ax = plt.subplots(4)
    fig.tight_layout()
    for i in range(est_err_hist_plot.shape[1]):
        ax[i].plot(time_plot, est_err_hist_plot[:, i])
        ax[i].plot(time_plot,  3*np.sqrt(est_cov_hist_plot[:, i, i]))
        ax[i].plot(time_plot, -3*np.sqrt(est_cov_hist_plot[:, i, i]))

    ax[0].set_ylabel('error (m)')
    ax[1].set_ylabel('error (m)')
    ax[2].set_ylabel('error (rad)')
    ax[3].set_ylabel('error (m/s)')
    ax[3].set_xlabel('time (s)')
    

    #================== animation ==================#
    fig2 = plt.figure()
    ax2  = plt.subplot(1, 1, 1)

    def init_func():
        ax2.clear()
        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')

    def update_plot(i):

        #clear plot
        ax2.clear()

        #plot true position
        ax2.plot(true_state_hist[i][0,0], true_state_hist[i][1,0], 'og')

        #plot estimate position
        if est_state_hist[i] is not None:
            ax2.plot(est_state_hist[i][0,0], est_state_hist[i][1,0], 'or')

            #plot uncertainty
            if est_cov_hist[i] is not None:
                cov = est_cov_hist[i]
                cov_mat = np.array([[cov[0][0],cov[0][1]],[cov[1][0],cov[1][1]]])
                U, S, V = np.linalg.svd(cov_mat)

                theta = np.linspace(0, 2*np.pi, 100)
                theta_mat = np.array([np.cos(theta),np.sin(theta)])
                D = np.matmul(np.matmul(U,np.diag(3.0*np.sqrt(S))),theta_mat)
                ax2.plot([x+est_state_hist[i][0,0] for x in D[0]], [y+est_state_hist[i][1,0] for y in D[1]], 'g-')

        ax2.legend(['true robot', 'estimated robot'])

        #plot heading
        ax2.plot([true_state_hist[i][0,0], true_state_hist[i][0,0] + 2*np.cos(true_state_hist[i][2,0])], 
                [true_state_hist[i][1,0], true_state_hist[i][1,0] + 2*np.sin(true_state_hist[i][2,0])], '-g')
        if est_state_hist[i] is not None:
            ax2.plot([est_state_hist[i][0,0], est_state_hist[i][0,0] + 2*np.cos(est_state_hist[i][2,0])], 
                    [est_state_hist[i][1,0], est_state_hist[i][1,0] + 2*np.sin(est_state_hist[i][2,0])], '-r')
            
        #plot gps history
        gps_data = np.array([m for m in gps_history[0:i+1] if m is not None])
        if gps_data is not None and len(gps_data) > 0:
            ax2.plot(gps_data[:, 0], gps_data[:, 1], '+k')

        #plot detected beacon location
        b = np.array(beacon_loc_history[i])
        if len(b) > 0:
            ax2.plot(b[:, 0], b[:, 1], 's')

            #plot linkage between robot and beacon
            for k in range(len(b)):
                x = [true_state_hist[i][0, 0], b[k, 0]]
                y = [true_state_hist[i][1, 0], b[k, 1]]
                ax2.plot(x, y, 'r-')

        ax2.set_xlim([true_state_hist[i][0, 0] - 20., true_state_hist[i][0, 0] + 20.])
        ax2.set_ylim([true_state_hist[i][1, 0] - 20., true_state_hist[i][1, 0] + 20.])
        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')

        #ax2.relim()
        #ax2.autoscale_view()

    anim = FuncAnimation(fig2, 
                        update_plot,
                        frames = np.arange(0, sim_steps, 10), 
                        init_func = init_func,
                        interval = 1,
                        repeat = False)

    if is_animate:
        plt.show()

    if is_save_gif:
        writer = animation.PillowWriter(fps=15,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        anim.save('demo.gif', writer=writer)

