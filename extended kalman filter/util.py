import numpy as np

def wrap_ang(ang):

    ang = ang % (2.*np.pi)

    if ang <= -np.pi:
        ang += 2.*np.pi
    elif ang > np.pi:
        ang -= 2.*np.pi

    return ang