import shutil
import os
import numpy as np
from numpy import array as npa


homeDir = os.path.expanduser("~")
originalSPEED_dir = homeDir + '/SPEED'
mySPEED_dir = homeDir + '/SPEED_MP'


def createDirectory(myDir):
    if os.path.exists(myDir):
        shutil.rmtree(myDir)
    os.mkdir(myDir)

setOrder = ['train', 'dev', 'test']



class Wireframe:
    """"
    Utility class that defines landmarks and coordinates
    of other points from the wireframe model.
    """

    # Define 11 landmark points (body coordinates)
    landmarks = [
    {'label': 'B1', 'r_B': [-0.37, 0.30, 0]},
    {'label': 'B2', 'r_B': [-0.37, -0.26, 0]},
    {'label': 'B3', 'r_B': [0.37, -0.26, 0]},
    {'label': 'B4', 'r_B': [0.37, 0.30, 0]},
    {'label': 'S1', 'r_B': [-0.37, 0.38, 0.32]},
    {'label': 'S2', 'r_B': [-0.37, -0.38, 0.32]},
    {'label': 'S3', 'r_B': [0.37, -0.38, 0.32]},
    {'label': 'S4', 'r_B': [0.37, 0.38, 0.32]},
    {'label': 'A1', 'r_B': [-0.54, 0.49, 0.255]},
    {'label': 'A2', 'r_B': [0.31, -0.56, 0.255]},
    {'label': 'A3', 'r_B': [0.54, 0.49, 0.255]}
        ]

    landmark_mat = np.column_stack( [point['r_B'] for point in landmarks] )

    # Top of the main body (not used as landmarks)
    topMainBody = [
    {'label': 'T1', 'r_B': [-0.37, 0.30, 0.305]},
    {'label': 'T2', 'r_B': [-0.37, -0.26, 0.305]},
    {'label': 'T3', 'r_B': [0.37, -0.26, 0.305]},
    {'label': 'T4', 'r_B': [0.37, 0.30, 0.305]}
        ]
    topMainBody_mat = np.column_stack( [point['r_B'] for point in topMainBody] )

    # Antenna clamps
    antClamps = [
    {'label': 'Ac1', 'r_B': [-0.23, 0.3, 0.255]},
    {'label': 'Ac2', 'r_B': [0.31, -0.26, 0.255]},
    {'label': 'Ac3', 'r_B': [0.23, 0.3, 0.255]}
        ]
    antClamps_mat = np.column_stack( [point['r_B'] for point in antClamps] )




# reference points @ body frame (for drawing axes)
p_axes = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])



class Camera:
    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length [m]
    fy = 0.0176  # focal length [m]
    nu = 1920  # no. horizontal pixels
    nv = 1200  # no. of vertical pixels
    ppx = 5.86e-6  # horizontal pixel pitch [m/px]
    ppy = ppx      # vertical pixel pitch [m/px]
    fpx = fx / ppx  # horizontal focal length [px]
    fpy = fy / ppy  # vertical focal length [px]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = npa(k)



def quat2dcm(q):

    """ Convert quaternion to Direction Cosine Matrix. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0**2 - 1 + 2 * q1**2
    dcm[1, 1] = 2 * q0**2 - 1 + 2 * q2**2
    dcm[2, 2] = 2 * q0**2 - 1 + 2 * q3**2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm