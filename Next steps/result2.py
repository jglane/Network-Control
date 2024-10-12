import numpy as np
from scipy.spatial.transform import Rotation
from network_simulator_v3 import NetworkSimulatorV3

m = np.array([7, 2, 2])*0.1
l = np.array([1, 1])*0.1
g = 9.81

dt = 0.01
tf = 10
t = np.arange(0, tf, dt)

Q_0 = np.array([[0, 0, 0], [0.1, 0, 0], [-0.1, 0, 0]])
Q_dot_0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

n = 3
k = n-1

D_of_G = np.array([[-1, -1],
                   [ 1,  0],
                   [ 0,  1]])

H = np.array([[0, 0],
              [1, 0],
              [0, 1]])

kp = 10
kd = 10

alpha = lambda t: 0
beta = lambda t: np.pi/4*np.sin(np.pi*t)
gamma = lambda t: np.pi/2*t

# R = lambda a, b, c: np.array([[np.cos(a)*np.cos(b), np.cos(a)*np.sin(b)*np.sin(c) - np.sin(a)*np.cos(c), np.cos(a)*np.sin(b)*np.cos(c) + np.sin(a)*np.sin(c)],
                                # [np.sin(a)*np.cos(b), np.sin(a)*np.sin(b)*np.sin(c) + np.cos(a)*np.cos(c), np.sin(a)*np.sin(b)*np.cos(c) - np.cos(a)*np.sin(c)],
                                # [-np.sin(b), np.cos(b)*np.sin(c), np.cos(b)*np.cos(c)]])

# def _hat(v):
#     return np.array([[0, -v[2], v[1]],
#                      [v[2], 0, -v[0]],
#                      [-v[1], v[0], 0]])

# alpha_dot = lambda t: -np.pi**2/4*np.sin(np.pi*t)
# beta_dot = lambda t: -np.pi**2/4*np.sin(np.pi*t)
# gamma_dot = lambda t: 0
# omega = lambda t: np.array([alpha_dot(t), beta_dot(t), gamma_dot(t)])
# R_dot = lambda a, b, c, a_dot, b_dot, c_dot: R(a, b, c)@_hat(np.array([a_dot, b_dot, c_dot]))

# R_ddot = lambda a, b, c, a_dot, b_dot, c_dot, a_ddot, b_ddot, c_ddot: R_dot(a, b, c, a_dot, b_dot, c_dot)@np.array([a_ddot, b_ddot, c_ddot])

def R(alpha, beta, gamma, t, multipliers=[1, 1, 1]):
    return Rotation.from_euler('xyz', [multipliers[0]*alpha(t), multipliers[1]*beta(t), multipliers[2]*gamma(t)], degrees=False).as_matrix()

def R_dot(alpha, beta, gamma, t, multipliers=[1, 1, 1]):
    R = Rotation.from_euler('xyz', [multipliers[0]*alpha(t), multipliers[1]*beta(t), multipliers[2]*gamma(t)], degrees=False).as_matrix()
    R_prev = Rotation.from_euler('xyz', [multipliers[0]*alpha(t-dt), multipliers[1]*beta(t-dt), multipliers[2]*gamma(t-dt)], degrees=False).as_matrix()
    return (R - R_prev)/dt

def R_ddot(alpha, beta, gamma, t, multipliers=[1, 1, 1]):
    R = Rotation.from_euler('xyz', [multipliers[0]*alpha(t), multipliers[1]*beta(t), multipliers[2]*gamma(t)], degrees=False).as_matrix()
    R_prev = Rotation.from_euler('xyz', [multipliers[0]*alpha(t-dt), multipliers[1]*beta(t-dt), multipliers[2]*gamma(t-dt)], degrees=False).as_matrix()
    R_dot = (R - R_prev)/dt
    R_2prev = Rotation.from_euler('xyz', [multipliers[0]*alpha(t-2*dt), multipliers[1]*beta(t-2*dt), multipliers[2]*gamma(t-2*dt)], degrees=False).as_matrix()
    R_dot_prev = (R_prev - R_2prev)/dt
    return (R_dot - R_dot_prev)/dt

def Qe_d(t):
    R1 = R(alpha, beta, gamma, t)
    R2 = R(alpha, beta, gamma, t, multipliers=[1, -1, 1])
    return np.array([
        l[0]*R1@np.array([-1, 0, 0]),
        l[1]*R2@np.array([1, 0, 0])
    ])

def Qe_d_dot(t):
    R1_dot = R_dot(alpha, beta, gamma, t)
    R2_dot = R_dot(alpha, beta, gamma, t, multipliers=[1, -1, 1])
    return np.array([
        l[0]*R1_dot@np.array([-1, 0, 0]),
        l[1]*R2_dot@np.array([1, 0, 0])
    ])

def Qe_d_ddot(t):
    R1_ddot = R_ddot(alpha, beta, gamma, t)
    R2_ddot = R_ddot(alpha, beta, gamma, t, multipliers=[1, -1, 1])
    return np.array([
        l[0]*R1_ddot@np.array([-1, 0, 0]),
        l[1]*R2_ddot@np.array([1, 0, 0])
    ])

f_l = lambda t: np.array([
    0,
    0,
    0.5*np.cos(np.pi*t)
])

sim = NetworkSimulatorV3(n, k, m, l, g, D_of_G, H, dt, tf, Q_0, Q_dot_0, f_l, Qe_d, Qe_d_dot, Qe_d_ddot, kp, kd, "result2")
sim.run()
sim.generate_plots(ylim=(-0.11, 0.11))
sim.generate_animation_v2("Fig. 2c", ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)), arrows_bool=True)