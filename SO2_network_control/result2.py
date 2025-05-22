import numpy as np
from network_simulator_v4 import NetworkSimulatorV4

m = np.array([0.7, 0.2, 0.2])
l = np.array([0.1, 0.1])
g = 9.81

dt = 0.001
tf = 10

Q_0 = np.array([[0,    0],
                [0, -0.1],
                [0, -0.1]])
Q_dot_0 = np.array([[0, 0],
                    [0, 0],
                    [0, 0]])

n = 3

D = np.array([[-1, -1],
              [ 1,  0],
              [ 0,  1]])

H = np.array([[0, 0],
              [1, 0],
              [0, 1]])

k_theta = 10
k_omega = 10

Qe_d = lambda t: np.array([
    l[0]*np.array([-1, 0]),
    l[1]*np.array([1, 0])
])
Qe_d_dot = lambda t: np.array([
    l[0]*np.array([0, 0]),
    l[1]*np.array([0, 0])
])
Qe_d_ddot = lambda t: np.array([
    l[0]*np.array([0, 0]),
    l[1]*np.array([0, 0])
])

f_l = lambda t: np.array([
    0,
    0.5*np.cos(np.pi*t)
])

sim = NetworkSimulatorV4(n, m, l, g, D, dt, tf, Q_0, Q_dot_0, f_l, Qe_d, Qe_d_dot, Qe_d_ddot, k_theta, k_omega, "result2")
sim.run()
sim.generate_plots(ylim=(-0.11, 0.11))
# sim.generate_animation_v2("Fig. 2c", ((-0.3, 0.3), (-0.4, 0.2)), arrows_bool=False)
# sim.generate_animation_v2("Fig. 2c", ((-0.3, 0.3), (-0.4, 0.2)), arrows_bool=True)