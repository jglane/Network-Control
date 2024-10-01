import numpy as np
from network_simulator_v2 import NetworkSimulatorV2

m = np.array([7, 2, 2])*0.1
l = np.array([1, 1])*0.1
g = 9.81

dt = 0.01
tf = 10
t = np.arange(0, tf, dt)

Q_0 = np.array([[0, 0], [0, -1], [0, -1]])*0.1
Q_dot_0 = np.array([[0, 0], [0, 0], [0, 0]])

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

sim = NetworkSimulatorV2(n, k, m, l, g, D_of_G, H, dt, tf, Q_0, Q_dot_0, f_l, Qe_d, Qe_d_dot, Qe_d_ddot, kp, kd)
sim.run()
# sim.generate_plots("result2", ylim=(-0.11, 0.11))
sim.generate_animation("result2", "Fig. 2c", ((-0.3, 0.3), (-0.4, 0.2)))
# sim.generate_animation_v2("result2", "Fig. 2c", ((-0.3, 0.3), (-0.4, 0.2)))