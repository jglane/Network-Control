import numpy as np
from network_simulator_v4 import NetworkSimulatorV4

m = np.array([7, 2, 2, 5, 1, 1])*0.1
l = np.array([3, 3, 3, 3, 3])*0.1
g = 9.81

dt = 0.01
tf = 10

Q_0 = np.array([[0, 0], [0, -3], [0, -3], [0, 3], [0, 6], [0, 6]])*0.1
Q_dot_0 = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

n = 6

D = np.array([[-1, -1, -1,  0,  0],
              [ 1,  0,  0,  0,  0],
              [ 0,  1,  0,  0,  0],
              [ 0,  0,  1, -1, -1],
              [ 0,  0,  0,  1,  0],
              [ 0,  0,  0,  0,  1]])

H = np.array([[0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]]).T

kp = 10
kd = 32

fact = 1
theta_d = lambda t: 3*np.pi/16*np.cos(fact*np.pi*t) + np.pi/16
theta_d_dot = lambda t: -3*np.pi/16*np.sin(fact*np.pi*t)*fact*np.pi
theta_d_ddot = lambda t: -3*np.pi/16*np.cos(fact*np.pi*t)*(fact*np.pi)**2
# theta_d = lambda t: np.pi/4
# theta_d_dot = lambda t: 0
# theta_d_ddot = lambda t: 0
# theta_d = lambda t: np.pi/4*np.cos(np.pi*t)
# theta_d_dot = lambda t: -np.pi**2/4*np.sin(np.pi*t)
# theta_d_ddot = lambda t: -np.pi**3/4*np.cos(np.pi*t)
Qe_d = lambda t: np.array([
    l[0]*np.array([-np.cos(theta_d(t)), -np.sin(theta_d(t))]),
    l[1]*np.array([np.cos(theta_d(t)), -np.sin(theta_d(t))]),
    l[2]*np.array([0, 1]),
    l[3]*np.array([-np.cos(theta_d(t)), np.sin(theta_d(t))]),
    l[4]*np.array([np.cos(theta_d(t)), np.sin(theta_d(t))])
])
Qe_d_dot = lambda t: np.array([
    l[0]*np.array([np.sin(theta_d(t))*theta_d_dot(t), -np.cos(theta_d(t))*theta_d_dot(t)]),
    l[1]*np.array([-np.sin(theta_d(t))*theta_d_dot(t), -np.cos(theta_d(t))*theta_d_dot(t)]),
    l[2]*np.array([0, 0]),
    l[3]*np.array([np.sin(theta_d(t))*theta_d_dot(t), np.cos(theta_d(t))*theta_d_dot(t)]),
    l[4]*np.array([-np.sin(theta_d(t))*theta_d_dot(t), np.cos(theta_d(t))*theta_d_dot(t)])
])
Qe_d_ddot = lambda t: np.array([
    l[0]*np.array([np.cos(theta_d(t))*theta_d_dot(t)**2 + np.sin(theta_d(t))*theta_d_ddot(t), np.sin(theta_d(t))*theta_d_dot(t)**2 - np.cos(theta_d(t))*theta_d_ddot(t)]),
    l[1]*np.array([-np.cos(theta_d(t))*theta_d_dot(t)**2 - np.sin(theta_d(t))*theta_d_ddot(t), np.sin(theta_d(t))*theta_d_dot(t)**2 - np.cos(theta_d(t))*theta_d_ddot(t)]),
    l[2]*np.array([0, 0]),
    l[3]*np.array([np.cos(theta_d(t))*theta_d_dot(t)**2 + np.sin(theta_d(t))*theta_d_ddot(t), -np.sin(theta_d(t))*theta_d_dot(t)**2 + np.cos(theta_d(t))*theta_d_ddot(t)]),
    l[4]*np.array([-np.cos(theta_d(t))*theta_d_dot(t)**2 - np.sin(theta_d(t))*theta_d_ddot(t), -np.sin(theta_d(t))*theta_d_dot(t)**2 + np.cos(theta_d(t))*theta_d_ddot(t)])
])

f_l = lambda t: np.array([0, np.sin(2*np.pi*t)])

sim = NetworkSimulatorV4(n, m, l, g, D, dt, tf, Q_0, Q_dot_0, f_l, Qe_d, Qe_d_dot, Qe_d_ddot, kp, kd, "result3")
sim.run()
sim.generate_plots(ylim=(-0.35, 0.35))
# sim.generate_animation_v2("Fig. 3c", ((-1.125, 1.125), (-0.5, 1.75)), arrows_bool=False)
# sim.generate_animation_v2("Fig. 3c", ((-1.125, 1.125), (-0.5, 1.75)), arrows_bool=True)