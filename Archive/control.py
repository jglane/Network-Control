from sympy import *
from sympy.physics.mechanics import *
import numpy as np
from scipy.integrate import solve_ivp
import csv

m1 = 3
m2 = 5
l = 1
g = 9.81

Kp = 100
Kd = 50
Ki = 50

theta_d = lambda t: pi/4*sin(pi*t) + pi/8
theta_d_dot = lambda t: pi**2/4*cos(pi*t)
theta_d_ddot = lambda t: -pi**3/4*sin(pi*t)
theta_d_dddot = lambda t: -pi**4/4*cos(pi*t)

theta_0 = np.array([[0], [0]])
theta_dot_0 = np.array([[0], [0]])
theta_ddot_0 = np.array([[0], [0]])

theta_e_0 = theta_d(0) - theta_0[0][0]
theta_e_dot_0 = theta_d_dot(0) - theta_dot_0[0][0]
theta_e_ddot_0 = theta_d_ddot(0) - theta_ddot_0[0][0]

def ODE(t, y):
    x1, x2, x3 = y
    return (
        x2,
        x3,
        (
        m2*l*(-cos(theta_d(t)-x1)*theta_d_dddot(t)-(theta_d_dot(t)-x2)*sin(theta_d(t)-x1)*theta_d_ddot(t)) + (m1+m2)*g*sec(theta_d(t)-x1)**2*(theta_d_dot(t)-x2) + m2*l*((theta_d_dot(t)-x2)**2*cos(theta_d(t)-x1)+(theta_d_ddot(t)-x3)*sin(theta_d(t)-x1)) + Kp*x2 + Kd*x3 + Ki*x1
        - x3*(theta_d_dot(t)-x2)*((m1+m2)*l*sec(theta_d(t)-x1)*tan(theta_d(t)-x1) + m2*l*sin(theta_d(t)-x1))
        ) / ((m1+m2)*l/cos(theta_d(t)-x1)-m2*l*cos(theta_d(t)-x1))
    )

t_stop = 60
fps = 30
sol = solve_ivp(ODE, (0, t_stop), (theta_e_0, theta_e_dot_0, theta_e_ddot_0), t_eval=np.linspace(0, t_stop, int(fps*t_stop)))

theta_e_sol, theta_e_dot_sol, theta_e_ddot_sol = sol.y
t = sol.t

theta_d_sol = np.array([np.pi/4*np.sin(np.pi*ti) + np.pi/8 for ti in t])
theta_d_dot_sol = np.array([np.pi**2/4*np.cos(np.pi*ti) for ti in t])
theta_d_ddot_sol = np.array([-np.pi**3/4*np.sin(np.pi*ti) for ti in t])

theta_sol = theta_d_sol - theta_e_sol
theta_dot_sol = theta_d_dot_sol - theta_e_dot_sol
theta_ddot_sol = theta_d_ddot_sol - theta_e_ddot_sol

with open(f'data_{Kp}_{Kd}_{Ki}.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(['t', 'theta_sol', 'theta_dot_sol', 'theta_ddot_sol', 'theta_e_sol', 'theta_e_dot_sol', 'theta_e_ddot_sol', 'theta_d_sol', 'theta_d_dot_sol', 'theta_d_ddot_sol'])
    for i in range(len(t)):
        writer.writerow([t[i], theta_sol[i], theta_dot_sol[i], theta_ddot_sol[i], theta_e_sol[i], theta_e_dot_sol[i], theta_e_ddot_sol[i], theta_d_sol[i], theta_d_dot_sol[i], theta_d_ddot_sol[i]])