import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

class NetworkSimulatorV2():
    def __init__(self, n: int, k: int,
                 m, l, g, D_of_G, dt, tf, Q_0, Q_dot_0,
                 i_leader, r_leader_d, r_leader_d_dot, r_leader_d_ddot,
                 Qe_d, Qe_d_dot, Qe_d_ddot,
                 kp, kd) -> None:
        self.n = n
        self.k = k

        self.m = m
        self.l = l
        self.g = g

        self.M = np.diag(m)
        self.M_inv = np.linalg.inv(self.M)
        self.G = np.array([[0, g]]*n)

        self.D_of_G = D_of_G
        self.Le_of_G = D_of_G.T@self.M_inv@D_of_G

        self.dt = dt
        self.tf = tf

        self.Q_0 = Q_0
        self.Q_dot_0 = Q_dot_0

        self.i_leader = i_leader
        self.r_leader_d = r_leader_d
        self.r_leader_d_dot = r_leader_d_dot
        self.r_leader_d_ddot = r_leader_d_ddot

        self.Qe_d = Qe_d
        self.Qe_d_dot = Qe_d_dot
        self.Qe_d_ddot = Qe_d_ddot

        self.kp = kp
        self.kd = kd
        
        self.t = None
        self.Q = None
        self.Q_dot = None
        self.Qe = None
        self.Qe_dot = None
        self.F_c = None
        self.F_g = None
        self.E = None

    def F_c_fun(self, Q, Q_dot, Qe, Qe_dot, t, inv=None):
        # lambda_simplified = inv @ np.diagonal(Qe_dot@Qe_dot.T)
        # Lambda_simplified = np.diag(lambda_simplified)
        S_temp = -self.kp*(Qe-self.Qe_d(t)) - self.kd*(Qe_dot-self.Qe_d_dot(t)) + self.Qe_d_ddot(t) #+ Le_of_G@Lambda_simplified@Qe        
        # S = S_temp - np.diag(np.diagonal(S_temp@Qe.T@np.diag([1/self.l[i]**2 for i in range(self.k)])))@Qe
        S = S_temp - np.multiply(S_temp@Qe.T, np.diag([1/self.l[i]**2 for i in range(self.k)]))@Qe

        F = np.ones((self.n, 2))*np.nan
        F[self.i_leader, :] = -self.kp*(Q[self.i_leader, :] - self.r_leader_d(t)) - self.kd*(Q_dot[self.i_leader, :] - self.r_leader_d_dot(t)) + self.r_leader_d_ddot(t)

        while np.isnan(F).any():
            for i in np.where(np.isnan(F[:, 0]))[0]: # i is the node where we want to set a force
                for j in np.where(self.D_of_G[i, :] != 0)[0]: # j is an edge incident to the ith node
                    head = np.where(self.D_of_G[:, j] == 1)[0][0]
                    tail = np.where(self.D_of_G[:, j] == -1)[0][0]

                    if head == i and ~np.isnan(F[tail, :]).any():
                        F[i, :] = self.m[i]*(S[j, :] + F[tail, :]/self.m[tail])
                    elif tail == i and ~np.isnan(F[head, :]).any():
                        F[i, :] = self.m[i]*(-S[j, :] + F[head, :]/self.m[head])

        # def sat(f, a):
        #     f_norm = np.linalg.norm(f)
        #     if f_norm > a:
        #         return a*f/f_norm
        #     else:
        #         return f
        # for i in range(self.n):
        #     F[i, :] = sat(F[i, :], 2)
        
        return F

    def ODE(self, t, y):
        Q = y[:2*self.n].reshape((self.n, 2))
        Q_dot = y[2*self.n:].reshape((self.n, 2))
        Qe = self.D_of_G.T@Q
        Qe_dot = self.D_of_G.T@Q_dot

        inv = np.linalg.inv(np.multiply(self.Le_of_G, Qe@Qe.T))
        
        F_c = self.F_c_fun(Q, Q_dot, Qe, Qe_dot, t, inv)
        F_g = self.M@self.G
        F = F_c + F_g
        
        lambda_full = inv @ np.diagonal(self.D_of_G.T@self.M_inv@F@Qe.T + Qe_dot@Qe_dot.T)
        Lambda_full = np.diag(lambda_full)
        Q_ddot = -self.G + self.M_inv@(F - self.D_of_G@Lambda_full@Qe)
        
        return np.concatenate((Q_dot.reshape(2*self.n), Q_ddot.reshape(2*self.n)))
    
    def run(self):
        self.t = np.arange(0, self.tf, self.dt)
        sol = solve_ivp(self.ODE, (0, self.tf), np.concatenate((self.Q_0.reshape(2*self.n), self.Q_dot_0.reshape(2*self.n))), t_eval=self.t, method="DOP853", rtol=1e-10, atol=1e-10)
        self.Q = sol.y[:2*self.n].reshape((self.n, 2, len(self.t)))
        self.Q_dot = sol.y[2*self.n:].reshape((self.n, 2, len(self.t)))

        self.Qe = np.empty((self.k, 2, len(self.t)))
        for idx in range(len(self.t)):
            self.Qe[:, :, idx] = self.D_of_G.T@self.Q[:, :, idx]
        
        self.Qe_dot = np.empty((self.k, 2, len(self.t)))
        for idx in range(len(self.t)):
            self.Qe_dot[:, :, idx] = self.D_of_G.T@self.Q_dot[:, :, idx]

        self.F_c = np.empty((self.n, 2, len(self.t)))
        for idx in range(len(self.t)):
            self.F_c[:, :, idx] = self.F_c_fun(self.Q[:, :, idx], self.Q_dot[:, :, idx], self.Qe[:, :, idx], self.Qe_dot[:, :, idx], self.t[idx])

        self.F_g = np.array([self.M@self.G for _ in range(len(self.t))]).transpose((1, 2, 0))

        Qe_d = np.array([self.Qe_d(ti) for ti in self.t]).transpose((1, 2, 0))
        self.E = self.Qe - Qe_d
    
    def generate_plots(self):
        if self.t is None or self.Q is None or self.Qe is None:
            raise Exception("Run the simulation first")
        
        # Nodes
        plt.figure()
        for i in range(self.n):
            plt.subplot(self.n, 2, 2*i+1)
            plt.plot(self.t, self.Q[i, 0, :], label='x'+str(i+1))
            if i == self.i_leader:
                plt.plot(self.t, [self.r_leader_d(t)[0] for t in self.t], color='black', linestyle='--')
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)

            plt.subplot(self.n, 2, 2*i+2)
            plt.plot(self.t, self.Q[i, 1], label='y'+str(i+1))
            if i == self.i_leader:
                plt.plot(self.t, [self.r_leader_d(t)[1] for t in self.t], color='black', linestyle='--')
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)

        # Edges
        plt.figure()
        for i in range(self.k):
            plt.subplot(self.k, 2, 2*i+1)
            plt.plot(self.t, self.Qe[i, 0, :], label='xe'+str(i+1))
            plt.plot(self.t, [self.Qe_d(ti)[i, 0] for ti in self.t], color='black', linestyle='--')
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)

            plt.subplot(self.k, 2, 2*i+2)
            plt.plot(self.t, self.Qe[i, 1, :], label='ye'+str(i+1))
            plt.plot(self.t, [self.Qe_d(ti)[i, 1] for ti in self.t], color='black', linestyle='--')
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)

        # Errors
        plt.figure()
        for i in range(self.k):
            plt.subplot(self.k, 2, 2*i+1)
            plt.plot(self.t, self.E[i, 0], label='ex'+str(i+1))
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)

            plt.subplot(self.k, 2, 2*i+2)
            plt.plot(self.t, self.E[i, 1], label='ey'+str(i+1))
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)

        # Forces
        plt.figure()
        for i in range(self.n):
            plt.subplot(self.n, 2, 2*i+1)
            plt.plot(self.t, self.F_c[i, 0], label='Fcx'+str(i+1))
            plt.plot(self.t, self.F_g[i, 0], label='Fgx'+str(i+1))
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)
            plt.gca().set_ylim(-1.1, 1.1)

            plt.subplot(self.n, 2, 2*i+2)
            plt.plot(self.t, self.F_c[i, 1], label='Fcy'+str(i+1))
            plt.plot(self.t, self.F_g[i, 1], label='Fgy'+str(i+1))
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)
            plt.gca().set_ylim(-1.1, 1.1)

    def generate_animation(self, filename: str, limits: tuple):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=limits[0], ylim=limits[1], aspect='equal')
        ax.grid()

        # Plot initial position
        bars = []
        for i in range(self.k):
            d_i = self.D_of_G[:, i]
            start_idx = np.where(d_i == 1)[0][0]
            end_idx = np.where(d_i == -1)[0][0]
            bar = ax.plot([self.Q_0[start_idx, 0], self.Q_0[end_idx, 0]], [self.Q_0[start_idx, 1], self.Q_0[end_idx, 1]], 'o-', color='purple', lw=2)[0]
            bars.append(bar)

        F_c_arrows = []
        F_n_arrows = []
        F_g_arrows = []
        for i in range(self.n):
            F_c_arrow = ax.arrow(self.Q[i, 0, 0], self.Q[i, 1, 0], self.F_c[i, 0, 0], self.F_c[i, 1, 0], head_width=0.1, head_length=0.1, fc='r', ec='r')
            F_g_arrow = ax.arrow(self.Q[i, 0, 0], self.Q[i, 1, 0], self.F_g[i, 0, 0]/10, self.F_g[i, 1, 0]/10, head_width=0.1, head_length=0.1, fc='b', ec='b')
            F_c_arrows.append(F_c_arrow)
            F_g_arrows.append(F_g_arrow)

        # ax.text(0.05, 0.9, f'time: {t[0]:.1f} s', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5), fontsize=18)

        fps = 30
        def animate(i):
            idx = int(i/(fps*self.dt))
            for i in range(self.k):
                d_i = self.D_of_G[:, i]
                start_idx = np.where(d_i == 1)[0][0]
                end_idx = np.where(d_i == -1)[0][0]
                bars[i].set_data([self.Q[start_idx, 0, idx], self.Q[end_idx, 0, idx]], [self.Q[start_idx, 1, idx], self.Q[end_idx, 1, idx]])
            
            for i in range(self.n):
                F_c_arrows[i].set_data(x=self.Q[i, 0, idx], y=self.Q[i, 1, idx], dx=self.F_c[i, 0, idx], dy=self.F_c[i, 1, idx])
                F_g_arrows[i].set_data(x=self.Q[i, 0, idx], y=self.Q[i, 1, idx], dx=self.F_g[i, 0, idx]/10, dy=self.F_g[i, 1, idx]/10)

        tf_sim = self.tf
        ani = animation.FuncAnimation(fig, animate, frames=fps*tf_sim)
        ffmpeg_writer = animation.FFMpegWriter(fps=fps)
        ani.save(filename + ".mp4", writer=ffmpeg_writer)
        plt.close(fig)