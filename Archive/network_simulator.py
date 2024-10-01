import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

class NetworkSimulator():
    def __init__(self, n: int, k, max_k: int,
                 m, l, g, D_of_G, dt, tf, Q_0, Q_dot_0,
                 Q_d, Q_d_dot, Qe_d, Qe_d_dot, Qe_d_ddot,
                 kp, kd, control_type: tuple) -> None:
        self.n = n
        self.k = k
        self.max_k = max_k

        self.m = m
        self.l = l
        self.g = g
        self.D_of_G = D_of_G

        self.M = np.diag(m)
        self.M_inv = np.linalg.inv(self.M)
        self.G = np.array([[0, g]]*n)

        self.dt = dt
        self.tf = tf

        self.Q_0 = Q_0
        self.Q_dot_0 = Q_dot_0

        self.Q_d = Q_d
        self.Q_d_dot = Q_d_dot

        self.Qe_d = Qe_d
        self.Qe_d_dot = Qe_d_dot
        self.Qe_d_ddot = Qe_d_ddot

        self.kp = kp
        self.kd = kd
        self.control_type = control_type

        self.R90 = np.array([[0, -1],
                             [1,  0]])
        
        self.t = None
        self.Q = None
        self.Qe = None

    def ODE(self, t, y):
        k = self.k(t)
        
        Q = y[:2*self.n].reshape((self.n, 2))
        Q_dot = y[2*self.n:].reshape((self.n, 2))
        Qe = self.D_of_G(k).T@Q
        Qe_dot = self.D_of_G(k).T@Q_dot
        
        temp = np.multiply(self.D_of_G(k).T@self.M_inv@self.D_of_G(k), Qe@Qe.T)
        lambda_simplified = np.linalg.solve(temp, (Qe_dot@Qe_dot.T).diagonal())
        Lambda_simplified = np.diag(lambda_simplified)

        F_pd = np.zeros((self.n, 2))
        mat = Qe.T@Lambda_simplified@self.D_of_G(k).T@self.M_inv@self.D_of_G(k)
        for i, j in enumerate(self.control_type):
            if j == None: # node control
                r = Q[i, :]
                r_d = self.Q_d(t, k)[i, :]
                r_dot = Q_dot[i, :]
                r_d_dot = self.Q_d_dot(t, k)[i, :]
                f = -self.kp[i]*(r - r_d) - self.kd[i]*(r_dot - r_d_dot)
            else:
                j = j - 1
                re = Qe[j, :]
                re_d = self.Qe_d(t, k)[j, :]
                re_dot = Qe_dot[j, :]
                re_d_dot = self.Qe_d_dot(t, k)[j, :]
                re_d_ddot = self.Qe_d_ddot(t, k)[j, :]
                u = (-self.kp[i]*(re - re_d) - self.kd[i]*(re_dot - re_d_dot) + re_d_ddot + mat[:, j]).dot(self.R90@re/self.l[j])
                f = self.m[i]*u*self.R90@re/self.l[j]
            F_pd[i, :] = f

        F_g = self.M@self.G
        F = F_pd + F_g
        
        lambda_full = np.linalg.solve(temp, (self.D_of_G(k).T@self.M_inv@F@Qe.T + Qe_dot@Qe_dot.T).diagonal())
        Lambda_full = np.diag(lambda_full)
        Q_ddot = -self.G + self.M_inv@(F - self.D_of_G(k)@Lambda_full@Qe)

        # if not collided_yet and np.abs(np.linalg.norm(r5-r6) - l5) < 3e-4:
        #     collided_yet = True
        #     print(f"Collision at time: {t}")
        #     k = 5
        #     return np.zeros(2*n*2)
        
        return np.concatenate((Q_dot.reshape(2*self.n), Q_ddot.reshape(2*self.n)))
    
    def run(self):
        self.t = np.arange(0, self.tf, self.dt)
        sol = solve_ivp(self.ODE, (0, self.tf), np.concatenate((self.Q_0.reshape(2*self.n), self.Q_dot_0.reshape(2*self.n))), t_eval=self.t, method="DOP853", rtol=1e-10, atol=1e-10)
        self.Q = sol.y[:2*self.n].reshape((self.n, 2, len(self.t)))

        # self.Q = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]])
        self.Qe = np.empty((self.max_k, 2, len(self.t)))
        for idx, t in enumerate(self.t):
            self.Qe[:self.k(t), :, idx] = self.D_of_G(self.k(t)).T@self.Q[:, :, idx]
            self.Qe[self.k(t):, :, idx] = np.nan

        return self.t, self.Q, self.Qe
    
    def generate_plots(self):
        if self.t is None or self.Q is None or self.Qe is None:
            raise Exception("Run the simulation first")
        
        # Nodes
        plt.figure()
        for i in range(self.n):
            plt.subplot(self.n, 2, 2*i+1)
            plt.plot(self.t, self.Q[i, 0], label='x'+str(i+1))
            if self.control_type[i] is None:
                plt.plot(self.t, [self.Q_d(t, self.k(t))[i][0] for t in self.t], color='black', linestyle='--')
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)

            plt.subplot(self.n, 2, 2*i+2)
            plt.plot(self.t, self.Q[i, 1], label='y'+str(i+1))
            if self.control_type[i] is None:
                plt.plot(self.t, [self.Q_d(t, self.k(t))[i][1] for t in self.t], color='black', linestyle='--')
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)

        # Edges
        plt.figure()
        for i in range(self.max_k):
            plt.subplot(self.max_k, 2, 2*i+1)
            plt.plot(self.t, self.Qe[i, 0, :], label='xe'+str(i+1))
            plt.plot(self.t, [(self.Qe_d(ti, self.k(ti))[i, 0] if i < self.k(ti) else np.nan) for j, ti in enumerate(self.t)], color='black', linestyle='--')
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)

            plt.subplot(self.max_k, 2, 2*i+2)
            plt.plot(self.t, self.Qe[i, 1, :], label='ye'+str(i+1))
            plt.plot(self.t, [(self.Qe_d(ti, self.k(ti))[i, 1] if i < self.k(ti) else np.nan) for j, ti in enumerate(self.t)], color='black', linestyle='--')
            # plt.legend(loc='upper right')
            plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)

    def generate_animation(self, filename: str, limits: tuple = ((-2, 2), (-2, 2))):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=limits[0], ylim=limits[1], aspect='equal')
        ax.grid()

        # Plot initial position
        bars = []
        for i in range(self.max_k):
            try:
                d_i = self.D_of_G(self.k(0))[:, i]
                start_idx = np.where(d_i == 1)[0][0]
                end_idx = np.where(d_i == -1)[0][0]
                bar = ax.plot([self.Q_0[start_idx, 0], self.Q_0[end_idx, 0]], [self.Q_0[start_idx, 1], self.Q_0[end_idx, 1]], 'o-', lw=2)[0]
                bars.append(bar)
            except:
                bars.append(ax.plot([], [], 'o-', lw=2)[0])

        # ax.text(0.05, 0.9, f'time: {t[0]:.1f} s', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5), fontsize=18)

        fps = 30
        def animate(frame):
            idx = int(frame/(fps*self.dt))
            t = self.t[idx]
            for j in range(self.k(t)):
                d_j = self.D_of_G(self.k(t))[:, j]
                start_idx = np.where(d_j == 1)[0][0]
                end_idx = np.where(d_j == -1)[0][0]
                bars[j].set_data([self.Q[start_idx, 0, idx], self.Q[end_idx, 0, idx]], [self.Q[start_idx, 1, idx], self.Q[end_idx, 1, idx]])
            for j in range(self.k(t), self.max_k):
                bars[j].set_data([], [])

        tf_sim = self.tf
        ani = animation.FuncAnimation(fig, animate, frames=fps*tf_sim)
        ffmpeg_writer = animation.FFMpegWriter(fps=fps)
        ani.save(filename + ".mp4", writer=ffmpeg_writer)
        plt.close(fig)