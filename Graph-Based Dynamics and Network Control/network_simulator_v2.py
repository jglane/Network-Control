import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

class NetworkSimulatorV2():
    def __init__(self, n: int, k: int,
                 m, l, g, D_of_G, H, dt, tf, Q_0, Q_dot_0,
                 f_l,#  r1_d, r1_d_dot, r1_d_ddot,
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
        self.H = H

        self.dt = dt
        self.tf = tf

        self.Q_0 = Q_0
        self.Q_dot_0 = Q_dot_0

        self.f_l = f_l
        # self.r1_d = r1_d
        # self.r1_d_dot = r1_d_dot
        # self.r1_d_ddot = r1_d_ddot

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
        self.E_dot = None

    def F_c_fun(self, Q, Q_dot, Qe, Qe_dot, t, inv=None):
        # lambda_simplified = inv @ np.diagonal(Qe_dot@Qe_dot.T)
        # Lambda_simplified = np.diag(lambda_simplified)
        U_prime = -self.kp*(Qe-self.Qe_d(t)) - self.kd*(Qe_dot-self.Qe_d_dot(t)) + self.Qe_d_ddot(t)
        U = U_prime - np.multiply(U_prime@Qe.T, np.diag([1/self.l[i]**2 for i in range(self.k)]))@Qe# + self.Le_of_G@Lambda_simplified@Qe
        
        #-self.kp*5*(Q[0, :] - self.r1_d(t)) - self.kd*5*(Q_dot[0, :] - self.r1_d_dot(t)) + self.r1_d_ddot(t)
        F = self.M@(1/self.m[0]*np.outer(np.ones(self.n), self.f_l(t)) + self.H@U)

        # F = np.zeros((self.n, 2))
        # F[0, :] = -self.kp*5*(Q[0, :] - self.r1_d(t)) - self.kd*5*(Q_dot[0, :] - self.r1_d_dot(t)) + self.r1_d_ddot(t)
        # A1 = (self.D_of_G.T@self.M_inv)[:, 0]
        # A_others = (self.D_of_G.T@self.M_inv)[:, 1:]
        # F[1:, :] = np.linalg.inv(A_others)@(U - np.outer(A1, F[0, :]))
        
        # F = np.ones((self.n, 2))*np.nan
        # while np.isnan(F).any():
        #     for i in np.where(np.isnan(F[:, 0]))[0]: # i is the node where we want to set a force
        #         for j in np.where(self.D_of_G[i, :] != 0)[0]: # j is an edge incident to the ith node
        #             head = np.where(self.D_of_G[:, j] == 1)[0][0]
        #             tail = np.where(self.D_of_G[:, j] == -1)[0][0]

        #             if head == i and ~np.isnan(F[tail, :]).any():
        #                 F[i, :] = self.m[i]*(U[j, :] + F[tail, :]/self.m[tail])
        #             elif tail == i and ~np.isnan(F[head, :]).any():
        #                 F[i, :] = self.m[i]*(-U[j, :] + F[head, :]/self.m[head])

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

        Qe_d_dot = np.array([self.Qe_d_dot(ti) for ti in self.t]).transpose((1, 2, 0))
        self.E_dot = self.Qe_dot - Qe_d_dot
    
    def generate_plots(self, filename: str):
        if self.t is None or self.Q is None or self.Qe is None:
            raise Exception("Run the simulation first")
        
        # Parameters
        # make it look like latex
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 15
        plt.rcParams['font.family'] = 'serif'
        # plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['lines.linewidth'] = 2.5
        
        # # Nodes
        # fig, axs = plt.subplots(self.n, 2, constrained_layout=True, figsize=(10, self.n * 2.5))
        # for i in range(self.n):
        #     ax = axs[i, 0]
        #     ax.plot(self.t, self.Q[i, 0, :], label='x' + str(i+1), linewidth=1.5)
        #     ax.set_xlim(0, self.tf)
        #     ax.set_ylim(-0.3, 0.3)
        #     ax.set_ylabel(f'Node {i+1} X', fontsize=20, fontfamily='serif')
        #     if i == self.n - 1:
        #         ax.set_xlabel('Time (s)', fontsize=20, fontfamily='serif')
        #     # ax.legend(loc='best')

        #     ax = axs[i, 1]
        #     ax.plot(self.t, self.Q[i, 1, :], label='y' + str(i+1), linewidth=1.5)
        #     ax.set_xlim(0, self.tf)
        #     ax.set_ylim(-0.3, 0.3)
        #     ax.set_ylabel(f'Node {i+1} Y', fontsize=20, fontfamily='serif')
        #     if i == self.n - 1:
        #         ax.set_xlabel('Time (s)', fontsize=20, fontfamily='serif')
        #     # ax.legend(loc='best')

        # Add global labels for better presentation
        # fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=14)
        # fig.text(0.04, 0.5, 'Position', va='center', rotation='vertical', fontsize=14)


        # Edges
        fig, axs = plt.subplots(self.k, 2, constrained_layout=True, figsize=(10, self.k * 2))
        for j in range(self.k):
            ax = axs[j, 0]
            ax.plot(self.t, self.Qe[j, 0, :])
            ax.plot(self.t, [self.Qe_d(t)[j, 0] for t in self.t], color='black', linestyle='--')
            ax.set_xlim(0, self.tf)
            ax.set_ylim(-0.11, 0.11)
            ax.set_ylabel('$r_{e' + str(j+1) + '}$', fontsize=25, fontfamily='serif')
            if j == 0:
                ax.set_title('x coordinates', fontsize=25, fontfamily='serif')
            if j == self.k - 1:
                ax.set_xlabel('Time (s)', fontsize=25, fontfamily='serif')

            ax = axs[j, 1]
            ax.plot(self.t, self.Qe[j, 1, :])
            ax.plot(self.t, [self.Qe_d(t)[j, 1] for t in self.t], color='black', linestyle='--')
            ax.set_xlim(0, self.tf)
            ax.set_ylim(-0.11, 0.11)
            if j == 0:
                ax.set_title('y coordinates', fontsize=25, fontfamily='serif')
            if j == self.k - 1:
                ax.set_xlabel('Time (s)', fontsize=25, fontfamily='serif')
        plt.savefig(filename + "_edges.pdf", format='pdf')

        # # Errors
        # plt.figure()
        # plt.suptitle('Errors')
        # for i in range(self.k):
        #     plt.subplot(self.k, 2, 2*i+1)
        #     plt.plot(self.t, self.E[i, 0], label='ex'+str(i+1))
        #     # plt.legend(loc='upper right')
        #     plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)
        #     plt.gca().set_ylim(-0.001, 0.001)

        #     plt.subplot(self.k, 2, 2*i+2)
        #     plt.plot(self.t, self.E[i, 1], label='ey'+str(i+1))
        #     # plt.legend(loc='upper right')
        #     plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)
        #     plt.gca().set_ylim(-0.001, 0.001)

        # J
        # plt.figure()
        # J_eig = np.zeros(len(self.t))
        # for idx in range(len(self.t)):
        #     J = np.multiply(self.Le_of_G, self.Qe[:, :, idx]@self.Qe[:, :, idx].T)
        #     print(J)
        #     J_eig[idx] = np.min(np.linalg.eigvals(J))
        # print(np.argmin(J_eig))
        # plt.plot(self.t, J_eig)
        # print(J_eig)
        # J_idk = np.multiply(self.Le_of_G, np.diag(np.square(self.l)))
        # J_eig_idk = np.min(np.linalg.eigvals(J_idk))
        # plt.axhline(J_eig_idk, color='r', linestyle='--')

        # # Lyapunov Function
        # epsilon = 1e-6
        # V = np.zeros(len(self.t))
        # V_dot = np.zeros(len(self.t))
        # psd = np.zeros(len(self.t))
        # for idx in range(len(self.t)):
        #     lambda_simplified = np.linalg.inv(np.multiply(self.Le_of_G, self.Qe[:, :, idx]@self.Qe[:, :, idx].T)) @ np.diagonal(self.Qe_dot[:, :, idx]@self.Qe_dot[:, :, idx].T)
        #     # lambda_full = np.linalg.inv(np.multiply(self.Le_of_G, self.Qe[:, :, idx]@self.Qe[:, :, idx].T)) @ np.diagonal(self.D_of_G.T@self.M_inv@self.F_c[:, :, idx]@self.Qe[:, :, idx].T + self.Qe_dot[:, :, idx]@self.Qe_dot[:, :, idx].T)
        #     for j in range(self.k):
        #         rej = self.Qe[j, :, idx]
        #         Pj =  np.eye(2) - np.outer(rej, rej)/self.l[j]**2
        #         epj = self.E[j, :, idx]
        #         edj = self.E_dot[j, :, idx]
        #         V[idx] += 0.5*self.kp*epj@((Pj+epsilon*np.eye(2))@epj)
        #         psd[idx] += 0.5*self.kp*edj@(Pj@edj)
        #         V_dot[idx] += -self.kd*edj@(Pj@edj)

        #     V[idx] += 0.5*np.trace(self.E_dot[:, :, idx].T@self.E_dot[:, :, idx])
        #     X = -np.array([(np.outer(self.Qe[j, :, idx], self.Qe[j, :, idx])/self.l[j]**2)@self.Qe_d_ddot(self.t[idx])[j, :] for j in range(self.k)]) - self.Le_of_G@np.diag(lambda_simplified)@self.Qe[:, :, idx]
        #     # X = np.ones((self.k, 2))*5
        #     V_dot[idx] += np.trace(self.E_dot[:, :, idx].T@(epsilon*self.kp*self.E[:, :, idx] + X))

        # plt.figure()
        # plt.suptitle('Lyapunov Function')
        # plt.plot(self.t, V)
        # plt.plot(self.t, np.gradient(V, self.t), linestyle='--')
        # # plt.plot(self.t, psd)
        # plt.plot(self.t, V_dot)
        # plt.gca().set_xlim(0, self.tf)#(-0.05*self.tf, self.tf + 0.05*self.tf)
        # # plt.gca().set_ylim(-0.15, 0.15)

        # # Forces
        # plt.figure()
        # plt.suptitle('Forces')
        # for i in range(self.n):
        #     plt.subplot(self.n, 2, 2*i+1)
        #     plt.plot(self.t, self.F_c[i, 0], label='Fcx'+str(i+1))
        #     plt.plot(self.t, self.F_g[i, 0], label='Fgx'+str(i+1))
        #     # plt.legend(loc='upper right')
        #     plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)
        #     plt.gca().set_ylim(-1.1, 1.1)

        #     plt.subplot(self.n, 2, 2*i+2)
        #     plt.plot(self.t, self.F_c[i, 1], label='Fcy'+str(i+1))
        #     plt.plot(self.t, self.F_g[i, 1], label='Fgy'+str(i+1))
        #     # plt.legend(loc='upper right')
        #     plt.gca().set_xlim(-0.05*self.tf, self.tf + 0.05*self.tf)
        #     plt.gca().set_ylim(-1.1, 1.1)

    def generate_animation(self, filename: str, limits: tuple):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=limits[0], ylim=limits[1], aspect='equal')
        ax.grid()

        # Plot initial position
        bars = []
        for i in range(self.k):
            d_i = self.D_of_G[:, i]
            head_idx = np.where(d_i == 1)[0][0]
            tail_idx = np.where(d_i == -1)[0][0]
            x, y = [self.Q_0[tail_idx, 0], self.Q_0[head_idx, 0]], [self.Q_0[tail_idx, 1], self.Q_0[head_idx, 1]]
            bar = ax.arrow(x[0], y[0], x[1]-x[0], y[1]-y[0], length_includes_head=True, lw=2, head_width=0.01, head_length=0.01, fc='k', ec='k')
            bars.append(bar)

        F_c_arrows = []
        # F_g_arrows = []
        for i in range(self.n):
            if i == 0:
                F_c_arrow = ax.arrow(self.Q[i, 0, 0], self.Q[i, 1, 0], self.f_l(0)[0], self.f_l(0)[1], head_width=0.01, head_length=0.01, fc='r', ec='r')
            else:
                F_c_arrow = ax.arrow(self.Q[i, 0, 0], self.Q[i, 1, 0], self.F_c[i, 0, 0], self.F_c[i, 1, 0], head_width=0.01, head_length=0.01, fc='b', ec='b')
            # F_g_arrow = ax.arrow(self.Q[i, 0, 0], self.Q[i, 1, 0], self.F_g[i, 0, 0]/10, self.F_g[i, 1, 0]/10, head_width=0.01, head_length=0.01, fc='b', ec='b')
            F_c_arrows.append(F_c_arrow)
            # F_g_arrows.append(F_g_arrow)

        # ax.text(0.05, 0.9, f'time: {t[0]:.1f} s', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5), fontsize=18)

        fps = 30
        def animate(i):
            idx = int(i/(fps*self.dt))
            for i in range(self.k):
                d_i = self.D_of_G[:, i]
                head_idx = np.where(d_i == 1)[0][0]
                tail_idx = np.where(d_i == -1)[0][0]
                x, y = [self.Q[tail_idx, 0, idx], self.Q[head_idx, 0, idx]], [self.Q[tail_idx, 1, idx], self.Q[head_idx, 1, idx]]
                bars[i].set_data(x=x[0], y=y[0], dx=x[1]-x[0], dy=y[1]-y[0])
            
            for i in range(self.n):
                if i == 0:
                    F_c_arrows[i].set_data(x=self.Q[i, 0, idx], y=self.Q[i, 1, idx], dx=self.f_l(self.t[idx])[0], dy=self.f_l(self.t[idx])[1])
                else:
                    F_c_arrows[i].set_data(x=self.Q[i, 0, idx], y=self.Q[i, 1, idx], dx=self.F_c[i, 0, idx], dy=self.F_c[i, 1, idx])
                # F_g_arrows[i].set_data(x=self.Q[i, 0, idx], y=self.Q[i, 1, idx], dx=self.F_g[i, 0, idx]/10, dy=self.F_g[i, 1, idx]/10)

        tf_sim = self.tf
        ani = animation.FuncAnimation(fig, animate, frames=fps*tf_sim)
        ffmpeg_writer = animation.FFMpegWriter(fps=fps)
        ani.save(filename + ".mp4", writer=ffmpeg_writer)
        plt.close(fig)