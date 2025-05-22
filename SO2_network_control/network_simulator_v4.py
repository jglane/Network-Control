import numpy as np
import numpy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

class NetworkSimulatorV4():
    def __init__(self, n: int,
                 m, l, g, D, dt, tf, Q_0, Q_dot_0,
                 f_l,#  r1_d, r1_d_dot, r1_d_ddot,
                 Qe_d, Qe_d_dot, Qe_d_ddot,
                 k_R, k_nu, dir: str) -> None:
        self.n = n

        self.m = m
        self.l = l
        self.g = g

        self.M = np.diag(m)
        self.M_inv = la.inv(self.M)
        self.G = np.array([[0, g]]*n)

        self.D = D
        self.Le = D.T@self.M_inv@D
        self.H = (la.pinv(D) - la.pinv(D)[0,0]*np.ones((n-1, n))).round()

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

        self.k_R = k_R
        self.k_nu = k_nu

        self.beta = NetworkSimulatorV4.beta_fun(self.m, self.l, self.D)
        print(f"beta: {self.beta}")

        self.dir = dir
        
        self.t = None
        self.Q = None
        self.Q_dot = None
        self.Qe = None
        self.Qe_dot = None
        # self.F_bar = None
        # self.F_g = None
        self.e_R = None
        self.nu = None
        self.nu_d = None
        self.e_nu = None
        self.e_nu_dot = None

    def beta_fun(m, l, D):
        Le = D.T@la.inv(np.diag(m))@D
        
        # Minimum value of the smallest eigenvlaue of J occurs when all edges are parallel (probably)
        Qe_for_min_J = np.array([l, np.zeros_like(l)]).T
        J_min = np.multiply(Le, Qe_for_min_J@Qe_for_min_J.T)
        # beta = np.max([la.norm(self.l)**3*la.norm(self.Le[:, j])/np.min(la.eigvals(J_min))/self.l[j] for j in range(self.n-1)])

        beta = 0
        for j in range(len(l)):
            Le_col = np.copy(Le[:, j])
            Le_col[j] = 0
            betaj = np.sqrt(np.dot(np.square(Le_col), np.square(l))) / np.min(la.eigvals(J_min))
            if betaj > beta:
                beta = betaj
        
        return beta
    
    def F_bar_fun(self, Qe, Qe_dot, t):
        U = np.empty((self.n-1, 2))
        
        for j in range(self.n-1):
            rej = Qe[j, :]
            rej_d = self.Qe_d(t)[j, :]

            rej_dot = Qe_dot[j, :]
            rej_d_dot = self.Qe_d_dot(t)[j, :]
            rej_d_ddot = self.Qe_d_ddot(t)[j, :]

            R = np.array([[rej[0], -rej[1]],
                            [rej[1], rej[0]]])/self.l[j]
            R_d = np.array([[rej_d[0], -rej_d[1]],
                              [rej_d[1], rej_d[0]]])/self.l[j]
            R_dot = np.array([[rej_dot[0], -rej_dot[1]],
                               [rej_dot[1], rej_dot[0]]])/self.l[j]
            R_d_dot = np.array([[rej_d_dot[0], -rej_d_dot[1]],
                                  [rej_d_dot[1], rej_d_dot[0]]])/self.l[j]
            R_d_ddot = np.array([[rej_d_ddot[0], -rej_d_ddot[1]],
                                   [rej_d_ddot[1], rej_d_ddot[0]]])/self.l[j]
            
            e_R = self.l[j]*SO2.logvee(R@R_d.T)
            nu = self.l[j]*SO2.vee(R_dot@R.T)
            nu_d = self.l[j]*SO2.vee(R_d_dot@R_d.T)
            e_nu = nu - nu_d
            nu_d_dot = self.l[j]*SO2.vee(R_d_ddot@R_d.T + R_d_dot@R_d_dot.T)

            B = 0.5551652475612766

            U[j, :] = SO2.hat(-self.k_R*e_R - self.k_nu*e_nu + nu_d_dot - self.beta*(self.n-1)*e_nu*np.abs(e_nu))/self.l[j]@rej

            # b1 = np.array([rej[0], rej[1]])/self.l[j]
            # b2 = np.array([-rej[1], rej[0]])/self.l[j]

            # b1_d = np.array([rej_d[0], rej_d[1]])/self.l[j]
            # b2_d = np.array([-rej_d[1], rej_d[0]])/self.l[j]
            
            # e_R = self.l[j]*np.atan2(np.dot(b2_d, b1), np.dot(b1_d, b1))
            # e_nu = np.dot(b2, rej_dot) - np.dot(b2_d, rej_d_dot)

            # nu_d_dot = np.dot(b2, rej_d_ddot)

            # U[j, :] = b2*(-self.k_R*e_R - self.k_nu*e_nu + nu_d_dot - self.beta*(self.n-1)*e_nu*np.abs(e_nu))
        
        return self.M@(1/self.m[0]*np.outer(np.ones(self.n), self.f_l(t)) + self.H.T@U)

    def ODE(self, t, y):
        Q = y[:2*self.n].reshape((self.n, 2))
        Q_dot = y[2*self.n:].reshape((self.n, 2))
        Qe = self.D.T@Q
        Qe_dot = self.D.T@Q_dot
        
        F = self.F_bar_fun(Qe, Qe_dot, t) + self.M@self.G
        
        lambda_full = la.inv(np.multiply(self.Le, Qe@Qe.T)) @ np.diagonal(self.D.T@self.M_inv@F@Qe.T + Qe_dot@Qe_dot.T)
        Lambda_full = np.diag(lambda_full)
        Q_ddot = -self.G + self.M_inv@(F - self.D@Lambda_full@Qe)
        return np.concatenate((Q_dot.reshape(2*self.n), Q_ddot.reshape(2*self.n)))
    
    def run(self):
        self.t = np.arange(0, self.tf, self.dt)
        sol = solve_ivp(self.ODE, (0, self.tf), np.concatenate((self.Q_0.reshape(2*self.n), self.Q_dot_0.reshape(2*self.n))), t_eval=self.t, method="DOP853", rtol=1e-10, atol=1e-10)
        self.Q = sol.y[:2*self.n].reshape((self.n, 2, len(self.t)))
        self.Q_dot = sol.y[2*self.n:].reshape((self.n, 2, len(self.t)))

        self.Qe = np.empty((self.n-1, 2, len(self.t)))
        for idx in range(len(self.t)):
            self.Qe[:, :, idx] = self.D.T@self.Q[:, :, idx]
        
        self.Qe_dot = np.empty((self.n-1, 2, len(self.t)))
        for idx in range(len(self.t)):
            self.Qe_dot[:, :, idx] = self.D.T@self.Q_dot[:, :, idx]

        self.F_bar = np.empty((self.n, 2, len(self.t)))
        for idx in range(len(self.t)):
            self.F_bar[:, :, idx] = self.F_bar_fun(self.Qe[:, :, idx], self.Qe_dot[:, :, idx], self.t[idx])

        self.F_g = np.array([self.M@self.G for _ in range(len(self.t))]).transpose((1, 2, 0))

        Qe_d = np.array([self.Qe_d(ti) for ti in self.t]).transpose((1, 2, 0))
        Qe_d_dot = np.array([self.Qe_d_dot(ti) for ti in self.t]).transpose((1, 2, 0))
        Qe_d_ddot = np.array([self.Qe_d_ddot(ti) for ti in self.t]).transpose((1, 2, 0))

        self.e_R = np.empty((self.n-1, len(self.t)))
        self.nu = np.empty((self.n-1, len(self.t)))
        self.nu_d = np.empty((self.n-1, len(self.t)))
        self.e_nu = np.empty((self.n-1, len(self.t)))
        self.e_nu_dot = np.empty((self.n-1, len(self.t)))
        for idx in range(len(self.t)):
            for j in range(self.n-1):
                Thetaj = np.array([[self.Qe[j, 0, idx], -self.Qe[j, 1, idx]],
                                [self.Qe[j, 1, idx], self.Qe[j, 0, idx]]]) / self.l[j]
                Thetaj_d = np.array([[Qe_d[j, 0, idx], -Qe_d[j, 1, idx]],
                                    [Qe_d[j, 1, idx], Qe_d[j, 0, idx]]]) / self.l[j]
                self.e_R[j, idx] = self.l[j]*SO2.logvee(Thetaj@Thetaj_d.T)

                self.nu[j, idx] = np.dot(np.array([-self.Qe[j, 1, idx], self.Qe[j, 0, idx]]), np.array([self.Qe_dot[j, 0, idx], self.Qe_dot[j, 1, idx]])) / self.l[j]
                self.nu_d[j, idx] = np.dot(np.array([-Qe_d[j, 1, idx], Qe_d[j, 0, idx]]), np.array([Qe_d_dot[j, 0, idx], Qe_d_dot[j, 1, idx]])) / self.l[j]
                self.e_nu[j, idx] = self.nu[j, idx] - self.nu_d[j, idx]

                # nuj_dot = np.dot(np.array([-self.Qe[j, 1, idx], self.Qe[j, 0, idx]]), np.array([self.Qe_ddot[j, 0, idx], self.Qe_ddot[j, 1, idx]])) / self.l[j]**2
                # nuj_d_dot = np.dot(np.array([-Qe_d[j, 1, idx], Qe_d[j, 0, idx]]), np.array([Qe_d_ddot[j, 0, idx], Qe_d_ddot[j, 1, idx]])) / self.l[j]**2
                # self.e_nu_dot[j, idx] = nuj_dot - nuj_d_dot
    
    def generate_plots(self, ylim: tuple):
        if self.t is None or self.Q is None or self.Qe is None:
            raise Exception("Run the simulation first")
        
        # Parameters
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 15
        plt.rcParams['font.family'] = 'serif'
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
        fig, axs = plt.subplots(self.n-1, 2, constrained_layout=True, figsize=(10, (self.n-1) * 2))
        for j in range(self.n-1):
            ax = axs[j, 0]
            ax.plot(self.t, self.Qe[j, 0, :])
            ax.plot(self.t, [self.Qe_d(t)[j, 0] for t in self.t], color='black', linestyle='--')
            ax.set_xlim(0, self.tf)
            ax.set_ylim(ylim)
            ax.set_ylabel('$r_{e' + str(j+1) + '}$', fontsize=25, fontfamily='serif')
            if j == 0:
                ax.set_title('x coordinates', fontsize=25, fontfamily='serif')
            if j == self.n-1 - 1:
                ax.set_xlabel('Time (s)', fontsize=25, fontfamily='serif')

            ax = axs[j, 1]
            ax.plot(self.t, self.Qe[j, 1, :])
            ax.plot(self.t, [self.Qe_d(t)[j, 1] for t in self.t], color='black', linestyle='--')
            ax.set_xlim(0, self.tf)
            ax.set_ylim(ylim)
            if j == 0:
                ax.set_title('y coordinates', fontsize=25, fontfamily='serif')
            if j == self.n-1 - 1:
                ax.set_xlabel('Time (s)', fontsize=25, fontfamily='serif')
        plt.savefig(f"{self.dir}/edges.pdf", format='pdf')
        plt.savefig(f"{self.dir}/edges.png", format='png')

        # Lyapunov Function
        V = 0.5*self.k_R*la.norm(self.e_R, axis=0)**2 + 0.5*la.norm(self.e_nu, axis=0)**2
        V_dot = np.zeros(len(self.t))
        V_dot_bound = np.zeros(len(self.t))
        
        for idx in range(len(self.t)):
            J = np.multiply(self.Le, self.Qe[:, :, idx]@self.Qe[:, :, idx].T)
            lambda_simplified = la.inv(J) @ np.diagonal(self.Qe_dot[:, :, idx]@self.Qe_dot[:, :, idx].T)
            # X = -np.array([(np.outer(self.Qe[j, :, idx], self.Qe[j, :, idx])/self.l[j]**2)@self.Qe_d_ddot(self.t[idx])[j, :] for j in range(self.n-1)]) - self.Le@np.diag(lambda_simplified)@self.Qe[:, :, idx]
            X = -self.Le@np.diag(lambda_simplified)@self.Qe[:, :, idx]

            V_dot[idx] = -self.k_nu*la.norm(self.e_nu[:, idx])**2
            
            # t1, t2, t3 = 0, 0, 0
            for j in range(self.n-1):
                rej_perp = np.array([-self.Qe[j, 1, idx], self.Qe[j, 0, idx]])
                V_dot[idx] += -self.beta*(self.n-1)*np.abs(self.e_nu[j, idx])**3 + self.e_nu[j, idx]*np.dot(rej_perp, X[j, :])/self.l[j]
                # t1 += np.abs(self.e_nu[j, idx])**3
                # t2 += np.abs(self.e_nu[j, idx])**2
                # t3 += np.abs(self.e_nu[j, idx])
            
            V_dot_bound[idx] = -self.k_nu*la.norm(self.e_nu[:, idx])**2

        print(f"Max of V_dot: {np.max(V_dot[1:])}")
        print(f"Max of V_dot_bound: {np.max(V_dot_bound[1:])}")
        if not np.all(V_dot <= V_dot_bound):
            print("\033[31mV_dot is not bounded by V_dot_bound\033[0m")

        plt.figure(figsize=(10, 5))
        plt.plot(self.t, V, label=r'$V$')
        plt.plot(self.t, V_dot, label=r'$\dot{V}$')
        plt.plot(self.t, V_dot_bound, label=r'$\dot{V}_{\mathrm{bound}}$')
        # plt.xlim(0, self.tf)
        plt.legend()
        plt.savefig(f"{self.dir}/lyapunov.pdf", format='pdf')
        # plt.show()

        # Exponential Stability
        V = np.zeros(len(self.t))
        V_dot = np.zeros(len(self.t))
        V_dot_bound = np.zeros(len(self.t))
        V_exp_bound = np.zeros(len(self.t))
        
        c2 = np.min([
            np.sqrt(self.k_R),
            self.k_nu / (1 + 2*self.beta*(self.n-1)*np.pi*np.max(self.l)),
            self.k_R*self.k_nu / (0.25*self.k_nu**2 + self.k_R*(1 + 2*self.beta*(self.n-1)*np.pi*np.max(self.l)))
        ])*0.5
        print(f"c2: {c2}")

        W1 = 0.5*np.kron(np.array([[self.k_R, c2],
                                   [      c2,  1]]), np.eye(self.n-1))
        W2 = np.kron(np.array([[    self.k_R*c2,                                                 0.5*self.k_nu*c2],
                               [0.5*self.k_nu*c2, self.k_nu - c2*(1 + 2*self.beta*(self.n-1)*np.pi*np.max(self.l))]]), np.eye(self.n-1))
        min_W1 = np.min(la.eigvals(W1))
        max_W1 = np.max(la.eigvals(W1))
        min_W2 = np.min(la.eigvals(W2))
        max_W2 = np.max(la.eigvals(W2))
        alpha = min_W2/max_W1
        
        for idx in range(len(self.t)):
            e = np.concatenate((self.e_R[:, idx], self.e_nu[:, idx]))
            V[idx] = np.dot(e, W1@e)
            V_dot_bound[idx] = -np.dot(e, W2@e)
            
            J = np.multiply(self.Le, self.Qe[:, :, idx]@self.Qe[:, :, idx].T)
            lambda_simplified = la.inv(J) @ np.diagonal(self.Qe_dot[:, :, idx]@self.Qe_dot[:, :, idx].T)
            X = -self.Le@np.diag(lambda_simplified)@self.Qe[:, :, idx]
            for j in range(self.n-1):
                rej_perp = np.array([-self.Qe[j, 1, idx], self.Qe[j, 0, idx]])
                V_dot[idx] += self.k_R*self.e_R[j, idx]*self.e_nu[j, idx] + c2*self.e_nu[j, idx]**2 + (self.e_nu[j, idx] + c2*self.e_R[j, idx])*(-self.k_R*self.e_R[j, idx] - self.k_nu*self.e_nu[j, idx] - self.beta*(self.n-1)*self.e_nu[j, idx]*np.abs(self.e_nu[j, idx]) + np.dot(rej_perp/self.l[j], X[j, :]))

            V0 = V[0]
            V_exp_bound[idx] = V0*np.exp(-alpha*self.t[idx])
                
        print(f"Max of V_dot: {np.max(V_dot)}")
        print(f"Max of V_dot_bound: {np.max(V_dot_bound)}")
        if not np.all(V_dot[1:] <= V_dot_bound[1:]):
            print("\033[31mV_dot is not bounded by V_dot_bound\033[0m")

        # C = 2*self.beta*la.norm(np.max(self.nu_d, axis=1), axis=0)*np.sqrt(self.n-1)
        # print(f"C: {C}")
        # D = self.beta*la.norm(np.max(self.nu_d, axis=1), axis=0)**2
        # print(f"Max of C: {np.max(la.eigvals(np.array([[0, 0.5*c2*C], [0.5*c2*C, C]])))}")

        B = np.max(self.nu_d)
        lmax = np.max(self.l)

        mu = self.beta*(self.n-1)*np.sqrt((B**2*np.sqrt(self.n-1) + 2*c2*np.pi*lmax*np.sqrt(self.n-1)*B)**2 + (B**2*np.sqrt(self.n-1))**2) / min_W2

        # mu_R = np.pi*lmax*np.sqrt(self.n-1)
        mu_nu = (self.n-1)**2*B**2 / (self.k_nu - 2*self.beta*B*(self.n-1))
        # mu_nu = np.max(la.norm(self.nu_d, axis=0))**2*(self.n-1)/self.k_nu

        mu_R = np.max(np.roots((
            -self.k_R*c2,
            self.k_nu*c2*mu_nu + 2*B*c2*(self.n-1)*mu_nu + c2*(self.n-1)**1.5*B**2,
            2*B*(self.n-1)*mu_nu**2 + (self.n-1)**1.5*B**2*mu_nu
        )))

        print(f"mu_R: {mu_R}")
        print(f"mu_nu: {mu_nu}")
        # mu = np.sqrt(mu_R**2 + mu_nu**2)
        print(f"mu: {mu}")
        V_max = max_W1*mu**2
        print(f"V_max: {V_max}")
        b = np.sqrt(max_W1/min_W1)*mu
        print(f"b: {b}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.t, V, label=r'$V$')
        plt.plot(self.t, V_dot, label=r'$\dot{V}$')
        plt.plot(self.t, V_dot_bound, label=r'$\dot{V}_{\mathrm{bound}}$')
        plt.plot(self.t, V_exp_bound, label=r'$V_{\mathrm{exp}}$')
        # plt.xlim(0, 0.1)
        plt.legend()
        plt.savefig(f"{self.dir}/lyapunov2.pdf", format='pdf')
        # plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.t, la.norm(self.e_R, axis=0), label=r'$\|e_R\|_2$')
        plt.plot(self.t, la.norm(self.e_nu, axis=0), label=r'$\|e_\nu\|_2$')
        plt.axhline(mu_R, color='tab:blue', linestyle='--', label=r'$\mu_R$')
        plt.axhline(mu_nu, color='tab:orange', linestyle='--', label=r'$\mu_\nu$')
        # plt.ylim(0, 0.01)
        plt.legend()
        plt.savefig(f"{self.dir}/e.pdf", format='pdf')

    def generate_animation(self, filename: str, title: str, limits: tuple):
        # Parameters
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 15
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=limits[0], ylim=limits[1], aspect='equal')
        ax.set_title(title, fontsize=30)

        # Plot initial position
        points = []
        bars = []
        F_c_arrows = []
        for j in range(self.n-1):
            d_j = self.D[:, j]
            head_idx = np.where(d_j == 1)[0][0]
            tail_idx = np.where(d_j == -1)[0][0]
            x, y = [self.Q_0[tail_idx, 0], self.Q_0[head_idx, 0]], [self.Q_0[tail_idx, 1], self.Q_0[head_idx, 1]]
            bar = ax.arrow(x[0], y[0], x[1]-x[0], y[1]-y[0], length_includes_head=True, lw=2, head_width=0.02, head_length=0.02, fc='k', ec='k')
            bars.append(bar)

        # F_g_arrows = []
        for i in range(self.n):
            if i == 0:
                point, = ax.plot([self.Q[i, 0, 0]], [self.Q[i, 1, 0]], 'o', color='r')
                F_c_arrow = ax.arrow(self.Q[i, 0, 0], self.Q[i, 1, 0], self.f_l(0)[0]/2, self.f_l(0)[1]/2, head_width=0.01, head_length=0.01, fc='r', ec='r')
            else:
                point, = ax.plot([self.Q[i, 0, 0]], [self.Q[i, 1, 0]], 'o', color='b')
                F_c_arrow = ax.arrow(self.Q[i, 0, 0], self.Q[i, 1, 0], self.F_bar[i, 0, 0]/2, self.F_bar[i, 1, 0]/2, head_width=0.01, head_length=0.01, fc='b', ec='b')
            points.append(point)
            # F_g_arrow = ax.arrow(self.Q[i, 0, 0], self.Q[i, 1, 0], self.F_g[i, 0, 0]/10, self.F_g[i, 1, 0]/10, head_width=0.01, head_length=0.01, fc='b', ec='b')
            F_c_arrows.append(F_c_arrow)
            # F_g_arrows.append(F_g_arrow)

        # ax.text(0.05, 0.9, f'time: {t[0]:.1f} s', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5), fontsize=18)

        fig.savefig(filename + "_frame1.png", format='png')

        fps = 30
        def animate(i):
            idx = int(i/(fps*self.dt))
            for i in range(self.n):
                points[i].set_data([self.Q[i, 0, idx]], [self.Q[i, 1, idx]])
            for j in range(self.n-1):
                d_j = self.D[:, j]
                head_idx = np.where(d_j == 1)[0][0]
                tail_idx = np.where(d_j == -1)[0][0]
                x, y = [self.Q[tail_idx, 0, idx], self.Q[head_idx, 0, idx]], [self.Q[tail_idx, 1, idx], self.Q[head_idx, 1, idx]]
                bars[j].set_data(x=x[0], y=y[0], dx=x[1]-x[0], dy=y[1]-y[0])
            
            for i in range(self.n):
                if i == 0:
                    F_c_arrows[i].set_data(x=self.Q[i, 0, idx], y=self.Q[i, 1, idx], dx=self.f_l(self.t[idx])[0]/2, dy=self.f_l(self.t[idx])[1]/2)
                else:
                    F_c_arrows[i].set_data(x=self.Q[i, 0, idx], y=self.Q[i, 1, idx], dx=self.F_bar[i, 0, idx]/2, dy=self.F_bar[i, 1, idx]/2)
                # F_g_arrows[i].set_data(x=self.Q[i, 0, idx], y=self.Q[i, 1, idx], dx=self.F_g[i, 0, idx]/10, dy=self.F_g[i, 1, idx]/10)

        tf_sim = self.tf
        ani = animation.FuncAnimation(fig, animate, frames=fps*tf_sim)
        ffmpeg_writer = animation.FFMpegWriter(fps=fps)
        ani.save(filename + ".mp4", writer=ffmpeg_writer)
        plt.close(fig)

    def generate_animation_v2(self, title: str, limits: tuple, arrows_bool: bool):
        assert np.isclose(limits[0][1]-limits[0][0], limits[1][1]-limits[1][0])
        head_size = 0.01/0.6 * (limits[0][1]-limits[0][0])
        
        # Parameters
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 15
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=limits[0], ylim=limits[1], aspect='equal')
        ax.set_title(title, fontsize=30)

        # Plot initial position
        points = []
        bars = []
        arrows = []
        for j in range(self.n-1):
            d_j = self.D[:, j]
            head_idx = np.where(d_j == 1)[0][0]
            tail_idx = np.where(d_j == -1)[0][0]
            x, y = [self.Q_0[tail_idx, 0], self.Q_0[head_idx, 0]], [self.Q_0[tail_idx, 1], self.Q_0[head_idx, 1]]
            bar, = ax.plot(x, y, 'k-', lw=5)
            bars.append(bar)
        for i in range(self.n):
            if i == 0:
                point, = ax.plot([self.Q[i, 0, 0]], [self.Q[i, 1, 0]], 'o', color='r', markersize=10)
            else:
                point, = ax.plot([self.Q[i, 0, 0]], [self.Q[i, 1, 0]], 'o', color='b', markersize=10)
                if arrows_bool:
                    arrow = ax.arrow(self.Q[i, 0, 0], self.Q[i, 1, 0], self.F_bar[i, 0, 0]/3, self.F_bar[i, 1, 0]/3, head_width=head_size, head_length=head_size, fc='b', ec='b', lw=2, zorder=10)
                    arrows.append(arrow)
            points.append(point)
        f_l_arrow = ax.arrow(self.Q[0, 0, 0], self.Q[0, 1, 0], self.f_l(0)[0]/3, self.f_l(0)[1]/3, head_width=head_size, head_length=head_size, fc='r', ec='r', lw=2, zorder=10)

        text = ax.text(0.05, 0.9, f't = {self.t[0]:.1f} s', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5), fontsize=20)
        
        arrows_txt = "_arrows" if arrows_bool else ""
        fig.savefig(f"{self.dir}/frame1{arrows_txt}.png", format='png')

        fps = 30
        def animate(i):
            idx = int(i/(fps*self.dt))
            for j in range(self.n-1):
                d_j = self.D[:, j]
                head_idx = np.where(d_j == 1)[0][0]
                tail_idx = np.where(d_j == -1)[0][0]
                x, y = [self.Q[tail_idx, 0, idx], self.Q[head_idx, 0, idx]], [self.Q[tail_idx, 1, idx], self.Q[head_idx, 1, idx]]
                bars[j].set_data(x, y)
            f_l_arrow.set_data(x=self.Q[0, 0, idx], y=self.Q[0, 1, idx], dx=self.f_l(self.t[idx])[0]/3, dy=self.f_l(self.t[idx])[1]/3)
            for i in range(self.n):
                points[i].set_data([self.Q[i, 0, idx]], [self.Q[i, 1, idx]])
                if i != 0 and arrows_bool:
                    arrows[i-1].set_data(x=self.Q[i, 0, idx], y=self.Q[i, 1, idx], dx=self.F_bar[i, 0, idx]/3, dy=self.F_bar[i, 1, idx]/3)
            text.set_text(f't = {self.t[idx]:.1f} s')

        tf_sim = self.tf
        ani = animation.FuncAnimation(fig, animate, frames=fps*tf_sim)
        ffmpeg_writer = animation.FFMpegWriter(fps=fps)
        ani.save(f"{self.dir}/video{arrows_txt}.mp4", writer=ffmpeg_writer)
        plt.close(fig)

class SO2:
    def hat(phi):
        return np.array([[0, -phi], [phi, 0]])
    
    def vee(phi_hat: np.ndarray):
        return phi_hat[1, 0]
    
    def exphat(phi):
        return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    
    def logvee(phi: np.ndarray):
        return np.arctan2(phi[1, 0], phi[0, 0])
    
if __name__ == "__main__":
    print(NetworkSimulatorV4.beta_fun(np.array([0.027]*3), np.array([0.3]*2), np.array([[-1, -1],
                                                                                        [ 1,  0],
                                                                                        [ 0,  1]])))