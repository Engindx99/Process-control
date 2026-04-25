import numpy as np

class KilnPlant:

    def __init__(self, N=60):

        self.N = N
        self.dt = 0.05
        self.dx = 1.0

        # states
        self.Ts = np.ones(N) * 1100
        self.Tg = np.ones(N) * 1400
        self.Xc = np.zeros(N)

        # physical params (scaled)
        self.Cps = 1000
        self.Cpg = 900          # 🔥 reduced inertia (important)
        self.h = 22

        self.vs = 0.03
        self.vg = 1.0

        self.k0 = 1.5
        self.E = 75000
        self.R = 8.314
        self.dH = 1.5e5

        # 🔥 stronger actuator gain
        self.Qcomb_base = 5.5e5

        # 🔥 actuator dynamics (VERY IMPORTANT INDUSTRIAL FEATURE)
        self.u_filt = 0.7
        self.tau_u = 3.0

    def reaction(self, T, X):
        T = np.clip(T, 600, 2000)
        return self.k0 * np.exp(-self.E / (self.R * T)) * (1 - X)

    def advect(self, T, v):
        T_new = T.copy()
        cfl = np.clip(v * self.dt / self.dx, 0, 0.7)

        for i in range(1, self.N):
            T_new[i] = T[i] - cfl * (T[i] - T[i-1])

        return T_new

    def step(self, u):

        # 🔥 actuator first-order lag (CRITICAL)
        u = np.clip(u, 0.0, 1.0)
        self.u_filt += (self.dt/self.tau_u) * (u - self.u_filt)

        self.Ts = self.advect(self.Ts, self.vs)
        self.Tg = self.advect(self.Tg, self.vg)

        Tg_mean = np.mean(self.Tg)

        for i in range(self.N):

            r = self.reaction(self.Ts[i], self.Xc[i])

            Qrxn = -self.dH * r
            Qconv = self.h * (self.Tg[i] - self.Ts[i])

            damp = 1.0 / (1.0 + (Tg_mean - 1500)/700)
            Qcomb = self.Qcomb_base * self.u_filt * damp

            T = np.clip(self.Tg[i], 300, 2000)
            Qrad = 3e-8 * (T**4 - 300**4)

            self.Ts[i] += self.dt * (Qconv + Qrxn) / self.Cps
            self.Tg[i] += self.dt * (Qcomb - Qconv - Qrad) / self.Cpg
            self.Xc[i] += self.dt * r

        self.Xc = np.clip(self.Xc, 0, 1)

        return self.Ts.copy(), self.Tg.copy(), self.Xc.copy()