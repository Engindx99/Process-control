import casadi as ca
import numpy as np
from src.Rotaryklin.Rotary_klin import KilnPlant

class CasADiMPC:

    def __init__(self, N=20):

        self.N = N
        self.opti = ca.Opti()

        # =========================
        # VARIABLES
        # =========================
        U  = self.opti.variable(N)
        dU = self.opti.variable(N)   # 🔥 rate
        Ts = self.opti.variable(N+1)
        Tg = self.opti.variable(N+1)
        X  = self.opti.variable(N+1)
        Z  = self.opti.variable(N+1)

        # =========================
        # PARAMETERS
        # =========================
        Ts0 = self.opti.parameter()
        Tg0 = self.opti.parameter()
        X0  = self.opti.parameter()
        Z0  = self.opti.parameter()
        Tref = self.opti.parameter()
        U_prev = self.opti.parameter()   # 🔥 warm start reference

        # =========================
        # CONST
        # =========================
        dt = 0.05
        Cps = 1000
        Cpg = 1200
        h = 18

        k0 = 1.5
        E = 75000
        R = 8.314
        dH = 1.5e5
        Qcomb_base = 2.5e5

        # =========================
        # INITIAL CONDITIONS
        # =========================
        self.opti.subject_to(Ts[0] == Ts0)
        self.opti.subject_to(Tg[0] == Tg0)
        self.opti.subject_to(X[0]  == X0)
        self.opti.subject_to(Z[0]  == Z0)

        cost = 0

        # =========================
        # MPC LOOP
        # =========================
        for k in range(N):

            T_safe = ca.fmax(650, ca.fmin(1900, Ts[k]))

            # reaction
            r = k0 * ca.exp(-E / (R * T_safe)) * (1 - X[k])
            r = ca.fmin(r, 4.0)

            Qrxn = -dH * r
            Qconv = h * (Tg[k] - Ts[k])

            # 🔥 smoother combustion (IMPORTANT FIX)
            damp = ca.exp(-(Tg[k]-1500)**2 / (2*400**2))
            Qcomb = Qcomb_base * U[k] * damp

            # radiation
            Qrad = 2.5e-8 * (Tg[k]**4 - 300**4)

            # dynamics
            Ts_next = Ts[k] + dt * (Qconv + Qrxn) / Cps
            Tg_next = Tg[k] + dt * (Qcomb - Qconv - Qrad) / Cpg
            X_next  = X[k] + dt * r
            Z_next  = Z[k] + dt * (Tg[k] - Tref)

            # constraints
            self.opti.subject_to(Ts[k+1] == Ts_next)
            self.opti.subject_to(Tg[k+1] == Tg_next)
            self.opti.subject_to(X[k+1]  == ca.fmin(1.0, X_next))
            self.opti.subject_to(Z[k+1]  == Z_next)

            # =========================
            # CONTROL RATE
            # =========================
            if k == 0:
                self.opti.subject_to(dU[k] == U[k] - U_prev)
            else:
                self.opti.subject_to(dU[k] == U[k] - U[k-1])

            # =========================
            # COST (CRITICAL TUNING)
            # =========================
            cost += 3.0 * (Tg[k] - Tref)**2      # tracking
            cost += 0.02 * U[k]**2              # fuel penalty
            cost += 2.5 * dU[k]**2              # 🔥 ANTI-OSCILLATION (MOST IMPORTANT)
            cost += 0.05 * Z[k]**2              # offset removal

        self.opti.minimize(cost)

        # =========================
        # HARD CONSTRAINTS
        # =========================
        self.opti.subject_to(self.opti.bounded(0.2, U, 0.9))
        self.opti.subject_to(self.opti.bounded(-0.15, dU, 0.15))  # 🔥 smooth actuator

        # =========================
        # SOLVER SETTINGS (IMPORTANT)
        # =========================
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 120,
            "ipopt.warm_start_init_point": "yes"
        }

        self.opti.solver("ipopt", opts)

        # store
        self.U = U
        self.dU = dU
        self.Ts0 = Ts0
        self.Tg0 = Tg0
        self.X0 = X0
        self.Z0 = Z0
        self.Tref = Tref
        self.U_prev = U_prev

    # =========================
    # SOLVE
    # =========================
    def solve(self, Ts, Tg, X, z, u_prev, Tref=1450):

        self.opti.set_value(self.Ts0, Ts)
        self.opti.set_value(self.Tg0, Tg)
        self.opti.set_value(self.X0, X)
        self.opti.set_value(self.Z0, z)
        self.opti.set_value(self.Tref, Tref)
        self.opti.set_value(self.U_prev, u_prev)

        try:
            sol = self.opti.solve()
            u = float(sol.value(self.U[0]))
        except:
            u = u_prev  # 🔥 fail-safe (no jump)

        z_new = z + 0.05 * (Tg - Tref)

        return u, z_new