import numpy as np
import matplotlib.pyplot as plt

SIGMA = 5.67e-8


class RotaryKilnFeedbackModel:
    def __init__(self, N=50, dt=0.1):

        self.N = N
        self.dt = dt

        self.Cp_s = 1000
        self.Cp_g = 1200

        self.m_s = 50.0
        self.m_g = 20.0

        self.h = 25.0
        self.A = 2.0
        self.eps = 0.85

        self.Ts = np.ones(N) * 1100
        self.Tg = np.ones(N) * 1400

        self.fuel = 1.0

        self.Kp = 0.02
        self.Ki = 0.001
        self.error_int = 0.0

        self.Tset = 1450

    def convection(self, i):
        return self.h * self.A * (self.Tg[i] - self.Ts[i])

    def radiation(self, i):
        return self.eps * SIGMA * self.A * (self.Tg[i]**4 - self.Ts[i]**4)

    def combustion(self):
        return 5e5 * self.fuel

    def feedback_control(self):
        error = self.Tset - np.mean(self.Ts)
        self.error_int += error * self.dt

        self.fuel = self.Kp * error + self.Ki * self.error_int
        self.fuel = np.clip(self.fuel, 0.2, 3.0)

    def step(self):

        self.feedback_control()

        Ts_new = self.Ts.copy()
        Tg_new = self.Tg.copy()

        Qcomb = self.combustion()

        for i in range(self.N):

            Qconv = self.convection(i)
            Qrad = self.radiation(i)

            dTs = (Qconv + Qrad) / (self.m_s * self.Cp_s)
            dTg = (Qcomb - Qconv - Qrad) / (self.m_g * self.Cp_g)

            Ts_new[i] += self.dt * dTs
            Tg_new[i] += self.dt * dTg

        self.Ts = Ts_new
        self.Tg = Tg_new

        return self.Ts, self.Tg, self.fuel


# =========================================================
# SIMULATION + VISUALIZATION
# =========================================================

model = RotaryKilnFeedbackModel(N=60, dt=0.1)

steps = 300

Ts_hist = []
Tg_hist = []
fuel_hist = []

x = np.linspace(0, 60, model.N)

plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

for t in range(steps):

    Ts, Tg, fuel = model.step()

    Ts_hist.append(np.mean(Ts))
    Tg_hist.append(np.mean(Tg))
    fuel_hist.append(fuel)

    # =========================
    # 1) Spatial profile
    # =========================
    axs[0].cla()
    axs[0].plot(x, Ts, label="Solid Temp")
    axs[0].plot(x, Tg, label="Gas Temp")
    axs[0].set_title("Kiln Temperature Profile")
    axs[0].set_ylabel("Temperature (K)")
    axs[0].legend()
    axs[0].grid()

    # =========================
    # 2) Time evolution
    # =========================
    axs[1].cla()
    axs[1].plot(Ts_hist, label="Ts mean")
    axs[1].plot(Tg_hist, label="Tg mean")
    axs[1].axhline(model.Tset, linestyle="--", color="red", label="Setpoint")
    axs[1].set_title("Temperature Tracking")
    axs[1].set_ylabel("K")
    axs[1].legend()
    axs[1].grid()

    # =========================
    # 3) Control signal
    # =========================
    axs[2].cla()
    axs[2].plot(fuel_hist, label="Fuel Input")
    axs[2].set_title("Feedback Control Signal")
    axs[2].set_ylabel("Fuel")
    axs[2].set_xlabel("Time step")
    axs[2].legend()
    axs[2].grid()

    plt.pause(0.01)

plt.ioff()
plt.show()