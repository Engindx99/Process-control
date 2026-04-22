import numpy as np
import pandas as pd
import logging


class RotaryKilnPlant:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.logger = logging.getLogger("KilnPlant")
        self.logger.setLevel(logging.INFO)

        self.reset()

    # =========================
    # RESET
    # =========================
    def reset(self):
        self.temp = 1450.0
        self.o2 = 2.8
        self.co2 = 18.0
        self.pressure = -3.0

        self.fuel = 18.5
        self.fan = 950.0

        # tuned params
        self.k_heat = np.random.normal(1.0, 0.1)
        self.k_leak = 0.02
        self.tau_gas = 6.0
        self.tau_pressure = 18.0

        # thermal
        self.dt = 1.0
        self.C_th = 9000.0

        # noise
        self.noise_o2 = 0.0
        self.noise_temp = 0.0
        self.noise_pressure = 0.0

        self.history_fuel = [self.fuel] * 10

        self.data = []
        self.step_count = 0

        return self.state()

    # =========================
    # STATE
    # =========================
    def state(self):
        return {
            "temp": self.temp,
            "o2": self.o2,
            "co2": self.co2,
            "pressure": self.pressure,
            "fuel": self.fuel,
            "fan": self.fan
        }

    # =========================
    # COMBUSTION EFFICIENCY
    # =========================
    def combustion_eff(self, o2):
        return 1.0 / (1.0 + np.exp(-(o2 - 2.5)))

    # =========================
    # AIR FLOW (bounded physical)
    # =========================
    def air_flow(self):
        x = (self.fan - 950) / 500.0
        return np.clip(0.6 + 0.8 * np.tanh(x), 0.2, 2.0)

    # =========================
    # HEAT LOSS (stable physics)
    # =========================
    def heat_loss(self, T, fan):
        T_ref = 1450.0

        radiation = 0.3 * ((T / T_ref) ** 4 - (298 / T_ref) ** 4)
        convection = 0.000011 * fan * (T - 25)

        return radiation + convection

    # =========================
    # STEP
    # =========================
    def step(self, fuel_cmd, fan_cmd):

        # actuator lag
        self.fuel += 0.1 * (fuel_cmd - self.fuel)
        self.fan += 0.15 * (fan_cmd - self.fan)

        self.history_fuel.append(self.fuel)
        fuel = self.history_fuel.pop(0)

        # =========================
        # AIR FLOW
        # =========================
        air_flow = self.air_flow()
        O2_in = 0.21 * air_flow

        # =========================
        # PRESSURE
        # =========================
        resistance = 0.5 * self.pressure + 0.2 * (self.temp - 1450) / 300
        pressure_target = -3.0 + 2.0 * (air_flow - 1.0) - resistance

        self.pressure += (1.0 / self.tau_pressure) * (pressure_target - self.pressure)

        self.noise_pressure = 0.3 * self.noise_pressure + np.random.normal(0, 0.02)
        self.pressure += self.noise_pressure
        self.pressure = np.clip(self.pressure, -6.0, -1.0)

        # =========================
        # O2 DYNAMICS
        # =========================
        eff = self.combustion_eff(self.o2)

        o2_sink = 0.05 * fuel * eff
        mixing = (2.6 - self.o2) / self.tau_gas

        self.noise_o2 = 0.3 * self.noise_o2 + np.random.normal(0, 0.02)

        self.o2 += 0.1 * (O2_in - o2_sink + mixing + self.k_leak)
        self.o2 += self.noise_o2
        self.o2 = np.clip(self.o2, 0.3, 5.0)

        # =========================
        # COMBUSTION (CRITICAL FIX)
        # =========================
        o2_limit = np.tanh(self.o2 / 2.5 * 3.0)  # saturation

        draft_effect = np.exp(-0.000002 * (self.fan - 950) ** 2)

        fuel_effective = fuel * o2_limit

        combustion = fuel_effective * draft_effect

        heat_gen = combustion * self.k_heat

        # =========================
        # HEAT LOSS
        # =========================
        heat_loss = self.heat_loss(self.temp, self.fan)

        # =========================
        # TEMPERATURE DYNAMICS
        # =========================
        dT = (heat_gen - heat_loss) / self.C_th
        self.temp += self.dt * dT

        self.noise_temp = 0.1 * self.noise_temp + np.random.normal(0, 0.02)
        self.temp += self.noise_temp

        self.temp = np.clip(self.temp, 1200.0, 1650.0)

        # =========================
        # LOG
        # =========================
        self.step_count += 1

        record = self.state()
        record["step"] = self.step_count
        self.data.append(record)

        return record

    # =========================
    # RUN
    # =========================
    def run(self, steps=20000, fuel_cmd=25, fan_cmd=1400):
        self.reset()

        for _ in range(steps):
            self.step(fuel_cmd, fan_cmd)

        return pd.DataFrame(self.data)