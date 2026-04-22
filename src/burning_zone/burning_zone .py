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
        self.temp = 1450.0          # °C
        self.o2 = 2.8               # %
        self.co2 = 18.0            # %
        self.pressure = -3.0       # mbar (realistic draft)

        self.fuel = 18.5
        self.fan = 950.0

        # process parameters
        self.k_heat = np.random.normal(22.0, 1.0)
        self.k_air = np.random.normal(0.02, 0.002)
        self.k_leak = np.random.normal(0.02, 0.005)
        self.tau_gas = np.random.normal(6.0, 0.8)

        # dynamics
        self.tau_pressure = 18.0

        # noise states
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
    # COMBUSTION EFF
    # =========================
    def combustion_eff(self, o2):
        return 1.0 / (1.0 + np.exp(-(o2 - 2.5)))

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
        air_flow = 1.0 + 0.002 * (self.fan - 950)

        # =========================
        # PRESSURE (REALISTIC DRAFT MODEL)
        # =========================
        resistance = 0.6 * self.pressure + 0.25 * (self.temp - 1450) / 300

        pressure_target = -3.0 + 2.5 * (air_flow - 1.0) - resistance

        self.pressure += (1.0 / self.tau_pressure) * (pressure_target - self.pressure)

        self.noise_pressure = 0.4 * self.noise_pressure + np.random.normal(0, 0.02)
        self.pressure += self.noise_pressure

        self.pressure = np.clip(self.pressure, -6.0, -1.0)

        # =========================
        # O2
        # =========================
        O2_in = 0.21 * air_flow
        eff = self.combustion_eff(self.o2)

        o2_sink = 0.055 * fuel * eff
        mixing = (2.6 - self.o2) / self.tau_gas

        self.noise_o2 = 0.4 * self.noise_o2 + np.random.normal(0, 0.02)

        self.o2 += 0.08 * (O2_in - o2_sink + mixing + self.k_leak)
        self.o2 += self.noise_o2
        self.o2 = np.clip(self.o2, 0.5, 5.0)

        # =========================
        # CO2
        # =========================
        self.co2 += 0.05 * (fuel * eff - self.co2 / 18.0)

        # =========================
        # TEMPERATURE
        # =========================
        baseline_heat = 754

        heat_gen = fuel * self.k_heat * eff + baseline_heat
        heat_loss = 0.0004 * self.fan * (self.temp - 25)

        net_heat = (heat_gen - heat_loss) / 3200

        self.temp += 0.004 * net_heat

        self.noise_temp = 0.4 * self.noise_temp + np.random.normal(0, 0.04)
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
    def run(self, steps=1000, fuel_cmd=18, fan_cmd=950):
        self.reset()

        for _ in range(steps):
            self.step(fuel_cmd, fan_cmd)

        return pd.DataFrame(self.data)