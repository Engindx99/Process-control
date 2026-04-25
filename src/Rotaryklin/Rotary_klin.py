import numpy as np

# =========================================================
# ZONE MODEL
# =========================================================
class KilnZone:
    def __init__(self, T0, inertia):
        self.T = T0
        self.inertia = inertia

    def update(self, Q_in, T_up, loss):
        conduction = 0.6 * (T_up - self.T)
        radiation = 2e-11 * ((self.T + 273.15)**4 - 298**4)

        dT = (Q_in + conduction - loss*(self.T - 25) - radiation)
        dT *= (1.2e-5 * self.inertia)

        self.T += dT
        return self.T


# =========================================================
# ACTUATOR MODEL
# =========================================================
class ActuatorSystem:
    def __init__(self):
        self.fuel = 18.5
        self.fan = 950.0
        self.fuel_buf = [18.5]*80
        self.fan_buf = [950.0]*40

    def process(self, fuel_cmd, fan_cmd):
        self.fuel_buf.append(np.clip(fuel_cmd, 0, 40))
        self.fan_buf.append(np.clip(fan_cmd, 500, 1200))

        tf = self.fuel_buf.pop(0)
        tn = self.fan_buf.pop(0)

        self.fuel += 0.01*(tf - self.fuel)
        self.fan  += 0.015*(tn - self.fan)

        return self.fuel, self.fan


# =========================================================
# SIGNAL PROCESSOR
# =========================================================
class SignalProcessor:
    def apply(self, T, o2, pressure, fuel, lhv):
        noise = lambda x: x + np.random.normal(0, 0.02)

        return {
            "T1": noise(T["preheat"]),
            "T2": noise(T["calcination"]),
            "T3": noise(T["burning"]),
            "T4": noise(T["cooling"]),
            "o2": noise(o2),
            "pressure": noise(pressure),
            "fuel": fuel,
            "LHV": lhv
        }


# =========================================================
# MAIN PLANT
# =========================================================
class RotaryKilnPlant:
    def __init__(self):

        self.actuators = ActuatorSystem()
        self.signal = SignalProcessor()

        self.zones = {
            "preheat": KilnZone(850, 0.4),
            "calcination": KilnZone(1100, 0.25),
            "burning": KilnZone(1450, 0.12),
            "cooling": KilnZone(800, 0.3)
        }

        self.o2 = 3.2
        self.pressure = -1.5

        self.gas_T = 1200.0
        self.gas_inertia = 8000.0

    # =====================================================
    def step(self, fuel_cmd, fan_cmd):

        # -------------------------
        # actuator
        # -------------------------
        fuel, fan = self.actuators.process(fuel_cmd, fan_cmd)

        # -------------------------
        # combustion
        # -------------------------
        eta_o2 = 1/(1 + np.exp(-(self.o2 - 2.2)))
        eta_temp = 1/(1 + np.exp(-(self.zones["burning"].T - 1300)/100))

        Q_chem = fuel * 7200 * eta_o2 * eta_temp * 0.6

        # -------------------------
        # gas energy (stable)
        # -------------------------
        heat_sink = 0
        for z in self.zones.values():
            heat_sink += 0.001 * (self.gas_T - z.T)

        self.gas_T += (Q_chem - heat_sink) / self.gas_inertia
        self.gas_T = np.clip(self.gas_T, 800, 2000)

        # -------------------------
        # zones
        # -------------------------
        T = {k: z.T for k, z in self.zones.items()}

        self.zones["burning"].T += 0.01 * (self.gas_T - T["burning"])
        self.zones["calcination"].T += 0.007 * (T["burning"] - T["calcination"])
        self.zones["preheat"].T += 0.005 * (T["calcination"] - T["preheat"])
        self.zones["cooling"].T += 0.004 * (T["burning"] - T["cooling"])

        # -------------------------
        # gas chemistry
        # -------------------------
        air = 2.0/(1 + np.exp(-(fan - 900)/200))

        self.o2 += 0.04*(0.21*air - 0.02*fuel - self.o2)
        self.o2 = np.clip(self.o2, 0.5, 5.5)

        self.pressure += 0.05*((-2.5 + 1.5*(air-1)) - self.pressure)

        # -------------------------
        # quality
        # -------------------------
        fcao = np.clip(2.5 - 0.0025*self.zones["burning"].T, 0.2, 3.5)

        # -------------------------
        # output
        # -------------------------
        return self.signal.apply(
            T=self.zones,
            o2=self.o2,
            pressure=self.pressure,
            fuel=fuel,
            lhv=7200
        )