import os
import numpy as np
import pandas as pd
from casadi import *
import do_mpc
import gymnasium as gym
from stable_baselines3 import PPO
from src.physics.digital_twin import AdvancedRotaryKiln

# Fizik motorunu başlat
kiln_physics = AdvancedRotaryKiln()

def setup_mpc():
    model = do_mpc.model.Model('continuous')
    
    # Değişkenler
    temp = model.set_variable('_x', 'temp')
    oxygen = model.set_variable('_x', 'oxygen')
    fuel = model.set_variable('_u', 'fuel')
    fan = model.set_variable('_u', 'fan') # Senin modelindeki fan_speed
    feed = model.set_variable('_tvp', 'feed')
    t_set = model.set_variable('_tvp', 't_set')

    # Denklemleri Gelişmiş Fizik Motorundan Al
    # NOT: Casadi içinde np.exp yerine exp, np.tanh yerine tanh kullanılır.
    # Bu yüzden get_dynamics içindeki fonksiyonları Casadi'ye uyumlu çağırıyoruz.
    eta = 0.6 + 0.4 * exp(-0.1 * (oxygen - 4)**2) * tanh(fan / 50)
    q_in = fuel * 30000 * eta
    q_loss = kiln_physics.k_conv * (temp - kiln_physics.ambient_temp) + \
             kiln_physics.k_rad * (temp**4 - kiln_physics.ambient_temp**4)
    q_material = feed * kiln_physics.cp * (temp - 100)
    q_fan_cooling = 0.02 * fan * (temp - kiln_physics.ambient_temp)

    dT_dt = (q_in - q_loss - q_material - q_fan_cooling) / (kiln_physics.mass * kiln_physics.cp)
    dO2_dt = 0.01 * (fan - 50) - 0.005 * fuel

    model.set_rhs('temp', dT_dt)
    model.set_rhs('oxygen', dO2_dt)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.settings.n_horizon = 15
    mpc.settings.t_step = 1.0
    mpc.settings.supress_ipopt_output = True
    
    # Hedef: Sıcaklık tutarlılığı + Optimum O2 (%4 senin modelindeki tepe nokta)
    lterm = (model.x['temp'] - model.tvp['t_set'])**2 + (model.x['oxygen'] - 4.0)**2 * 1000
    mpc.set_objective(lterm=lterm, mterm=lterm)
    mpc.set_rterm(fuel=0.1, fan=0.01)

    mpc.bounds['lower', '_u', 'fuel'], mpc.bounds['upper', '_u', 'fuel'] = 0.1, 15.0
    mpc.bounds['lower', '_u', 'fan'], mpc.bounds['upper', '_u', 'fan'] = 10.0, 100.0
    
    tvp_template = mpc.get_tvp_template()
    def tvp_fun(t_now):
        for i in range(16):
            tvp_template['_tvp', i, 't_set'] = 1400.0
            tvp_template['_tvp', i, 'feed'] = 5.0
        return tvp_template
    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()
    return mpc

class RotaryKilnEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.mpc_controller = setup_mpc()
        self.action_space = gym.spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32)
        # Gözlem: [Sıcaklık, Oksijen, Yakıt, Fan, Besleme]
        self.observation_space = gym.spaces.Box(low=0, high=2000, shape=(5,), dtype=np.float32)
        self.state = np.array([1200.0, 3.0, 5.0, 50.0, 5.0], dtype=np.float32)

    def step(self, action):
        temp, o2, _, _, feed = self.state
        t_target = 1400.0 + action[0]
        
        self.mpc_controller.x0 = np.array([temp, o2])
        u_opt = self.mpc_controller.make_step(np.array([temp, o2]))
        fuel_opt, fan_opt = float(u_opt[0]), float(u_opt[1])

        # Gelişmiş Dijital İkizden dinamikleri al (Gerçek sayı simülasyonu)
        dT, dO2 = kiln_physics.get_dynamics(temp, o2, fuel_opt, feed, fan_opt)
        
        new_temp = np.clip(temp + dT + np.random.normal(0, 0.3), kiln_physics.min_temp, kiln_physics.max_temp)
        new_o2 = np.clip(o2 + dO2 + np.random.normal(0, 0.01), 1.0, 10.0)

        reward = -(abs(new_temp - t_target) * 0.1 + abs(new_o2 - 4.0) * 50)
        self.state = np.array([new_temp, new_o2, fuel_opt, fan_opt, feed], dtype=np.float32)
        return self.state, float(reward), False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([1200.0, 3.0, 5.0, 50.0, 5.0], dtype=np.float32)
        return self.state, {}

if __name__ == "__main__":
    env = RotaryKilnEnv()
    if not os.path.exists("outputs"): os.makedirs("outputs")
    
    # Kayıtlı model kontrolü
    model_path = "outputs/advanced_kiln_model"
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        model.save(model_path)
    
    # Test ve Kayıt
    obs, _ = env.reset()
    results = []
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
        results.append({"adim": i, "sicaklik": obs[0], "oksijen": obs[1], "yakit": obs[2], "fan": obs[3], "hedef": 1400.0 + action[0]})

    pd.DataFrame(results).to_csv("outputs/simulasyon_verileri.csv", index=False)
    print("\n--- Gelişmiş fizik verileri kaydedildi. ---")