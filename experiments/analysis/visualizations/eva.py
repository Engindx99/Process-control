import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.rl.rl import make_env

import yaml
import numpy as np
import logging
import os


# =========================================================
# LOGGER
# =========================================================
logger = logging.getLogger("EVALUATION")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =========================================================
# LOAD CONFIG
# =========================================================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


# =========================================================
# ENV + MODEL
# =========================================================
env = make_env(cfg, seed=42, rank=0)()
model = PPO.load(cfg['paths']['model_save_path'])


obs, _ = env.reset()

history = {
    "temp": [],
    "fuel": [],
    "fan": [],
    "action_fuel": [],
    "action_fan": []
}


logger.info("Evaluation started (3000 steps)")


# =========================================================
# SIMULATION LOOP
# =========================================================
n_steps = cfg['rl'].get("eval_steps", 3000)

for i in range(n_steps):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    plant = env.unwrapped.plant

    history["temp"].append(plant.temp)
    history["fuel"].append(plant.fuel)
    history["fan"].append(plant.fan)

    history["action_fuel"].append(float(action[0]))
    history["action_fan"].append(float(action[1]))

    if i % 500 == 0:
        logger.info(f"step={i} temp={plant.temp:.2f} reward={reward:.2f}")

    if terminated:
        logger.warning(f"Episode terminated at step {i}")
        break


# =========================================================
# METRICS
# =========================================================
temp_arr = np.array(history["temp"])
setpoint = cfg['system']['setpoint']

mae = np.mean(np.abs(temp_arr - setpoint))
std = np.std(temp_arr)
max_dev = np.max(np.abs(temp_arr - setpoint))

logger.info(f"MAE: {mae:.3f}")
logger.info(f"STD: {std:.3f}")
logger.info(f"MAX DEV: {max_dev:.3f}")


# =========================================================
# PLOTTING
# =========================================================

# ---------------- TEMP + FUEL ----------------
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

ax1.axhline(setpoint, color='red', linestyle='--', linewidth=2, label='SETPOINT')
ax1.plot(history['temp'], label='Temperature', linewidth=1.5)
ax1.set_ylabel("Temp (°C)")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax1.set_title(f"Temperature Control | MAE: {mae:.2f}")

ax2.plot(history['fuel'], label='Fuel')
ax2.set_ylabel("Fuel")
ax2.set_xlabel("Step")
ax2.grid(True, alpha=0.3)
ax2.legend()


# ---------------- ACTIONS ----------------
plt.figure(figsize=(15, 6))

plt.plot(history['action_fuel'], label='Fuel Action')
plt.plot(history['action_fan'], label='Fan Action')
plt.axhline(0, color='black', linewidth=0.5)

plt.title(
    f"RL Actions | limit={cfg['rl'].get('action_limit', 0.18)}"
)

plt.xlabel("Step")
plt.ylabel("Action")
plt.legend()
plt.grid(True, alpha=0.2)


plt.tight_layout()
plt.show()


# =========================================================
# SAVE ARTIFACT
# =========================================================
os.makedirs("data", exist_ok=True)

df = pd.DataFrame(history)
df.to_csv("data/evaluation_results.csv", index=False)

logger.info("Evaluation results saved -> data/evaluation_results.csv")