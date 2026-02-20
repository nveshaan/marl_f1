import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

model = DQN(
    "CnnPolicy", 
    env,
    buffer_size=200000,
    verbose=1,
    device="mps"
)
# model.learn(total_timesteps=1000000, log_interval=4)
# model.save("models/baseline_dqn")
model.load("models/baseline_dqn")

env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

obs, info = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    env.render()