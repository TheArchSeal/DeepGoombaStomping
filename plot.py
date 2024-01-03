import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import os
from stable_baselines3 import DQN

import re
import matplotlib.pyplot as plt

MODEL_DIR = "D:\gymarb\DQNmodels"


JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

points = []

for file in os.listdir(MODEL_DIR):
    model = DQN.load(os.path.join(MODEL_DIR, file.removesuffix(".zip")))
    state = env.reset()

    done = False
    cumulative_reward = 0

    while not done:
        action, _ = model.predict(state)
        state, reward, done, info, _ = env.step(action)
        cumulative_reward += reward
        env.render()

    x = int(re.search("(?<=best_model_)[0-9]+(?=.zip)"))
    y = cumulative_reward
    points.append((x, y))

plt.plot(points)
plt.show()
