# import game and action space
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# import preprocessing tools
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# import AI model and tools to load it
import os
from stable_baselines3 import DQN

# import tools to plot result
import re
import matplotlib.pyplot as plt

# file paths for models and logs
MODEL_DIR = "D:\gymarb\DQNmodels"
# number of games to average over
NUMBER_OF_RUNS = 10

# resolves compatibility issues caused by version mismatches
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
# setup environment
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# preprocess frame data
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

# points to plot
points_x = []
points_y = []

# load each model
for file in os.listdir(MODEL_DIR):
    model = DQN.load(os.path.join(MODEL_DIR, file))
    total_reward = 0

    for _ in range(NUMBER_OF_RUNS):
        # reset state
        state = env.reset()
        done = False
        # run game until done
        while not done:
            # choose action
            action, _ = model.predict(state)
            # advance to next state
            state, reward, done, info, _ = env.step(action)
            # increment reward
            total_reward += reward
            env.render()
    # divide for average
    total_reward /= NUMBER_OF_RUNS

    # use iteration number as x value
    points_x.append(int(re.search("(?<=best_model_)[0-9]+(?=.zip)")))
    # use result as y value
    points_y.append(total_reward)

# plot all values
plt.plot(points_x, points_y)
plt.show()
