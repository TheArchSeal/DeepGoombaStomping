# import game and action space
import gym_super_mario_bros as smb
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# import preprocessing tools
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# import AI model
from stable_baselines3 import DQN as Model

# resolve compatibility issues caused by version mismatches
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# setup environment
env = smb.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# preprocess frame data
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

# setup model
model = Model("CnnPolicy", env, verbose=1, buffer_size=20000)

# run model
model.learn(total_timesteps=10000000)
