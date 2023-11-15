import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


CHECKPOINT_DIR = ".\\models\\"
LOG_DIR = ".\\logs\\"


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)
        return True


JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.000001,
    # n_steps=512,
    buffer_size=10000,
)

model.learn(total_timesteps=100000, callback=callback)
