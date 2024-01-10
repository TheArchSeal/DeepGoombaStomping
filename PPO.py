# import game and action space
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# import preprocessing tools
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# import AI model and tools to automatically save it
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# file paths for models and logs
CHECKPOINT_DIR = ".\\PPOmodels\\"
LOG_DIR = ".\\logs\\"


# callback class to automatically save models
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


# resolves compatibility issues caused by version mismatches
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
# setup environment
env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True  # , render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# preprocess frame data
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

# setup model and auto saves
callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.000001,
)

# run the model
model.learn(total_timesteps=5000000, callback=callback)
