# import game and action space
import gym_super_mario_bros as smb
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT as ActionSpace

# import preprocessing tools
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# import AI model
from stable_baselines3 import DQN as Model


class HighJumpSpace(JoypadSpace):
    # add new high jump action
    _button_map = JoypadSpace._button_map | {"AA": 0b100000000}

    # new parameter for number of frames to high jump
    def __init__(self, env, actions, high_jump_length):
        super().__init__(env, actions)
        self.high_jump_length = high_jump_length

    def step(self, action):
        byte_action = self._action_map[action]
        # check if action contains high jump
        if byte_action & self._button_map["AA"]:
            # replace high jump with regular jump
            byte_action = byte_action & ~self._button_map["AA"] | self._button_map["A"]
            # repeat jump action multiple frames
            for _ in range(self.high_jump_length - 1):
                state, reward, done, truncated, info = self.env.step(byte_action)
                if done:  # don't step after done
                    return (state, reward, done, truncated, info)

        return self.env.step(byte_action)

    # resolve compatibility issues caused by version mismatches
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# add high jump versions of all jump actions
ActionSpace.extend(
    ["AA" if key == "A" else key for key in action]
    for action in ActionSpace
    if "A" in action
)

# setup environment
env = smb.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = HighJumpSpace(env, ActionSpace, 10)

# preprocess frame data
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

# setup model
model = Model("CnnPolicy", env, verbose=1, buffer_size=10000)

# run model
model.learn(total_timesteps=10000000)
