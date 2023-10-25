import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(10000):
    if done:
        env.reset()
    state, reward, done, info, _ = env.step(env.action_space.sample())
    env.render()
env.close()
