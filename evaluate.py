# import game and action space
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# import preprocessing tools
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# import AI model and tools to load it
import os
from stable_baselines3 import A2C

# file paths for models and logs
MODEL_DIR = ".\\A2Cmodels"
OUTPUT_FILE = ".\\plots\\A2C.txt"
MAX_ITERATIONS = 400 * 20
# number of games to average over
NUMBER_OF_RUNS = 3

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

# points to plot
points = []

# load each model
for file in os.listdir(MODEL_DIR):
    model = A2C.load(os.path.join(MODEL_DIR, file))
    total_reward = 0

    for _ in range(NUMBER_OF_RUNS):
        # reset state
        state = env.reset()
        for _ in range(MAX_ITERATIONS):
            # choose action
            action, _ = model.predict(state)
            # advance to next state
            state, reward, done, info = env.step(action)
            # increment reward
            total_reward += reward
            # env.render()

            if done:
                break

    # divide for average
    total_reward = round(total_reward[0] / NUMBER_OF_RUNS, 3)

    iter_num = int(file.removeprefix("best_model_").removesuffix(".zip"))
    print(f"Finished model number {iter_num}")

    points.append((iter_num, total_reward))

points.sort(key=lambda p: p[0])
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(f"{x} {y}\n" for x, y in points)
