import gym_super_mario_bros as smb
from nes_py.wrappers import JoypadSpace
import pygame


def powerset(s):
    for i in range(1, 1 << len(s)):
        yield [e for j, e in enumerate(s) if i >> j & 1]


FPS = 60
RESOLUTION = (256, 240)
ACTIONS = ["up", "down", "right", "left", "A", "B"]
ACTION_SPACE = [["NOOP"], *powerset(ACTIONS)]
KEYBINDS = {
    pygame.K_w: "up",
    pygame.K_s: "down",
    pygame.K_a: "left",
    pygame.K_d: "right",
    pygame.K_o: "A",
    pygame.K_p: "B",
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
    pygame.K_x: "A",
    pygame.K_z: "B",
}

env = smb.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="rgb_array"
)
env = JoypadSpace(env, ACTION_SPACE)

pygame.init()
screen = pygame.display.set_mode(RESOLUTION)
clock = pygame.time.Clock()

total_reward = 0


def stop():
    print("\nYour reward: ", total_reward)
    env.close()
    pygame.quit()
    exit()


state = env.reset()
keys = set()
done = False
while not done:

    for event in pygame.event.get():
        match event.type:
            case pygame.QUIT:
                stop()
            case pygame.KEYDOWN:
                if event.key in KEYBINDS:
                    keys.add(KEYBINDS[event.key])
            case pygame.KEYUP:
                if event.key in KEYBINDS:
                    keys.remove(KEYBINDS[event.key])

    action = sum(1 << ACTIONS.index(key) for key in keys)
    state, reward, done, truncate, info = env.step(action)
    total_reward += reward

    frame = env.render()
    surface = pygame.pixelcopy.make_surface(frame.transpose(1, 0, 2))
    screen.blit(surface, (0, 0))
    pygame.display.update()

    clock.tick(FPS)

stop()