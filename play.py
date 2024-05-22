import gym_super_mario_bros as smb
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT as ActionSpace
import pygame

FPS = 60
RESOLUTION = (256, 240)
KEYBINDS = {
    pygame.K_RIGHT: "right",
    pygame.K_LEFT: "left",
    pygame.K_UP: "A",
}

env = smb.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="rgb_array"
)
env = JoypadSpace(env, ActionSpace)

pygame.init()
screen = pygame.display.set_mode(RESOLUTION)
clock = pygame.time.Clock()

total_reward = 0


def quit():
    env.close()
    pygame.quit()
    print("\nYour reward: ", total_reward)
    exit()


state = env.reset()
keys = set()
done = False
while not done:

    for event in pygame.event.get():
        match event.type:
            case pygame.QUIT:
                quit()
            case pygame.KEYDOWN:
                if event.key in KEYBINDS:
                    keys.add(KEYBINDS[event.key])
            case pygame.KEYUP:
                if event.key in KEYBINDS:
                    keys.remove(KEYBINDS[event.key])

    action = 0
    for i, a in enumerate(ActionSpace):
        if keys == set(a):
            action = i
            break

    state, reward, done, truncate, info = env.step(action)
    total_reward += reward

    frame = env.render()
    surface = pygame.pixelcopy.make_surface(frame.transpose(1, 0, 2))
    screen.blit(surface, (0, 0))
    pygame.display.update()

    clock.tick(FPS)

quit()
