import pygame
import numpy as np
import random
import sys
import os

STEP = 20
APPLE_SIZE = 20
SCREEN_SIZE = 321
ACTIONS = 4
START_X = 5 * STEP#SCREEN_SIZE / 2 - 1 * STEP
START_Y = 5 * STEP#SCREEN_SIZE / 2 - STEP

BACKGROUND_COLOR = (0, 0, 0)
SNAKE_COLOR = (255, 255, 255)
APPLE_COLOR = (255, 255, 255)

SCREENSHOT_DIMS = (84, 84)

def init_snake():
    """
    Restores the game to the initial state.
    """
    global xs, ys, dirs, score, episode_length, episode_reward, applepos, s, \
        action, state, next_state, must_die
    xs = [START_Y,
          START_Y,
          START_Y,
          START_Y,
          START_Y]
    ys = [START_X + 5 * STEP,
          START_X + 4 * STEP,
          START_X + 3 * STEP,
          START_X + 2 * STEP,
          START_X]
    dirs = random.choice([0, 1, 3])
    score = 0
    episode_length = 0
    episode_reward = 0
    must_die = False
    applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE),
                random.randint(0, SCREEN_SIZE - APPLE_SIZE))

    # The direction is randomly selected
    action = random.randint(0, ACTIONS - 1)
    # Initialize the states
    #state = [screenshot(), screenshot()]
    #next_state = [screenshot(), screenshot()]

    # Redraw game surface
    s.fill(BACKGROUND_COLOR)
    for ii in range(0, len(xs)):
        s.blit(snake_image, (xs[ii], ys[ii]))
    s.blit(apple_image, applepos)
    pygame.display.update()

def collide(x1, x2, y1, y2, w1, w2, h1, h2):
    """
    Returns True if the positions of the two object are the same after applying
    the movements.
    :param x1: x of object 1
    :param x2: x of object 2
    :param y1: y of object 1
    :param y2: y of object 2
    :param w1: horizontal movement of object 1
    :param w2: horizontal movement of object 2
    :param h1: vertical movement of object 1
    :param h2: vertical movement of object 2
    """
    return x1 + w1 > x2 and x1 < x2 + w2 and y1 + h1 > y2 and y1 < y2 + h2

def die():
    pygame.display.update()
    init_snake()

xs = [START_Y,
      START_Y,
      START_Y,
      START_Y,
      START_Y]
ys = [START_X + 5 * STEP,
      START_X + 4 * STEP,
      START_X + 3 * STEP,
      START_X + 2 * STEP,
      START_X]

dirs = random.choice([0, 1, 3])
must_die = False
applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE),
            random.randint(0, SCREEN_SIZE - APPLE_SIZE))

pygame.init()
s = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Snake")
apple_image = pygame.Surface((APPLE_SIZE, APPLE_SIZE))
apple_image.fill(APPLE_COLOR)
snake_image = pygame.Surface((STEP, STEP))
snake_image.fill(SNAKE_COLOR)
clock = pygame.time.Clock()

userControls = True
action = random.randint(0, ACTIONS - 1)

while True:
    clock.tick()

    events = pygame.event.get()
    if(userControls):
        for event in events:
            if event.type == pygame.QUIT:
                sys.exit(0)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 3
                elif event.key == pygame.K_DOWN:
                    action = 0
                elif event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 1
        # Change direction according to the action

    if action == 2 and dirs != 0:  # up
        dirs = 2
    elif action == 0 and dirs != 2:  # down
        dirs = 0
    elif action == 3 and dirs != 1:  # left
        dirs = 3
    elif action == 1 and dirs != 3:  # right
        dirs = 1

        # Check if snake ate apple
    if collide(xs[0], applepos[0],
               ys[0], applepos[1],
               STEP, APPLE_SIZE,
               STEP, APPLE_SIZE):
        #score += 1
        #reward = len(xs) if APPLE_REWARD is None else APPLE_REWARD
        xs.append(1)
        ys.append(1)
        applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE),
                    random.randint(0, SCREEN_SIZE - APPLE_SIZE))  # Ate apple

    i = len(xs) - 1
    while i >= 2:
        if collide(xs[0], xs[i],
                   ys[0], ys[i],
                   STEP, STEP,
                   STEP, STEP):
            must_die = True
            #reward = DEATH_REWARD  # Hit itself
        i -= 1

    # La serpiente choca la pared?
    if xs[0] < 0 or xs[0] > SCREEN_SIZE - APPLE_SIZE * 1 or ys[0] < 0 or ys[0] > SCREEN_SIZE - APPLE_SIZE * 1:
        must_die = True


    i = len(xs) - 1
    while i >= 1:
        xs[i] = xs[i - 1]
        ys[i] = ys[i - 1]
        i -= 1

    if dirs == 0:
        ys[0] += STEP
    elif dirs == 1:
        xs[0] += STEP
    elif dirs == 2:
        ys[0] -= STEP
    elif dirs == 3:
        xs[0] -= STEP

    s.fill(BACKGROUND_COLOR)
    for i in range(0, len(xs)):
        s.blit(snake_image, (xs[i], ys[i]))
    s.blit(apple_image, applepos)
    pygame.display.update()

    pygame.time.delay(100)
    if must_die: #or episode_length > len(xs) * MAX_EPISODE_LENGTH_FACTOR:
        die()