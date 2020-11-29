import pygame
from pygame.locals import *
from PIL import Image, ImageOps
from DQA import DQAgent
import numpy as np
import random
import sys
import os
import argparse
from Logger import Logger

parser = argparse.ArgumentParser()

MAX_EPISODE_LENGTH_FACTOR = 100
MAX_EPISODES_BETWEEN_TRAININGS = 1500
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

    # Direccion elegida de manera random.
    action = random.randint(0, ACTIONS - 1)
    # Initialize the states
    state = [screenshot(), screenshot()]
    next_state = [screenshot(), screenshot()]

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

def screenshot():
    """
    Takes a screenshot of the game, converts it to greyscale, resizes it to
    60x60 and returns it as np.array
    :return:
    """
    global s, is_headless
    data = pygame.image.tostring(s, 'RGB')  # Captura de pantalla
    image = Image.frombytes('RGB', (SCREEN_SIZE, SCREEN_SIZE), data)
    image = image.convert('L')  # Escala de grises
    image = image.resize(SCREENSHOT_DIMS)  # Ajustamos el size
    image = ImageOps.invert(image) if is_headless else image  # TODO ???
    image = image.convert('1')
    matrix = np.asarray(image.getdata(), dtype=np.float64)
    return matrix.reshape(image.size[0], image.size[1])

def die():
    global logger, remaining_iters, score, episode_length, episode_reward, \
        must_test, experience_buffer, exp_backup_counter, global_episode_counter

    global_episode_counter += 1

    # If agent is stuck, kill the process
    if global_episode_counter > MAX_EPISODES_BETWEEN_TRAININGS:
        logger.log('Shutting process down because something seems to have gone '
                   'wrong during training. Please manually check that '
                   'all is OK and restart the training with the -l flag.')
        DQA.quit()
        sys.exit(0)

    # Antes de probar, debemos guardas datos.
    if must_test:
        logger.to_csv('test_data.csv', [score, episode_length, episode_reward])
        logger.log('Test episode - Score: %s; Steps: %s'
                   % (score, episode_length))

    # Reset, cada fase de entranmiento tiene un testing
    must_test = False

    # Agregar el episodio a las experiencias
    if score >= 1 and episode_length >= 10:
        exp_backup_counter += len(experience_buffer)
        print(f"Adding episode to experiences - Score: {score}; Episode length: {episode_length}")
        logger.to_csv('train_data.csv', [score, episode_length, episode_reward])
        print(f"Got {exp_backup_counter} samples of {DQA.batch_size}")
        for exp in experience_buffer:
            DQA.add_experience(*exp)

    # Entrenamiento
    if DQA.must_train() and args.train:
        exp_backup_counter = 0
        logger.log('Episodes elapsed: %d' % global_episode_counter)
        global_episode_counter = 0
        # Quit at the last iteration
        if remaining_iters == 0:
            DQA.quit()
            sys.exit(0)

        # Entrenar el DQN
        DQA.train()

        remaining_iters -= 1 if remaining_iters != -1 else 0
        # Despues de entrenar, el siguiente episodio sera un test
        must_test = True
        logger.log('Test episode')

    # Limpio experiencias
    experience_buffer = []

    pygame.display.update()
    init_snake()

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true',
                    help='train the agent')
parser.add_argument('-l', '--load', type=str, required=False, default='',
                    help='load the DQN weights from disk')
parser.add_argument('-i', '--iterations', type=int, required=False, default=-1,
                    help='number of training iterations before quitting')
parser.add_argument('-v', '--novideo', action='store_true',
                    help='suppress video output')
parser.add_argument('-d', '--debug', action='store_true',
                    help='do not print anything to file and do not create the '
                         'output folder.')
parser.add_argument('--gamma', type=float, required=False, default=0.95,
                    help='discount factor for the MDP')
parser.add_argument('--dropout', type=float, required=False, default=0.1,
                    help='dropout rate for the DQN')
parser.add_argument('--reward', type=str, required=False, default='1,-1,0',
                    help='comma separated list representing rewards for apple, '
                         'death and life. Pass \'N\' as apple reward to use the'
                         ' current snake length.')
args = parser.parse_args()

remaining_iters = args.iterations

# Controlar salida de video
is_headless = args.novideo
if is_headless:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    BACKGROUND_COLOR = (255, 255, 255)  # Pygame acts weird on headless servers

# Recompensas
rewards = args.reward.split(',')
if rewards[0] is 'N':
    APPLE_REWARD = None
else:
    APPLE_REWARD = float(rewards[0])
DEATH_REWARD = float(rewards[1])
LIFE_REWARD = float(rewards[2])

# Logger
logger = Logger(debug=args.debug)
logger.log({
    'Action space': ACTIONS,
    'Reward apple': 'snake lenght' if APPLE_REWARD is None else APPLE_REWARD,
    'Reward death': DEATH_REWARD,
    'Reward life': LIFE_REWARD
})
logger.to_csv('test_data.csv', ['score,episode_length,episode_reward'])
logger.to_csv('train_data.csv', ['ecore,episode_length,episode_reward'])
logger.to_csv('loss_history.csv', ['loss'])

# Agente
DQA = DQAgent(
    ACTIONS, # 4 Acciones
    gamma=args.gamma,
    dropout_prob=args.dropout,
    load_path=args.load,
    logger=logger
)
experience_buffer = []  # This will store the SARS tuples at each episode

# Estadisticas
score = 0
episode_length = 0
episode_reward = 0
episode_nb = 0
exp_backup_counter = 0
global_episode_counter = 0  # Keeps track of how many episodes there were between traning iterations
must_test = False

# Inicializar variables de juego
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

action = random.randint(0, ACTIONS - 1)

# Estados iniciales
state = [screenshot(), screenshot()]
next_state = [screenshot(), screenshot()]

while True:
    episode_length += 1
    reward = LIFE_REWARD # Recompensa por mantenerse vivo, pero no comer
    next_state[0] = state[1]

    clock.tick()

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            DQA.quit()
            sys.exit(0)

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
        score += 1
        reward = len(xs) if APPLE_REWARD is None else APPLE_REWARD
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
            reward = DEATH_REWARD  # Choco consigo mismo
        i -= 1

    # La serpiente choca la pared?
    if xs[0] < 0 or xs[0] > SCREEN_SIZE - APPLE_SIZE * 1 or ys[0] < 0 or ys[0] > SCREEN_SIZE - APPLE_SIZE * 1:
        must_die = True
        reward = DEATH_REWARD # Choco la pared

    # Movimiento
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

    # Actualizar siguiente estado
    next_state[1] = screenshot()

    # Experiencia actualizada
    experience_buffer.append((np.asarray([state]), action, reward,
                              np.asarray([next_state]),
                              True if must_die else False))
    episode_reward += reward

    # Cambiar el estado actual
    state = list(next_state)

    # Pedir la siguiente accion
    action = DQA.get_action(np.asarray([state]), testing=must_test)

    #pygame.time.delay(100)
    if must_die or episode_length > len(xs) * MAX_EPISODE_LENGTH_FACTOR:
        die()



