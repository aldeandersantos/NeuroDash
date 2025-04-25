import os
import sys

projeto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projeto_dir)

import pygame
from stable_baselines3 import DQN
from env.jogo_com_obstaculos import JogoComObstaculos
from utils.visualizacao import GameVisualizer

model_path = os.path.join(projeto_dir, "models/dqn_jogo_obstaculos.zip")
model_path_v2 = os.path.join(projeto_dir, "models/dqn_jogo_obstaculos_v2.zip")

def treinar_modelo():
    env = JogoComObstaculos(render_mode=None)
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0005,
        buffer_size=50000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05
    )
    model.learn(total_timesteps=25000)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    return model

pygame.init()
pygame.display.set_mode((800, 600), flags=pygame.HWSURFACE | pygame.DOUBLEBUF)

if os.path.exists(model_path_v2):
    model = DQN.load(model_path_v2)
elif os.path.exists(model_path):
    model = DQN.load(model_path)
else:
    model = treinar_modelo()

env = JogoComObstaculos(render_mode="human")

screen_width, screen_height = 800, 600
if not hasattr(env, 'screen') or env.screen is None:
    screen = pygame.display.set_mode((screen_width, screen_height), 
                                    flags=pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("NeuroDash")
    env.screen = screen
    env.render_mode = "human"
else:
    screen = env.screen

if not hasattr(env, 'clock') or env.clock is None:
    env.clock = pygame.time.Clock()

visualizer = GameVisualizer(screen)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

reset_return = env.reset()
if isinstance(reset_return, tuple):
    obs = reset_return[0]
else:
    obs = reset_return

clock = pygame.time.Clock()
running = True
frame_count = 0
max_fps = 60
log_interval = 60

while running and frame_count < 1000:
    frame_count += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
    action, _states = model.predict(obs, deterministic=True)
    if not hasattr(env, '_ultimo_pulo'):
        env._ultimo_pulo = 0
        env._pulos_excessivos = 0
    if action == 1:
        if frame_count - env._ultimo_pulo < 20:
            obstaculo_proximo = False
            if hasattr(env, 'obstaculos'):
                for obj in env.obstaculos:
                    if 0 < obj.x - env.jogador.x < 200:
                        obstaculo_proximo = True
                        break
            if not obstaculo_proximo:
                action = 0
                env._pulos_excessivos += 1
        else:
            env._ultimo_pulo = frame_count
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    elif len(result) == 4:
        obs, reward, done, info = result
    else:
        obs = result[0]
        reward = result[1]
        done = result[2]
        info = {}
    if hasattr(env, 'render') and not hasattr(env, '_rendered_current_frame'):
        env.render()
        env._rendered_current_frame = True
    clock.tick(max_fps)
    if done:
        reset_return = env.reset()
        if isinstance(reset_return, tuple):
            obs = reset_return[0]
        else:
            obs = reset_return

pygame.quit()
env.close()
