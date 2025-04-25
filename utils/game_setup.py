import pygame
from env.jogo_com_obstaculos import JogoComObstaculos
from utils.visualizacao import GameVisualizer

def configurar_ambiente(screen_width=800, screen_height=600):
    pygame.init()
    pygame.display.set_mode((screen_width, screen_height), 
                           flags=pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    env = JogoComObstaculos(render_mode="human")
    
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
    
    return env, visualizer

def resetar_ambiente(env):
    reset_return = env.reset()
    if isinstance(reset_return, tuple):
        return reset_return[0]
    else:
        return reset_return

def executar_passo(env, action):
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
    
    return obs, done
