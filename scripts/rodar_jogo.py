import os
import sys
import pygame

projeto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projeto_dir)

from utils.model_manager import ModelManager
from utils.ai_controller import AIController
from utils.game_setup import configurar_ambiente, resetar_ambiente, executar_passo

TEMPO_DE_JOGO = 120 # Segundos
MAX_FPS = 60

def main():
    model_manager = ModelManager(projeto_dir)
    model = model_manager.carregar_modelo()
    
    env, visualizer = configurar_ambiente()
    ai_controller = AIController()
    
    obs = resetar_ambiente(env)
    clock = pygame.time.Clock()
    
    running = True
    frame_count = 0
    score = 0
    
    print("Iniciando jogo...")
    tempo_de_jogo = (TEMPO_DE_JOGO * MAX_FPS)
    while running and frame_count < tempo_de_jogo:
        frame_count += 1
        score += 0.1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        action = ai_controller.processar_acao(env, model, obs, frame_count)
        obs, done = executar_passo(env, action)
        
        visualizer.display_fps()
        visualizer.display_score(int(score))
        pygame.display.update()
        
        clock.tick(MAX_FPS)
        
        if done:
            obs = resetar_ambiente(env)
            score = 0
    
    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
