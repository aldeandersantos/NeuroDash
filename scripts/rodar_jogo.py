import os
import sys
import pygame
import time

projeto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, projeto_dir)

from utils.model_manager import ModelManager
from utils.ai_controller import AIController
from utils.game_setup import configurar_ambiente, resetar_ambiente, executar_passo

TEMPO_DE_JOGO = 120 # Segundos
MAX_FPS = 60
COLETA_EXPERIENCIAS = True
TREINAR_APOS_JOGO = True

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
    ultima_obs = None
    ultima_acao = None
    
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
        
        if COLETA_EXPERIENCIAS:
            ultima_obs = obs.copy()
        
        action = ai_controller.processar_acao(env, model, obs, frame_count)
        
        if COLETA_EXPERIENCIAS:
            ultima_acao = action
        
        nova_obs, done = executar_passo(env, action)
        
        if COLETA_EXPERIENCIAS and ultima_obs is not None and ultima_acao is not None:
            reward = 0.1
            if done:
                reward = -10
                
            model_manager.adicionar_experiencia(
                ultima_obs, ultima_acao, reward, nova_obs, done
            )
        
        obs = nova_obs
        
        visualizer.display_fps()
        visualizer.display_score(int(score))
        visualizer.display_text(f"Experiências: {len(model_manager.experiencias_buffer)}", (10, 70))
        pygame.display.update()
        
        clock.tick(MAX_FPS)
        
        if done:
            obs = resetar_ambiente(env)
            score = 0
    
    print("Fim do jogo!")
    
    if COLETA_EXPERIENCIAS:
        model_manager.salvar_experiencias()
    
    if TREINAR_APOS_JOGO and len(model_manager.experiencias_buffer) > 0:
        print("Treinando modelo com novas experiências...")
        experiencias = model_manager.carregar_experiencias()
        
        if experiencias and len(experiencias) > 0:
            if confirmar_treinamento():
                model_manager.treinar_modelo_incremental(experiencias)
                print("Treinamento concluído!")
    
    pygame.quit()
    env.close()

def confirmar_treinamento():
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 32)
    
    dialog_width, dialog_height = 500, 200
    dialog = pygame.display.set_mode((dialog_width, dialog_height))
    pygame.display.set_caption("Treinar modelo?")
    
    running = True
    treinar = False
    
    while running:
        dialog.fill((240, 240, 240))
        
        text1 = font.render("Deseja treinar o modelo com as novas experiências?", True, (0, 0, 0))
        text2 = font.render("S - Sim | N - Não", True, (0, 0, 0))
        
        text1_rect = text1.get_rect(center=(dialog_width/2, dialog_height/2 - 30))
        text2_rect = text2.get_rect(center=(dialog_width/2, dialog_height/2 + 30))
        
        dialog.blit(text1, text1_rect)
        dialog.blit(text2, text2_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    treinar = True
                    running = False
                elif event.key == pygame.K_n:
                    running = False
        
        time.sleep(0.05)
    
    return treinar
