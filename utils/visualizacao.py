import pygame

class GameVisualizer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", 24)
        self.clock = pygame.time.Clock()

    def display_fps(self):
        fps = self.clock.get_fps()
        fps_text = self.font.render(f"FPS: {fps:.2f}", True, (0, 0, 0))
        self.screen.blit(fps_text, (10, 10))

    def tick(self):
        self.clock.tick(60)  # FPS
