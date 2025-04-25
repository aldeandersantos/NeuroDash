import gym
import numpy as np
from gym import spaces
import pygame
import random

class JogoComObstaculos(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(JogoComObstaculos, self).__init__()

        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(
            low=np.array([0, -1, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float32
        )

        self.width = 800
        self.height = 600
        self.agent_width = 40
        self.agent_height = 40
        self.agent_x = 100
        self.ground_y = self.height - self.agent_height
        self.agent_y = self.ground_y
        self.agent_speed = 5
        self.gravity = 0.8
        self.jump_power = 15
        self.agent_vy = 0

        self.obstacle_speed = 7
        self.min_obstacle_gap = 250
        self.obstacles = []

        self.screen = None
        self.clock = None
        self.render_mode = render_mode

    def _get_obs(self):
        next_obstacle = None
        min_dist = float('inf')
        for obstacle in self.obstacles:
            if obstacle['x'] + obstacle['width'] > self.agent_x:
                dist = obstacle['x'] - (self.agent_x + self.agent_width)
                if dist < min_dist:
                    min_dist = dist
                    next_obstacle = obstacle

        if next_obstacle:
            norm_agent_y = self.agent_y / self.height
            norm_agent_vy = np.clip(self.agent_vy / (self.jump_power * 1.5), -1, 1)
            norm_dist_obst = np.clip(min_dist / self.width, 0, 1)
            norm_obst_y_top = next_obstacle['y'] / self.height
            norm_obst_height = next_obstacle['height'] / self.height
            obs = np.array([norm_agent_y, norm_agent_vy, norm_dist_obst, norm_obst_y_top, norm_obst_height], dtype=np.float32)
        else:
            norm_agent_y = self.agent_y / self.height
            norm_agent_vy = np.clip(self.agent_vy / (self.jump_power * 1.5), -1, 1)
            obs = np.array([norm_agent_y, norm_agent_vy, 1.0, 0.5, 0.1], dtype=np.float32)

        return obs

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_y = self.ground_y
        self.agent_vy = 0
        self.obstacles = []
        last_obstacle_x = self.width + random.randint(100, 300)
        for _ in range(5):
            new_obstacle = self.create_obstacle(min_x=last_obstacle_x + self.min_obstacle_gap)
            self.obstacles.append(new_obstacle)
            last_obstacle_x = new_obstacle['x']

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            pygame.display.set_caption("NeuroDash Obstacle Env")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def create_obstacle(self, min_x):
        x = min_x + random.randint(0, 200)
        height = random.randint(30, 80)
        y = self.height - height
        width = random.randint(30, 50)
        return {'x': x, 'y': y, 'width': width, 'height': height}

    def step(self, action):
        is_on_ground = (self.agent_y >= self.ground_y)
        
        if action == 2 and is_on_ground:
            self.agent_vy = -self.jump_power
        
        self.agent_vy += self.gravity
        
        max_fall_speed = 15
        if self.agent_vy > max_fall_speed:
            self.agent_vy = max_fall_speed
            
        self.agent_y += self.agent_vy
        
        if self.agent_y >= self.ground_y:
            self.agent_y = self.ground_y
            self.agent_vy = 0
            
        min_height = 50
        if self.agent_y < min_height:
            self.agent_y = min_height
            self.agent_vy = 0
            
        last_obstacle_x = 0
        if self.obstacles:
            last_obstacle_x = max(o['x'] for o in self.obstacles)

        for obstacle in self.obstacles:
            obstacle['x'] -= self.obstacle_speed

        self.obstacles = [obs for obs in self.obstacles if obs['x'] + obs['width'] > 0]

        if not self.obstacles or last_obstacle_x < self.width - self.min_obstacle_gap:
            min_x = max(self.width, last_obstacle_x + self.min_obstacle_gap if self.obstacles else self.width)
            new_obstacle = self.create_obstacle(min_x=min_x)
            self.obstacles.append(new_obstacle)

        terminated = False
        reward = 0.1

        agent_rect = pygame.Rect(self.agent_x, self.agent_y, self.agent_width, self.agent_height)
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(obstacle['x'], obstacle['y'], obstacle['width'], obstacle['height'])
            if agent_rect.colliderect(obstacle_rect):
                terminated = True
                reward = -10
                break

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.screen is None or self.clock is None:
            return

        self.screen.fill((135, 206, 235))

        pygame.draw.rect(self.screen, (34, 139, 34), (0, self.ground_y + self.agent_height, self.width, self.height - (self.ground_y + self.agent_height)))

        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, (200, 0, 0), (obstacle['x'], obstacle['y'], obstacle['width'], obstacle['height']))

        pygame.draw.rect(self.screen, (0, 0, 200), (self.agent_x, self.agent_y, self.agent_width, self.agent_height))

        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
