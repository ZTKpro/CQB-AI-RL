import pygame
import sys
import math
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH, SCREEN_HEIGHT = 500, 500
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("CQB AI Game")

# Color definitions
WHITE = (255, 255, 255)
PLAYER_COLOR = (0, 0, 255)
BULLET_COLOR = (255, 255, 0)
ENEMY_COLOR = (255, 0, 0)
WALL_COLOR = (50, 50, 50)

# Q-learning settings
MODEL_FILE = './models/base_stage1.pkl'
ACTION_SPACE = ['rotate_left', 'rotate_right', 'shoot']
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate

# Load or initialize Q-table
try:
    with open(MODEL_FILE, 'rb') as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    q_table = {}  # Initialize empty Q-table

# Bullet settings
BULLET_SPEED = 20
BULLET_SIZE = 5

# Viewing area settings
VIEW_DISTANCE = 500
VIEW_ANGLE = 90  # The total angle of the player's field of view
# Player settings
PLAYER_SIZE = 20
PLAYER_SPEED = 3
PLAYER_ROTATION_SPEED = 4

# Text settings
FONT_SIZE = 36
font = pygame.font.Font(None, FONT_SIZE)

# Player class
class Player:
    def __init__(self, x, y, color, is_ai=False):
        self.sensors = []
        self.rect = pygame.Rect(x, y, PLAYER_SIZE, PLAYER_SIZE)
        self.color = color
        self.angle = 0
        self.speed = PLAYER_SPEED
        self.rotation_speed = PLAYER_ROTATION_SPEED
        self.bullets = []
        self.is_ai = is_ai
        self.reward = 0
        self.last_shot_time = 0
        self.previous_state = None
        self.previous_action = None

    def move(self, walls, enemy):
        
        if self.is_ai:
            self.ai_move(walls, enemy)
        self.draw(screen)

    def ai_move(self, walls, enemy):
        current_state = self.extract_state(enemy)
        action = self.choose_action(current_state)
        self.perform_action(action)

        # Update Q-table
        if self.previous_state is not None and self.previous_action is not None:
            reward = self.reward
            best_future_q = max(q_table.get((current_state, a), 0) for a in ACTION_SPACE)
            old_q = q_table.get((self.previous_state, self.previous_action), 0)
            q_table[(self.previous_state, self.previous_action)] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

        self.previous_state = current_state
        self.previous_action = action

        # Reward is adjusted during specific actions, but not reset after each move

    def choose_action(self, state):
        if random.uniform(0, 1) < EPSILON:
            return random.choice(ACTION_SPACE)  # Explore
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = {action: q_table.get((state, action), 0) for action in ACTION_SPACE}
            return max(q_values, key=q_values.get)

    def perform_action(self, action):
        if action == 'rotate_left':
            self.angle = (self.angle - self.rotation_speed) % 360
        elif action == 'rotate_right':
            self.angle = (self.angle + self.rotation_speed) % 360
        elif action == 'shoot':
            self.shoot()

    def extract_state(self, enemy):
        angle_to_enemy = self.angle_to_enemy(enemy)
        return (int(angle_to_enemy / 10),)  # Discretize the angle to enemy

    def angle_to_enemy(self, enemy):
        dx = enemy.rect.centerx - self.rect.centerx
        dy = enemy.rect.centery - self.rect.centery
        angle = math.degrees(math.atan2(dy, dx))
        return ((angle - self.angle) + 360) % 360 if ((angle - self.angle) + 360) % 360 < 180 else ((angle - self.angle) + 360) % 360 - 360

    def distance_to(self, enemy):
        dx = enemy.rect.centerx - self.rect.centerx
        dy = enemy.rect.centery - self.rect.centery
        return math.sqrt(dx ** 2 + dy ** 2)

    def is_enemy_in_view(self, enemy):
        angle_to_enemy = abs(self.angle_to_enemy(enemy))
        return angle_to_enemy <= 5

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time > 500:
            angle_rad = math.radians(self.angle)
            bullet_dx = BULLET_SPEED * math.cos(angle_rad)
            bullet_dy = BULLET_SPEED * math.sin(angle_rad)
            bullet = Bullet(self.rect.centerx, self.rect.centery, bullet_dx, bullet_dy, BULLET_COLOR)
            self.bullets.append(bullet)
            self.last_shot_time = current_time

    def update_bullets(self, walls, enemy):
        for bullet in self.bullets[:]:
            bullet.move()
            if bullet.is_off_screen():
                self.bullets.remove(bullet)
            else:
                for wall in walls:
                    if bullet.rect.colliderect(wall):
                        self.bullets.remove(bullet)
                        break
                if bullet in self.bullets and bullet.rect.colliderect(enemy.rect):
                    self.bullets.remove(bullet)
                    if self.is_ai:
                        self.reward += round(500)  # Large reward for successfully hitting the enemy


    def draw(self, surface):
        # Draw the viewing area
        start_angle = math.radians(self.angle - VIEW_ANGLE / 2)
        end_angle = math.radians(self.angle + VIEW_ANGLE / 2)
        points = [self.rect.center]
        for angle in np.linspace(start_angle, end_angle, num=50):
            x = self.rect.centerx + VIEW_DISTANCE * math.cos(angle)
            y = self.rect.centery + VIEW_DISTANCE * math.sin(angle)
            points.append((x, y))
        viewing_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(viewing_surface, (0, 255, 0, 20), points, 0)  # Increased transparency
        surface.blit(viewing_surface, (0, 0))        
        reward_text = font.render(f"Reward: {round(self.reward)}", True, (0, 0, 0))
        surface.blit(reward_text, (10, 70))
        laser_end_x = self.rect.centerx + 1000 * math.cos(math.radians(self.angle))
        laser_end_y = self.rect.centery + 1000 * math.sin(math.radians(self.angle))
        pygame.draw.line(surface, (255, 0, 0), self.rect.center, (laser_end_x, laser_end_y), 1)
        if pygame.Rect(laser_end_x, laser_end_y, 1, 1).colliderect(enemy.rect):
            if self.is_ai:
                self.reward += 100  # Bonus for directly targeting the enemy
        pygame.draw.rect(surface, self.color, self.rect)
        for bullet in self.bullets:
            bullet.draw(surface)

# Bullet class
class Bullet:
    def __init__(self, x, y, dx, dy, color):
        self.rect = pygame.Rect(x, y, BULLET_SIZE, BULLET_SIZE)
        self.dx = dx
        self.dy = dy
        self.color = color

    def move(self):
        self.rect.x += round(self.dx)
        self.rect.y += round(self.dy)

    def is_off_screen(self):
        return not (0 <= self.rect.x <= SCREEN_WIDTH and 0 <= self.rect.y <= SCREEN_HEIGHT)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

# Enemy class
class Enemy:
    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, PLAYER_SIZE, PLAYER_SIZE)
        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

# Create players, enemy, and walls in a small room
players = [Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, PLAYER_COLOR, is_ai=True) for _ in range(5)]
enemy = Enemy(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), ENEMY_COLOR)
walls = [
    pygame.Rect(0, 0, SCREEN_WIDTH, 10),  # Top wall
    pygame.Rect(0, SCREEN_HEIGHT - 10, SCREEN_WIDTH, 10),  # Bottom wall
    pygame.Rect(0, 0, 10, SCREEN_HEIGHT),  # Left wall
    pygame.Rect(SCREEN_WIDTH - 10, 0, 10, SCREEN_HEIGHT)  # Right wall
]

# Main game loop
clock = pygame.time.Clock()
game_start_time = pygame.time.get_ticks()
total_rewards = []

while True:
    # Reset game after 10 seconds
    if len(total_rewards) > 0 and len(total_rewards) % 10 == 0:
        plt.plot(total_rewards)
        plt.xlabel('Game Number')
        plt.ylabel('Total Reward')
        plt.title('AI Learning Progress Over Time')
        plt.savefig('learning_progress.png')
        plt.close()
    if pygame.time.get_ticks() - game_start_time > 10000:
        best_player = max(players, key=lambda p: p.reward)
        total_rewards.append(round(best_player.reward))
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(q_table, f)
        print(f'Game Over. Best Reward: {best_player.reward}')
        game_start_time = pygame.time.get_ticks()
        for player in players:
            player.reward = 0  # Reset reward after game end
            player.rect.x, player.rect.y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        enemy = Enemy(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), ENEMY_COLOR)
        continue
    if pygame.time.get_ticks() - game_start_time > 10000:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(q_table, f)
        pygame.quit()
        sys.exit()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(q_table, f)
            pygame.quit()
            sys.exit()

    # Update game state
    for player in players:
        player.move(walls, enemy)
        player.update_bullets(walls, enemy)

    # Draw
    screen.fill(WHITE)
    for wall in walls:
        pygame.draw.rect(screen, WALL_COLOR, wall)
    for player in players:
        player.draw(screen)
    enemy.draw(screen)

    best_player = max(players, key=lambda p: p.reward)
    reward_text = font.render(f"Best Player Reward: {best_player.reward}", True, (0, 0, 0))
    stage_text = font.render(f"Model Stage: Q-Learning", True, (0, 0, 0))
    screen.blit(reward_text, (10, 40))
    screen.blit(stage_text, (10, 100))

    pygame.display.flip()
    clock.tick(30)