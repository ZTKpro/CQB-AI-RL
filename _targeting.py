import pygame
import sys
import math
import random
import pickle
import numpy as np
from collections import deque

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

# Q-learning settings
MODEL_FILE = './models/targeting.pkl'
ACTION_SPACE = ['rotate_left', 'rotate_right', 'shoot']
ALPHA = 1.0  # Learning rate
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate for exploration
BATCH_SIZE = 256  # Batch size for experience replay

# Experience replay buffer
REPLAY_BUFFER_SIZE = 100000
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

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
PLAYER_ROTATION_SPEED = 5  # Rotation speed

# Text settings
FONT_SIZE = 36
font = pygame.font.Font(None, FONT_SIZE)

# Player class
class Player:
    def __init__(self, x, y, color, is_ai=False):
        self.rect = pygame.Rect(x, y, PLAYER_SIZE, PLAYER_SIZE)
        self.color = color
        self.angle = 0
        self.rotation_speed = PLAYER_ROTATION_SPEED
        self.bullets = []
        self.is_ai = is_ai
        self.reward = 0
        self.last_shot_time = 0
        self.previous_state = None
        self.previous_action = None

    def move(self, enemy):
        if self.is_ai:
            self.ai_move(enemy)
        self.draw(screen)

    def ai_move(self, enemy):
        angle_difference = self.angle_to_enemy(enemy)
        # Reward increases as angle difference decreases
        self.reward += (180 - angle_difference) / 180

        current_state = self.extract_state(enemy)
        action = self.choose_action(current_state)
        self.perform_action(action)

        # Store experience in replay buffer
        if self.previous_state is not None and self.previous_action is not None:
            replay_buffer.append((self.previous_state, self.previous_action, self.reward, current_state))

        # Experience replay
        if len(replay_buffer) >= BATCH_SIZE:
            for _ in range(10):
                self.experience_replay()

        self.previous_state = current_state
        self.previous_action = action

    def choose_action(self, state):
        if random.uniform(0, 1) < EPSILON:
            return random.choice(ACTION_SPACE)  # Explore
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = {action: q_table.get((state, action), 0) for action in ACTION_SPACE}
            max_q = max(q_values.values())
            actions_with_max_q = [action for action, q in q_values.items() if q == max_q]
            return random.choice(actions_with_max_q)

    def perform_action(self, action):
        if action == 'rotate_left':
            self.angle = (self.angle - self.rotation_speed) % 360
        elif action == 'rotate_right':
            self.angle = (self.angle + self.rotation_speed) % 360
        elif action == 'shoot':
            self.shoot()

    def extract_state(self, enemy):
        angle_difference = self.angle_to_enemy(enemy)
        angle_discrete = int(angle_difference / 10)
        return (angle_discrete,)

    def angle_to_enemy(self, enemy):
        dx = enemy.rect.centerx - self.rect.centerx
        dy = enemy.rect.centery - self.rect.centery
        angle_to_enemy = (math.degrees(math.atan2(dy, dx))) % 360
        angle_difference = (angle_to_enemy - self.angle + 360) % 360
        if angle_difference > 180:
            angle_difference = 360 - angle_difference
        return angle_difference

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time > 200:  # Cooldown
            angle_rad = math.radians(self.angle)
            bullet_dx = BULLET_SPEED * math.cos(angle_rad)
            bullet_dy = BULLET_SPEED * math.sin(angle_rad)
            bullet = Bullet(self.rect.centerx, self.rect.centery, bullet_dx, bullet_dy, BULLET_COLOR)
            self.bullets.append(bullet)
            self.last_shot_time = current_time

    def update_bullets(self, enemies):
        for bullet in self.bullets[:]:
            bullet.move()
            if bullet.is_off_screen():
                self.bullets.remove(bullet)
                if self.is_ai:
                    self.reward -= 10  # Penalty for missing a shot
            else:
                for enemy in enemies:
                    if bullet.rect.colliderect(enemy.rect):
                        self.bullets.remove(bullet)
                        if self.is_ai:
                            self.reward += 500  # Reward for hitting the enemy

    def experience_replay(self):
        batch = random.sample(replay_buffer, BATCH_SIZE)
        for previous_state, action, reward, current_state in batch:
            best_future_q = max(q_table.get((current_state, a), 0) for a in ACTION_SPACE)
            old_q = q_table.get((previous_state, action), 0)
            q_table[(previous_state, action)] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

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
        pygame.draw.polygon(viewing_surface, (0, 255, 0, 20), points, 0)
        surface.blit(viewing_surface, (0, 0))
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

# Create AI players and enemies
players = [Player(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), PLAYER_COLOR, is_ai=True) for _ in range(10)]
enemies = [Enemy(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), ENEMY_COLOR) for _ in range(3)]

# Main game loop
clock = pygame.time.Clock()
game_start_time = pygame.time.get_ticks()
total_rewards = []

while True:
    # Reset game after 10 seconds
    if pygame.time.get_ticks() - game_start_time > 10000:
        best_player = max(players, key=lambda p: p.reward)
        total_rewards.append(round(best_player.reward))
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(q_table, f)
        print(f'Game Over. Best Player Total Reward: {best_player.reward}')
        game_start_time = pygame.time.get_ticks()
        for player in players:
            player.reward = 0  # Reset reward after game end
            player.rect.x, player.rect.y = random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50)
        enemies = [Enemy(random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50), ENEMY_COLOR) for _ in range(3)]
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)  # Decay epsilon
        continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(q_table, f)
            pygame.quit()
            sys.exit()

    # Update game state
    for player in players:
        player.move(random.choice(enemies))
        player.update_bullets(enemies)

    # Draw
    screen.fill(WHITE)
    for player in players:
        player.draw(screen)
    for enemy in enemies:
        enemy.draw(screen)

    best_player = max(players, key=lambda p: p.reward)
    reward_text = font.render(f"Best Player Reward: {best_player.reward}", True, (0, 0, 0))
    screen.blit(reward_text, (5, 40))

    pygame.display.flip()
    clock.tick(60)  # FPS