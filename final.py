import pygame
import sys
import math
import random
import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("CQB AI Game")

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
WALL_COLOR = (50, 50, 50)
FLOOR_COLOR = (200, 200, 200)
PLAYER_COLOR = (0, 0, 255)
BULLET_COLOR = (255, 255, 0)
ENEMY_COLOR = (255, 0, 0)
ENEMY_BULLET_COLOR = (255, 0, 255)

# Load or initialize machine learning model
try:
    with open('ai_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1, warm_start=True)
    model.fit([[0] * 8], [0])  # Initialize with dummy data

# Player class
class Player:
    def __init__(self, x, y, color, controls, is_ai=False):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.color = color
        self.angle = 0  # Rotation angle in degrees
        self.speed = 3
        self.rotation_speed = 5
        self.view_distance = 400  # Increased view distance
        self.view_angle = 60  # Field of view angle
        self.controls = controls  # Dictionary of assigned keys
        self.bullets = []
        self.is_ai = is_ai
        self.reward = 0
        self.last_shot_time = 0
        self.last_kill_time = pygame.time.get_ticks()
        self.previous_state = None
        self.previous_reward = 0

    def move(self, walls, enemies):
        self.previous_position = (self.rect.x, self.rect.y)
        if self.is_ai:
            self.ai_move(walls, enemies)
        else:
            self.manual_move(walls)

    def manual_move(self, walls):
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0

        if keys[self.controls['left']]:
            dx = -self.speed
        if keys[self.controls['right']]:
            dx = self.speed
        if keys[self.controls['up']]:
            dy = -self.speed
        if keys[self.controls['down']]:
            dy = self.speed

        # Horizontal movement with collisions
        self.rect.x += dx
        for wall in walls:
            if self.rect.colliderect(wall):
                if dx > 0:
                    self.rect.right = wall.left
                if dx < 0:
                    self.rect.left = wall.right

        # Vertical movement with collisions
        self.rect.y += dy
        for wall in walls:
            if self.rect.colliderect(wall):
                if dy > 0:
                    self.rect.bottom = wall.top
                if dy < 0:
                    self.rect.top = wall.bottom

        # Rotate player
        if keys[self.controls['rotate_left']]:
            self.angle = (self.angle - self.rotation_speed) % 360
        if keys[self.controls['rotate_right']]:
            self.angle = (self.angle + self.rotation_speed) % 360

    def ai_move(self, walls, enemies):
        # Extract features for current state
        features = self.extract_features(walls, enemies)

        # Predict action based on current state
        action = model.predict([features])[0]

        # Perform the action
        self.angle = (self.angle + action * self.rotation_speed) % 360
        dx = self.speed * math.cos(math.radians(self.angle))
        dy = self.speed * math.sin(math.radians(self.angle))

        self.rect.x += int(dx)
        if (dx == 0 and dy == 0):
            self.reward -= 0.1  # Penalty for standing still
        for wall in walls:
            if self.rect.colliderect(wall):
                if dx > 0:
                    self.rect.right = wall.left
                if dx < 0:
                    self.rect.left = wall.right

        self.rect.y += int(dy)
        for wall in walls:
            if self.rect.colliderect(wall):
                if dy > 0:
                    self.rect.bottom = wall.top
                if dy < 0:
                    self.rect.top = wall.bottom

        # Reward for facing enemies
        for enemy in enemies:
            if self.is_enemy_in_view(enemy):
                self.reward += 0.1  # Reward for keeping enemy in view

        # Randomly shoot bullets based on model prediction
        if random.random() < 0.05:
            self.shoot()

        # Update model with previous action and reward
        if self.previous_state is not None:
            model.partial_fit([self.previous_state], [self.previous_reward])

        # Update previous state and reward
        self.previous_state = features
        self.previous_reward = self.reward

    def extract_features(self, walls, enemies):
        # Features: player position, angle, distance to closest enemy, angle to closest enemy
        closest_enemy = min(enemies, key=lambda enemy: self.distance_to(enemy))
        distance_to_enemy = self.distance_to(closest_enemy)
        angle_to_enemy = self.angle_to_enemy(closest_enemy)
        return [
            self.rect.centerx / SCREEN_WIDTH,
            self.rect.centery / SCREEN_HEIGHT,
            self.angle / 360,
            distance_to_enemy / self.view_distance,
            angle_to_enemy / 360,
            len(walls),
            len(enemies),
            self.reward
        ]

    def angle_to_enemy(self, enemy):
        dx = enemy.rect.centerx - self.rect.centerx
        dy = enemy.rect.centery - self.rect.centery
        return (math.degrees(math.atan2(dy, dx)) - self.angle) % 360

    def is_enemy_in_view(self, enemy):
        angle_to_enemy = self.angle_to_enemy(enemy)
        return -self.view_angle / 2 <= angle_to_enemy <= self.view_angle / 2 and self.distance_to(enemy) <= self.view_distance

    def distance_to(self, enemy):
        dx = enemy.rect.centerx - self.rect.centerx
        dy = enemy.rect.centery - self.rect.centery
        return math.sqrt(dx ** 2 + dy ** 2)

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time > 500:  # Cooldown period
            bullet_speed = 20
            angle_rad = math.radians(self.angle)
            bullet_dx = bullet_speed * math.cos(angle_rad)
            bullet_dy = bullet_speed * math.sin(angle_rad)
            bullet = Bullet(self.rect.centerx, self.rect.centery, bullet_dx, bullet_dy, BULLET_COLOR)
            self.bullets.append(bullet)
            self.last_shot_time = current_time

    def update_bullets(self, walls, enemies):
        for bullet in self.bullets[:]:
            bullet.move()
            if bullet.is_off_screen() or bullet.collides_with_walls(walls):
                self.bullets.remove(bullet)
            else:
                for enemy in enemies[:]:
                    if bullet.rect.colliderect(enemy.rect):
                        enemies.remove(enemy)
                        self.bullets.remove(bullet)
                        global score
                        score += 1
                        if self.is_ai:
                            self.reward += 10  # Reward for killing an enemy
                            self.last_kill_time = pygame.time.get_ticks()
                        break

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

    def draw_view(self, surface, walls):
        # Calculate the points of the view cone
        start_angle = self.angle - self.view_angle / 2
        end_angle = self.angle + self.view_angle / 2

        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)  # Surface with alpha channel

        for i in range(int(start_angle), int(end_angle) + 1, 2):
            angle_rad = math.radians(i)
            for distance in range(0, self.view_distance, 5):
                x = self.rect.centerx + distance * math.cos(angle_rad)
                y = self.rect.centery + distance * math.sin(angle_rad)
                if not (0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT):
                    break
                if any(wall.collidepoint(x, y) for wall in walls):
                    break
                s.set_at((int(x), int(y)), self.color + (50,))

        surface.blit(s, (0, 0))

    def draw_bullets(self, surface):
        for bullet in self.bullets:
            bullet.draw(surface)

# Bullet class
class Bullet:
    def __init__(self, x, y, dx, dy, color):
        self.rect = pygame.Rect(x, y, 5, 5)
        self.dx = dx
        self.dy = dy
        self.color = color

    def move(self):
        self.rect.x += round(self.dx)
        self.rect.y += round(self.dy)

    def is_off_screen(self):
        return not (0 <= self.rect.x <= SCREEN_WIDTH and 0 <= self.rect.y <= SCREEN_HEIGHT)

    def collides_with_walls(self, walls):
        for wall in walls:
            if self.rect.colliderect(wall):
                return True
        return False

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

# Enemy class
class Enemy:
    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.color = color
        self.angle = random.randint(0, 360)
        self.speed = 2
        self.bullets = []

    def move(self, walls):
        direction = random.choice(['left', 'right', 'up', 'down'])
        dx, dy = 0, 0

        if direction == 'left':
            dx = -self.speed
        elif direction == 'right':
            dx = self.speed
        elif direction == 'up':
            dy = -self.speed
        elif direction == 'down':
            dy = self.speed

        # Horizontal movement with collisions
        self.rect.x += dx
        for wall in walls:
            if self.rect.colliderect(wall):
                if dx > 0:
                    self.rect.right = wall.left
                if dx < 0:
                    self.rect.left = wall.right

        # Vertical movement with collisions
        self.rect.y += dy
        for wall in walls:
            if self.rect.colliderect(wall):
                if dy > 0:
                    self.rect.bottom = wall.top
                if dy < 0:
                    self.rect.top = wall.bottom

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

# Walls definition (house with rooms)
walls = []

# Outer walls of the house
walls.append(pygame.Rect(50, 50, 700, 10))  # Top wall
walls.append(pygame.Rect(50, 540, 700, 10))  # Bottom wall
walls.append(pygame.Rect(50, 50, 10, 500))  # Left wall
walls.append(pygame.Rect(740, 50, 10, 500))  # Right wall

# Inner walls forming rooms
# Vertical walls
walls.append(pygame.Rect(250, 50, 10, 200))   # Left vertical wall
walls.append(pygame.Rect(550, 350, 10, 200))  # Right vertical wall

# Horizontal walls
walls.append(pygame.Rect(50, 250, 250, 10))   # Top horizontal wall
walls.append(pygame.Rect(300, 350, 250, 10))  # Middle horizontal wall

# Doors (openings in walls)
doors = []

# Door in the left vertical wall
doors.append(pygame.Rect(250, 140, 10, 50))  # Door opening

# Remove wall sections where doors are present
for door in doors:
    for wall in walls:
        if wall.colliderect(door):
            # Split the wall into two parts, excluding the door
            walls.remove(wall)
            if wall.width > wall.height:
                # Horizontal wall
                walls.append(pygame.Rect(wall.left, wall.top, door.left - wall.left, wall.height))
                walls.append(pygame.Rect(door.right, wall.top, wall.right - door.right, wall.height))
            else:
                # Vertical wall
                walls.append(pygame.Rect(wall.left, wall.top, wall.width, door.top - wall.top))
                walls.append(pygame.Rect(wall.left, door.bottom, wall.width, wall.bottom - door.bottom))
            break

# Player controls
player_controls = {
    'left': pygame.K_a,
    'right': pygame.K_d,
    'up': pygame.K_w,
    'down': pygame.K_s,
    'rotate_left': pygame.K_q,
    'rotate_right': pygame.K_e,
    'shoot': pygame.K_SPACE
}

# Create player (AI player)
player = Player(100, 100, PLAYER_COLOR, player_controls, is_ai=True)

# Create enemies in random positions
enemies = [
    Enemy(random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT - 100), ENEMY_COLOR),
    Enemy(random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT - 100), ENEMY_COLOR),
    Enemy(random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT - 100), ENEMY_COLOR)
]

# Score
score = 0
font = pygame.font.Font(None, 36)

# Main game loop
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Save the model before exiting
            with open('ai_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == player_controls['shoot']:
                keys = pygame.key.get_pressed()
                if not (keys[player_controls['left']] or keys[player_controls['right']] or keys[player_controls['up']] or keys[player_controls['down']]):
                    player.shoot()

    # Update game state
    current_time = pygame.time.get_ticks()
    if current_time - player.last_kill_time > 10000:  # 10-second timeout
        if current_time - player.last_kill_time > 10000 and player.reward >= 0:
            player.reward -= 20  # One-time penalty for not killing an enemy in time
            # Reset player position
            player.rect.x, player.rect.y = 100, 100
            player.angle = 0
            player.reward = 0
            player.bullets.clear()
            player.last_shot_time = 0
            player.previous_state = None
            player.previous_reward = 0

    player.move(walls, enemies)
    player.update_bullets(walls, enemies)

    # If all enemies are defeated, give a large reward and spawn new enemies
    if not enemies:
        player.reward += 50  # Large reward for clearing all enemies
        enemies = [
            Enemy(random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT - 100), ENEMY_COLOR),
            Enemy(random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT - 100), ENEMY_COLOR),
            Enemy(random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT - 100), ENEMY_COLOR)
        ]

    for enemy in enemies:
        enemy.move(walls)

    # Draw
    screen.fill(FLOOR_COLOR)  # Floor color

    # Draw walls
    for wall in walls:
        pygame.draw.rect(screen, WALL_COLOR, wall)

    # Draw doors (optional visualization)
    for door in doors:
        pygame.draw.rect(screen, FLOOR_COLOR, door)

    # Draw player and bullets
    player.draw_view(screen, walls)
    player.draw(screen)
    player.draw_bullets(screen)

    # Draw enemies
    for enemy in enemies:
        enemy.draw(screen)

    # Draw score
    score_text = font.render(f"Score: {score}  Reward: {player.reward}", True, WHITE)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(30)