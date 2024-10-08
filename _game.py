import sys
import math
import random
import pickle

import pygame
import numpy as np

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
WALL_COLOR = (50, 50, 50)
FLOOR_COLOR = (200, 200, 200)
PLAYER_COLOR = (0, 0, 255)
BULLET_COLOR = (255, 255, 0)
ENEMY_COLOR = (255, 0, 0)
ENEMY_BULLET_COLOR = (255, 0, 255)
FONT_SIZE = 36
PLAYER_SIZE = 20
BULLET_SIZE = 5
BULLET_SPEED = 20
ENEMY_SIZE = 20
VIEW_DISTANCE = 400
VIEW_ANGLE = 60
AI_MODEL_PATH = './models/game.pkl'
AI_MODEL_SAVE_PATH = './models/game.pkl'
COOLDOWN_PERIOD = 500
KILL_TIMEOUT = 10000
REWARD_STAND_STILL = -0.1
REWARD_ENEMY_IN_VIEW = 0.1
REWARD_KILL_ENEMY = 10
PENALTY_NO_KILL = -20
REWARD_CLEAR_ENEMIES = 50
ENEMY_SPEED = 2
ENEMY_COUNT = 3
FPS = 30
EPSILON = 0.1  # Exploration rate
ALPHA = 0.1    # Learning rate
GAMMA = 0.95   # Discount factor

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("CQB AI Game with Q-Learning")
font = pygame.font.Font(None, FONT_SIZE)
clock = pygame.time.Clock()

def load_or_initialize_q_table():
    try:
        with open(AI_MODEL_PATH, 'rb') as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        q_table = {}
    return q_table

def save_q_table(q_table):
    with open(AI_MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(q_table, f)

class Player:
    def __init__(self, x, y, color, controls, is_ai=False):
        self.rect = pygame.Rect(x, y, PLAYER_SIZE, PLAYER_SIZE)
        self.color = color
        self.angle = 0
        self.speed = 3
        self.rotation_speed = 5
        self.view_distance = VIEW_DISTANCE
        self.view_angle = VIEW_ANGLE
        self.controls = controls
        self.bullets = []
        self.is_ai = is_ai
        self.reward = 0
        self.last_shot_time = 0
        self.last_kill_time = pygame.time.get_ticks()
        self.previous_state = None
        self.previous_action = None
        self.q_table = load_or_initialize_q_table()

    def move(self, walls, enemies):
        if self.is_ai:
            self.ai_move(walls, enemies)
        else:
            self.manual_move(walls)

    def manual_move(self, walls):
        keys = pygame.key.get_pressed()
        dx = (keys[self.controls['right']] - keys[self.controls['left']]) * self.speed
        dy = (keys[self.controls['down']] - keys[self.controls['up']]) * self.speed

        self.rect.x += dx
        self.handle_collisions(dx, 0, walls)
        self.rect.y += dy
        self.handle_collisions(0, dy, walls)

        if keys[self.controls['rotate_left']]:
            self.angle = (self.angle - self.rotation_speed) % 360
        if keys[self.controls['rotate_right']]:
            self.angle = (self.angle + self.rotation_speed) % 360

    def ai_move(self, walls, enemies):
        state = self.get_state(walls, enemies)
        action = self.choose_action(state)
        self.perform_action(action, walls)

        reward = self.calculate_reward(enemies)
        next_state = self.get_state(walls, enemies)
        self.update_q_table(state, action, reward, next_state)

        self.previous_state = state
        self.previous_action = action

    def handle_collisions(self, dx, dy, walls):
        for wall in walls:
            if self.rect.colliderect(wall):
                if dx > 0:
                    self.rect.right = wall.left
                if dx < 0:
                    self.rect.left = wall.right
                if dy > 0:
                    self.rect.bottom = wall.top
                if dy < 0:
                    self.rect.top = wall.bottom

    def get_state(self, walls, enemies):
        closest_enemy = min(enemies, key=lambda enemy: self.distance_to(enemy))
        distance_to_enemy = self.distance_to(closest_enemy)
        angle_to_enemy = self.angle_to_enemy(closest_enemy)
        state = (
            round(self.rect.centerx / SCREEN_WIDTH, 2),
            round(self.rect.centery / SCREEN_HEIGHT, 2),
            round(self.angle / 360, 2),
            round(distance_to_enemy / self.view_distance, 2),
            round(angle_to_enemy / 360, 2),
            len(walls),
            len(enemies)
        )
        return state

    def choose_action(self, state):
        if random.uniform(0, 1) < EPSILON:
            action = random.choice(list(range(len(self.action_space()))))
        else:
            q_values = self.q_table.get(state, [0] * len(self.action_space()))
            max_q = max(q_values)
            actions_with_max_q = [i for i, q in enumerate(q_values) if q == max_q]
            action = random.choice(actions_with_max_q)
        return action

    def perform_action(self, action_index, walls):
        action = self.action_space()[action_index]
        move = action['move']
        rotate = action['rotate']
        shoot = action['shoot']

        self.angle = (self.angle + rotate * self.rotation_speed) % 360
        dx = self.speed * move * math.cos(math.radians(self.angle))
        dy = self.speed * move * math.sin(math.radians(self.angle))

        self.rect.x += int(dx)
        self.handle_collisions(dx, 0, walls)
        self.rect.y += int(dy)
        self.handle_collisions(0, dy, walls)

        if shoot:
            self.shoot()

    def calculate_reward(self, enemies):
        reward = 0
        for enemy in enemies:
            if self.is_enemy_in_view(enemy):
                reward += REWARD_ENEMY_IN_VIEW
        if self.previous_state and self.previous_action is not None:
            if self.previous_action == 0:  # If doing nothing
                reward += REWARD_STAND_STILL
        return reward

    def update_q_table(self, state, action, reward, next_state):
        q_values = self.q_table.get(state, [0] * len(self.action_space()))
        next_q_values = self.q_table.get(next_state, [0] * len(self.action_space()))
        max_next_q = max(next_q_values)

        q_values[action] = q_values[action] + ALPHA * (reward + GAMMA * max_next_q - q_values[action])
        self.q_table[state] = q_values

    def action_space(self):
        return [
            {'move': 0, 'rotate': 0, 'shoot': False},  # Do nothing
            {'move': 1, 'rotate': 0, 'shoot': False},  # Move forward
            {'move': -1, 'rotate': 0, 'shoot': False},  # Move backward
            {'move': 0, 'rotate': -1, 'shoot': False},  # Rotate left
            {'move': 0, 'rotate': 1, 'shoot': False},  # Rotate right
            {'move': 0, 'rotate': 0, 'shoot': True},  # Shoot
            {'move': 1, 'rotate': -1, 'shoot': False},  # Move forward and rotate left
            {'move': 1, 'rotate': 1, 'shoot': False},  # Move forward and rotate right
            {'move': -1, 'rotate': -1, 'shoot': False},  # Move backward and rotate left
            {'move': -1, 'rotate': 1, 'shoot': False},  # Move backward and rotate right
        ]

    def angle_to_enemy(self, enemy):
        dx = enemy.rect.centerx - self.rect.centerx
        dy = enemy.rect.centery - self.rect.centery
        return (math.degrees(math.atan2(dy, dx)) - self.angle) % 360

    def is_enemy_in_view(self, enemy):
        angle_to_enemy = self.angle_to_enemy(enemy)
        in_view_angle = -self.view_angle / 2 <= angle_to_enemy <= self.view_angle / 2
        in_view_distance = self.distance_to(enemy) <= self.view_distance
        return in_view_angle and in_view_distance

    def distance_to(self, enemy):
        dx = enemy.rect.centerx - self.rect.centerx
        dy = enemy.rect.centery - self.rect.centery
        return math.hypot(dx, dy)

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time > COOLDOWN_PERIOD:
            angle_rad = math.radians(self.angle)
            bullet_dx = BULLET_SPEED * math.cos(angle_rad)
            bullet_dy = BULLET_SPEED * math.sin(angle_rad)
            bullet = Bullet(self.rect.centerx, self.rect.centery, bullet_dx, bullet_dy, BULLET_COLOR)
            self.bullets.append(bullet)
            self.last_shot_time = current_time

    def update_bullets(self, walls, enemies, score):
        for bullet in self.bullets[:]:
            bullet.move()
            if bullet.is_off_screen() or bullet.collides_with_walls(walls):
                self.bullets.remove(bullet)
            else:
                for enemy in enemies[:]:
                    if bullet.rect.colliderect(enemy.rect):
                        enemies.remove(enemy)
                        self.bullets.remove(bullet)
                        score += 1
                        if self.is_ai:
                            self.reward += REWARD_KILL_ENEMY
                            self.last_kill_time = pygame.time.get_ticks()
                        break
        return score

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

    def draw_view(self, surface, walls):
        start_angle = self.angle - self.view_angle / 2
        end_angle = self.angle + self.view_angle / 2
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
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

    def collides_with_walls(self, walls):
        return any(self.rect.colliderect(wall) for wall in walls)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

class Enemy:
    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, ENEMY_SIZE, ENEMY_SIZE)
        self.color = color
        self.angle = random.randint(0, 360)
        self.speed = ENEMY_SPEED

    def move(self, walls):
        direction = random.choice(['left', 'right', 'up', 'down'])
        dx = dy = 0
        if direction == 'left':
            dx = -self.speed
        elif direction == 'right':
            dx = self.speed
        elif direction == 'up':
            dy = -self.speed
        elif direction == 'down':
            dy = self.speed
        self.rect.x += dx
        self.handle_collisions(dx, 0, walls)
        self.rect.y += dy
        self.handle_collisions(0, dy, walls)

    def handle_collisions(self, dx, dy, walls):
        for wall in walls:
            if self.rect.colliderect(wall):
                if dx > 0:
                    self.rect.right = wall.left
                if dx < 0:
                    self.rect.left = wall.right
                if dy > 0:
                    self.rect.bottom = wall.top
                if dy < 0:
                    self.rect.top = wall.bottom

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

def create_walls_and_doors():
    walls = [
        pygame.Rect(50, 50, 700, 10),
        pygame.Rect(50, 540, 700, 10),
        pygame.Rect(50, 50, 10, 500),
        pygame.Rect(740, 50, 10, 500),
        pygame.Rect(250, 50, 10, 200),
        pygame.Rect(550, 350, 10, 200),
        pygame.Rect(50, 250, 250, 10),
        pygame.Rect(300, 350, 250, 10)
    ]
    doors = [pygame.Rect(250, 140, 10, 50)]
    for door in doors:
        for wall in walls[:]:
            if wall.colliderect(door):
                walls.remove(wall)
                if wall.width > wall.height:
                    walls.append(pygame.Rect(wall.left, wall.top, door.left - wall.left, wall.height))
                    walls.append(pygame.Rect(door.right, wall.top, wall.right - door.right, wall.height))
                else:
                    walls.append(pygame.Rect(wall.left, wall.top, wall.width, door.top - wall.top))
                    walls.append(pygame.Rect(wall.left, door.bottom, wall.width, wall.bottom - door.bottom))
                break
    return walls, doors

def create_enemies():
    return [Enemy(random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT - 100), ENEMY_COLOR)
            for _ in range(ENEMY_COUNT)]

def main():
    walls, doors = create_walls_and_doors()
    player_controls = {
        'left': pygame.K_a,
        'right': pygame.K_d,
        'up': pygame.K_w,
        'down': pygame.K_s,
        'rotate_left': pygame.K_q,
        'rotate_right': pygame.K_e,
        'shoot': pygame.K_SPACE
    }
    player = Player(100, 100, PLAYER_COLOR, player_controls, is_ai=True)
    enemies = create_enemies()
    score = 0

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        if current_time - player.last_kill_time > KILL_TIMEOUT and player.reward >= 0:
            player.reward += PENALTY_NO_KILL
            player.rect.topleft = (100, 100)
            player.angle = 0
            player.reward = 0
            player.bullets.clear()
            player.last_shot_time = 0
            player.previous_state = None
            player.previous_action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_q_table(player.q_table)
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == player_controls['shoot']:
                    keys = pygame.key.get_pressed()
                    if not any(keys[player_controls[dir]] for dir in ['left', 'right', 'up', 'down']):
                        player.shoot()

        player.move(walls, enemies)
        score = player.update_bullets(walls, enemies, score)

        if not enemies:
            player.reward += REWARD_CLEAR_ENEMIES
            enemies = create_enemies()

        for enemy in enemies:
            enemy.move(walls)

        screen.fill(FLOOR_COLOR)
        for wall in walls:
            pygame.draw.rect(screen, WALL_COLOR, wall)
        for door in doors:
            pygame.draw.rect(screen, FLOOR_COLOR, door)
        player.draw_view(screen, walls)
        player.draw(screen)
        player.draw_bullets(screen)
        for enemy in enemies:
            enemy.draw(screen)
        score_text = font.render(f"Score: {score}  Reward: {player.reward}", True, WHITE)
        screen.blit(score_text, (10, 10))
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
