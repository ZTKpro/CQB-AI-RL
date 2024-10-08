# CQB AI Game

## Overview

CQB AI Game is a Python-based 2D shooter game developed using the Pygame library. The game features a player, controlled either manually or by an AI, navigating a complex map with walls and rooms while trying to eliminate enemies. The AI is powered by a Multi-Layer Perceptron Regressor, which learns the optimal actions to maximize rewards during gameplay.

## Features
- **AI Controlled Player**: The player can be either manually controlled or guided by a machine learning model.
- **Enemy Interaction**: Randomly moving enemies that the player must eliminate.
- **View Cone Visualization**: The player's field of view is displayed, and the AI rewards are based on enemy visibility.
- **Learning Model**: The AI player learns to maximize rewards through shooting enemies and maintaining line of sight.
- **Score Tracking**: A score is maintained based on successful enemy eliminations.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/cqb-ai-game.git
   cd cqb-ai-game
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   The `requirements.txt` file includes:
   - pygame
   - scikit-learn
   - numpy

## How to Play

- **Manual Controls**:
  - Use `W`, `A`, `S`, `D` to move the player.
  - Use `Q` and `E` to rotate the player left and right.
  - Press `SPACE` to shoot.

- **AI Player**: The AI-controlled player navigates the environment, tracks enemies, and tries to maximize the reward by facing enemies and shooting.

## Game Mechanics

- **Player Movement**: The player can move forward, backward, and rotate to change directions.
- **Shooting Mechanic**: The player can shoot bullets to eliminate enemies.
- **Rewards System**: The AI player gets rewards for keeping enemies in view and eliminating them, while penalties are applied for inactivity.
- **Wall and Room Layout**: The game includes walls, rooms, and doorways, adding complexity to navigation and shooting.
- **Model Training**: The game uses a `MLPRegressor` from scikit-learn. The model is updated with each action taken, based on the reward system.

## Saving and Loading the Model

- The trained model is saved to a file named `ai_model.pkl`.
- If a saved model is found, it is loaded at the start of the game; otherwise, a new model is initialized.

## Running the Game

To start the game, run:
```sh
python main.py
```
This will launch the Pygame window, where you can choose to either control the player manually or let the AI control the player.


