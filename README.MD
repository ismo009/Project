# Snake AI Project

This project implements an AI agent to play the classic Snake game using reinforcement learning. The AI is trained using a deep Q-learning algorithm and visualizes the neural network's activations in real-time.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Snake AI project aims to create an intelligent agent capable of playing the Snake game autonomously. The agent is trained using deep Q-learning, a reinforcement learning technique, to maximize its score by learning from its actions and rewards.

## Features

- **Deep Q-Learning**: Uses a neural network to approximate the Q-value function.
- **Real-time Visualization**: Displays the neural network's activations and the agent's decision-making process.
- **Customizable Training**: Allows for continued training with adjustable parameters.
- **Performance Metrics**: Tracks and displays the agent's score and rewards during gameplay.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/snake-ai.git
    cd snake-ai
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Playing the Game

To play the game with the trained AI model, run the following command:

```sh
python play.py --model [final_model.keras](http://_vscodecontentref_/1)

Controls
ESC: Exit the game
SPACE: Reset the game
UP ARROW: Increase speed
DOWN ARROW: Decrease speed

Training
Initial Training
To train the AI from scratch, run the following command:
python train.py --episodes 1000

Continuing Training
To continue training an existing model, run the following command:
python train10.py --model [final_model.keras](http://_vscodecontentref_/2) --episodes 1000

Epsilon Parameters
The epsilon parameters for training are set to encourage exploration initially and gradually shift to exploitation:

Initial Epsilon: 0.8
Minimum Epsilon: 0.05
Epsilon Decay: 0.995
Visualization
The game includes a real-time visualization of the neural network's activations and the agent's decision-making process. The visualization panel is displayed on the left side of the game window, showing the activations of each layer and the Q-values for each action.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

License
This project is licensed under the MIT License. See the LICENSE file for details.