# Snake AI Project

This project implements a deep learning model to play the classic Snake game using a Deep Q-Network (DQN). The model learns to play the game by interacting with the environment and optimizing its actions based on rewards received.

## Project Structure

```
snake-ai
├── data
│   └── training_records.pkl  # Contains training records for the DQN model
├── models
│   ├── __init__.py           # Marks the models directory as a package
│   └── dqn_model.py          # Implementation of the DQN model
├── src
│   ├── __init__.py           # Marks the src directory as a package
│   ├── agent.py              # Implementation of the agent interacting with the environment
│   ├── environment.py        # Defines the Snake game environment
│   ├── game.py               # Main game loop and rendering logic
│   └── utils.py              # Utility functions for various tasks
├── train.py                  # Entry point for training the DQN model
├── play.py                   # Entry point for playing the Snake game with the trained model
├── requirements.txt          # Lists project dependencies
├── .gitignore                # Specifies files to ignore by Git
└── README.md                 # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd snake-ai
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Prepare the training data if necessary and ensure `training_records.pkl` is available in the `data` directory.

## Usage

### Training the Model

To train the DQN model, run the following command:
```sh
python train.py
```
This will initialize the environment and agent, and start the training loop. The trained model will be saved after training.

### Playing the Game

To play the Snake game using the trained model, run:
```sh
python play.py --model models/final_model.keras
```
This will load the trained model and allow the agent to play the game in real-time.

#### Controls
- **ESC**: Exit the game
- **SPACE**: Reset the game
- **UP ARROW**: Increase speed
- **DOWN ARROW**: Decrease speed

## Features

- **Deep Q-Learning**: Uses a neural network to approximate the Q-value function.
- **Real-time Visualization**: Displays the neural network's activations and the agent's decision-making process.
- **Customizable Training**: Allows for continued training with adjustable parameters.
- **Performance Metrics**: Tracks and displays the agent's score and rewards during gameplay.
- **Enhanced Graphics**: Includes smooth animations, pulsating food effects, and a trail effect for the snake.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.
