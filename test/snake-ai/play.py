import numpy as np
import pygame
import time
import os
import argparse
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from src.snake_env import SnakeEnvironment
from src.agent import Agent

# Define colors with more appealing palette
BLACK = (20, 20, 20)
WHITE = (255, 255, 255)
DARK_GREEN = (0, 100, 0)
GREEN = (0, 200, 50)
LIGHT_GREEN = (150, 255, 150)
RED = (220, 30, 30)
BLUE = (30, 120, 255)
YELLOW = (255, 255, 0)
GRID_COLOR = (40, 40, 40)
APPLE_RED = (255, 40, 40)
APPLE_HIGHLIGHT = (255, 180, 180)

class GamePlayer:
    def __init__(self, model_path, cell_size=40, delay=100):
        # Initialize the environment
        self.env = SnakeEnvironment()
        self.cell_size = cell_size
        self.delay = delay  # Default delay between frames in ms
        
        # Initialize the agent
        self.agent = Agent(state_size=self.env.state_size, action_size=self.env.action_size)
        
        # Load the model
        if not model_path.endswith('.keras'):
            model_path += '.keras'
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            available_models = [f for f in os.listdir('models') if f.endswith('.keras')]
            if available_models:
                print("Available models:")
                for model in available_models:
                    print(f"  - models/{model}")
            else:
                print("No models found in the 'models' directory")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.agent.load(model_path)
        
        # Visual enhancements
        self.snake_radius = 6  # Rounded corners radius
        self.food_pulse = 0  # For pulsating food effect
        self.pulse_speed = 0.12  # Speed of the pulse
        self.padding = cell_size // 2  # Add padding around the board
        
        # Add trail effect variables
        self.trail = []  # To hold previous positions for trail effect
        self.max_trail_length = 5  # How many trail segments to display
        
        # Animation state
        self.animation_step = 0  # Current animation step
        self.animation_steps = 2  # Reduced animation steps for faster movement
        self.prev_head_pos = None
        self.current_head_pos = None
        self.target_head_pos = None
        
        # Set agent to evaluation mode (no exploration)
        self.agent.epsilon = 0
        
        # Calculate dimensions for neural network panel and game grid
        self.grid_width = self.env.width * self.cell_size
        self.grid_height = self.env.height * self.cell_size
        self.neural_panel_width = 220  # Width of the neural network visualization panel
        
        # Initialize the display with extra width for neural network
        self.screen_width = self.grid_width + 2 * self.padding + self.neural_panel_width
        self.screen_height = self.grid_height + 2 * self.padding
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake AI Visualization")
        
        # Try to load a nicer font, fall back to default if not available
        try:
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
            self.tiny_font = pygame.font.Font(None, 18)  # For smaller labels
        except:
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
            self.tiny_font = pygame.font.Font(None, 18)
        
        # Reset the environment and initialize animation state
        self.state = self.env.reset()
        self.total_reward = 0
        self.done = False
        self.prev_head_pos = self.env.snake_pos.copy()
        self.current_head_pos = self.env.snake_pos.copy()
        self.target_head_pos = self.env.snake_pos.copy()

    def get_activations(self, input_data):
        """Get activations of all layers in the network for visualization"""
        layer_outputs = []
        
        # Using a more compatible approach that works with newer Keras versions
        state_tensor = tf.convert_to_tensor(input_data.reshape(1, -1), dtype=tf.float32)
        
        # Get dense layers only
        dense_layers = [layer for layer in self.agent.model.model.layers if 'dense' in layer.name.lower()]
        
        # Create a new model that outputs all layer activations for dense layers only
        layer_outputs_model = tf.keras.Model(
            inputs=self.agent.model.model.input,
            outputs=[layer.output for layer in dense_layers]
        )
        
        # Get all layer outputs at once
        all_layer_outputs = layer_outputs_model(state_tensor, training=False)
        
        # If only one layer, wrap it in a list
        if not isinstance(all_layer_outputs, list):
            all_layer_outputs = [all_layer_outputs]
        
        # Format the outputs with layer names
        for i, (layer, layer_output) in enumerate(zip(dense_layers, all_layer_outputs)):
            layer_name = f"Layer {i}: {layer.name}"
            layer_outputs.append((layer_name, layer_output.numpy()))
        
        return layer_outputs

    def render_network_activations(self):
        """Render neural network activations on the left side panel"""
        # Get the model's prediction for the current state
        state_tensor = np.expand_dims(self.state, axis=0)
        q_values = self.agent.model.model.predict(state_tensor, verbose=0)[0]
        
        # Define panel_start_x here so it's available in the except block too
        panel_start_x = 10
        
        try:
            # Get layer activations
            layer_outputs = self.get_activations(self.state)
            
            # Calculate panel position
            panel_start_y = self.padding + 10
            
            # Draw panel background
            panel_rect = pygame.Rect(
                0, 0, 
                self.neural_panel_width, 
                self.screen_height
            )
            pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
            
            # Draw dividing line
            pygame.draw.line(
                self.screen,
                (80, 80, 80),
                (self.neural_panel_width, 0),
                (self.neural_panel_width, self.screen_height),
                2
            )
            
            # Draw title
            title = self.small_font.render("Neural Network", True, WHITE)
            self.screen.blit(title, (panel_start_x, panel_start_y))
            
            # Draw each layer of the network
            y_pos = panel_start_y + 30
            
            for layer_index, (layer_name, activations) in enumerate(layer_outputs):
                # Draw layer name
                layer_name_short = f"Layer {layer_index+1}"
                label = self.small_font.render(layer_name_short, True, WHITE)
                self.screen.blit(label, (panel_start_x, y_pos))
                
                # Extract activation values
                if len(activations.shape) > 2:
                    act_values = activations[0].flatten()
                else:
                    act_values = activations[0]
                
                # Draw neurons as a grid of dots
                neuron_count = len(act_values)
                max_width = self.neural_panel_width - panel_start_x * 2
                
                # Calculate grid dimensions
                cols = min(10, max(5, int(np.sqrt(neuron_count))))
                rows = (neuron_count + cols - 1) // cols
                
                neuron_size = min(6, max(3, int(max_width / cols) - 2))
                neuron_spacing = neuron_size + 2
                
                # Draw neurons
                neuron_y = y_pos + 25
                for i, value in enumerate(act_values):
                    if i >= cols * rows:  # Don't go beyond visible area
                        break
                        
                    row = i // cols
                    col = i % cols
                    
                    # Position
                    neuron_x = panel_start_x + col * neuron_spacing
                    
                    # Skip if we'd go off the panel
                    if neuron_x + neuron_size > self.neural_panel_width - 5:
                        continue
                        
                    # Normalize value for color
                    normalized = max(0, min(1, (float(value) + 1) / 2))
                    
                    # Determine color
                    if float(value) >= 0:
                        color = (int(100 * (1-normalized)), int(255 * normalized), 100)
                    else:
                        color = (int(255 * (1-normalized)), int(100 * normalized), 100)
                    
                    # Draw neuron
                    pygame.draw.rect(
                        self.screen,
                        color,
                        [
                            neuron_x, 
                            neuron_y + row * neuron_spacing, 
                            neuron_size, 
                            neuron_size
                        ],
                        border_radius=1
                    )
                
                # Advance position for next layer
                y_pos = neuron_y + rows * neuron_spacing + 15
            
            # Draw Q-values (action outputs)
            q_value_y = self.screen_height - 140
            
            # Draw title
            q_title = self.small_font.render("Action Values", True, WHITE)
            self.screen.blit(q_title, (panel_start_x, q_value_y))
            
            # Get the chosen action
            chosen_action = np.argmax(q_values)
            
            # Draw each action's Q-value
            actions = ["Up", "Right", "Down", "Left"]
            for i, (action, q_value) in enumerate(zip(actions, q_values)):
                color = YELLOW if i == chosen_action else WHITE
                q_text = self.small_font.render(f"{action}: {q_value:.2f}", True, color)
                self.screen.blit(q_text, (panel_start_x, q_value_y + 25 + i*20))
                
            # Draw score and reward at the bottom of the panel
            score_y = self.screen_height - 60
            score_text = self.small_font.render(f"Score: {self.env.score}", True, WHITE)
            self.screen.blit(score_text, (panel_start_x, score_y))
            
            reward_text = self.small_font.render(f"Reward: {self.total_reward:.1f}", True, WHITE)
            self.screen.blit(reward_text, (panel_start_x, score_y + 25))
        
        except Exception as e:
            # Fallback if visualization fails
            error_y = self.padding + 10
            error_text = self.small_font.render("Visualization error:", True, RED)
            self.screen.blit(error_text, (panel_start_x, error_y))
            
            # Show error message
            error_detail = self.tiny_font.render(str(e)[:25], True, RED)
            self.screen.blit(error_detail, (panel_start_x, error_y + 25))
            
            # Still show Q-values
            q_value_y = error_y + 60
            q_title = self.small_font.render("Action Values", True, WHITE)
            self.screen.blit(q_title, (panel_start_x, q_value_y))
            
            chosen_action = np.argmax(q_values)
            actions = ["Up", "Right", "Down", "Left"]
            for i, (action, q_value) in enumerate(zip(actions, q_values)):
                color = YELLOW if i == chosen_action else WHITE
                q_text = self.small_font.render(f"{action}: {q_value:.2f}", True, color)
                self.screen.blit(q_text, (panel_start_x, q_value_y + 25 + i*20))

    def render_simplified_network(self):
        """Render a simplified visualization when the detailed one fails"""
        # Get the model's prediction for the current state
        state_tensor = np.expand_dims(self.state, axis=0)
        q_values = self.agent.model.model.predict(state_tensor, verbose=0)[0]
        
        panel_start_x = 10
        panel_start_y = self.padding + 10
        
        # Draw panel background
        panel_rect = pygame.Rect(0, 0, self.neural_panel_width, self.screen_height)
        pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
        
        # Draw dividing line
        pygame.draw.line(
            self.screen,
            (80, 80, 80),
            (self.neural_panel_width, 0),
            (self.neural_panel_width, self.screen_height),
            2
        )
        
        # Draw title
        title = self.small_font.render("AI Brain (Simplified)", True, WHITE)
        self.screen.blit(title, (panel_start_x, panel_start_y))
        
        # Just draw a simple representation of the network architecture
        # Get model layers
        layers = [layer for layer in self.agent.model.model.layers if hasattr(layer, 'units')]
        
        if not layers:
            # If we can't detect layers with 'units', just use all layers
            layers = self.agent.model.model.layers
        
        # Draw layer boxes
        layer_height = 50
        max_layer_width = self.neural_panel_width - 40
        y_pos = panel_start_y + 40
        
        for i, layer in enumerate(layers):
            # Try to get units if possible, otherwise estimate
            try:
                units = layer.units
            except:
                units = 10  # Default guess
            
            # Scale width based on neurons
            layer_width = min(max_layer_width, max(60, int(units * 3)))
            
            # Center the layer box
            layer_x = panel_start_x + (max_layer_width - layer_width) // 2
            
            # Draw layer box
            layer_color = (70, 130, 180)  # Steel blue
            pygame.draw.rect(
                self.screen,
                layer_color,
                [layer_x, y_pos, layer_width, layer_height],
                border_radius=5
            )
            
            # Draw layer name
            try:
                layer_name = f"Layer {i+1}: {layer.name}" 
            except:
                layer_name = f"Layer {i+1}"
                
            label = self.small_font.render(layer_name, True, WHITE)
            self.screen.blit(label, (layer_x + 10, y_pos + layer_height//2 - 10))
            
            y_pos += layer_height + 20
        
        # Draw Q-values (action outputs)
        q_value_y = self.screen_height - 140
        
        # Draw title
        q_title = self.small_font.render("Action Values", True, WHITE)
        self.screen.blit(q_title, (panel_start_x, q_value_y))
        
        # Get the chosen action
        chosen_action = np.argmax(q_values)
        
        # Draw each action's Q-value
        actions = ["Up", "Right", "Down", "Left"]
        for i, (action, q_value) in enumerate(zip(actions, q_values)):
            color = YELLOW if i == chosen_action else WHITE
            q_text = self.small_font.render(f"{action}: {q_value:.2f}", True, color)
            self.screen.blit(q_text, (panel_start_x, q_value_y + 25 + i*20))
            
        # Draw score and reward at the bottom of the panel
        score_y = self.screen_height - 60
        score_text = self.small_font.render(f"Score: {self.env.score}", True, WHITE)
        self.screen.blit(score_text, (panel_start_x, score_y))
        
        reward_text = self.small_font.render(f"Reward: {self.total_reward:.1f}", True, WHITE)
        self.screen.blit(reward_text, (panel_start_x, score_y + 25))

    def play_step(self):
        """Play a single step of the game with smooth animation"""
        # Only take a new step if the animation is complete
        if self.animation_step >= self.animation_steps:
            if self.done:
                print(f"Game over! Total reward: {self.total_reward}")
                self.state = self.env.reset()
                self.total_reward = 0
                self.done = False
                self.trail = []
                time.sleep(1)
                
                # Reset animation state after game reset
                self.prev_head_pos = self.env.snake_pos.copy()
                self.current_head_pos = self.env.snake_pos.copy()
                self.target_head_pos = self.env.snake_pos.copy()
                self.animation_step = self.animation_steps
                return
            
            # Store previous head position for animation
            self.prev_head_pos = self.env.snake_pos.copy()
            
            # Agent makes decision
            action = self.agent.act(self.state)
            next_state, reward, done = self.env.step(action)
            
            # Add to trail for visual effect
            if len(self.trail) >= self.max_trail_length:
                self.trail.pop(0)
            self.trail.append(self.prev_head_pos.copy())
            
            # Update target position for animation
            self.target_head_pos = self.env.snake_pos.copy()
            
            # Reset animation counter
            self.animation_step = 0
            
            self.state = next_state
            self.total_reward += reward
            self.done = done
        else:
            # Increment animation step
            self.animation_step += 1
            
            # Update current head position for smooth animation
            progress = self.animation_step / self.animation_steps
            self.current_head_pos = [
                self.prev_head_pos[0] + (self.target_head_pos[0] - self.prev_head_pos[0]) * progress,
                self.prev_head_pos[1] + (self.target_head_pos[1] - self.prev_head_pos[1]) * progress
            ]

    def render(self):
        """Render the current state of the game with enhanced graphics"""
        self.screen.fill(BLACK)
        
        # Try to draw detailed neural network, fall back to simplified if it fails
        try:
            self.render_network_activations()
        except Exception as e:
            print(f"Detailed visualization failed: {e}")
            self.render_simplified_network()
        
        # Calculate the offset for the game grid
        grid_offset_x = self.neural_panel_width
        
        # Draw grid with checkerboard pattern
        for x in range(self.env.width):
            for y in range(self.env.height):
                if (x + y) % 2 == 0:
                    pygame.draw.rect(
                        self.screen,
                        GRID_COLOR,
                        [
                            x * self.cell_size + self.padding + grid_offset_x,
                            y * self.cell_size + self.padding,
                            self.cell_size,
                            self.cell_size
                        ]
                    )
        
        # Draw trail effect (fading segments behind the snake)
        for i, trail_pos in enumerate(self.trail):
            # Adjust alpha/opacity based on position in the trail
            alpha = int(255 * (i + 1) / len(self.trail) * 0.3)
            trail_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            trail_surface.fill((GREEN[0], GREEN[1], GREEN[2], alpha))
            self.screen.blit(
                trail_surface,
                (
                    trail_pos[0] * self.cell_size + self.padding + grid_offset_x,
                    trail_pos[1] * self.cell_size + self.padding
                )
            )
        
        # Draw snake body with gradient effect
        for i, segment in enumerate(self.env.snake_body[1:]):  # Skip head
            # Calculate gradient color - darker as we move toward the tail
            segment_factor = max(0.5, 1 - i / len(self.env.snake_body))
            r = int(GREEN[0] * segment_factor)
            g = int(GREEN[1] * segment_factor)
            b = int(GREEN[2] * segment_factor)
            
            # Draw rounded segment
            pygame.draw.rect(
                self.screen,
                (r, g, b),
                [
                    segment[0] * self.cell_size + self.padding + grid_offset_x + 1,
                    segment[1] * self.cell_size + self.padding + 1,
                    self.cell_size - 2,
                    self.cell_size - 2
                ],
                border_radius=self.snake_radius
            )
            
            # Add highlight on top edge for 3D effect
            pygame.draw.rect(
                self.screen,
                LIGHT_GREEN,
                [
                    segment[0] * self.cell_size + self.padding + grid_offset_x + 3,
                    segment[1] * self.cell_size + self.padding + 3,
                    self.cell_size - 6,
                    3
                ],
                border_radius=2
            )
        
        # Draw snake head with animated position
        head_x = self.current_head_pos[0] * self.cell_size + self.padding + grid_offset_x
        head_y = self.current_head_pos[1] * self.cell_size + self.padding
        
        # Draw head
        pygame.draw.rect(
            self.screen,
            BLUE,
            [
                head_x + 1,
                head_y + 1,
                self.cell_size - 2,
                self.cell_size - 2
            ],
            border_radius=self.snake_radius
        )
        
        # Add eyes to snake head
        eye_size = self.cell_size // 5
        eye_offset = self.cell_size // 3
        
        # Draw eyes based on direction
        if self.env.direction == 'RIGHT':
            left_eye = (head_x + self.cell_size - eye_offset, head_y + eye_offset)
            right_eye = (head_x + self.cell_size - eye_offset, head_y + self.cell_size - eye_offset)
        elif self.env.direction == 'LEFT':
            left_eye = (head_x + eye_offset - eye_size//2, head_y + eye_offset)
            right_eye = (head_x + eye_offset - eye_size//2, head_y + self.cell_size - eye_offset)
        elif self.env.direction == 'UP':
            left_eye = (head_x + eye_offset, head_y + eye_offset - eye_size//2)
            right_eye = (head_x + self.cell_size - eye_offset, head_y + eye_offset - eye_size//2)
        else:  # DOWN
            left_eye = (head_x + eye_offset, head_y + self.cell_size - eye_offset)
            right_eye = (head_x + self.cell_size - eye_offset, head_y + self.cell_size - eye_offset)
        
        pygame.draw.circle(self.screen, WHITE, left_eye, eye_size)
        pygame.draw.circle(self.screen, WHITE, right_eye, eye_size)
        
        pygame.draw.circle(self.screen, BLACK, left_eye, eye_size//2)
        pygame.draw.circle(self.screen, BLACK, right_eye, eye_size//2)
        
        # Draw food with pulsating effect
        food_x = self.env.food_pos[0] * self.cell_size + self.padding + grid_offset_x
        food_y = self.env.food_pos[1] * self.cell_size + self.padding
        
        # Calculate pulsation using sine wave
        self.food_pulse += self.pulse_speed
        if self.food_pulse > 2 * math.pi:
            self.food_pulse = 0
            
        pulse_size = 0.2 * math.sin(self.food_pulse) + 0.9  # Oscillate between 0.7 and 1.1
        
        # Draw apple-like food
        apple_center = (
            food_x + self.cell_size // 2,
            food_y + self.cell_size // 2
        )
        apple_radius = int(self.cell_size // 2 * pulse_size)
        
        # Draw apple body
        pygame.draw.circle(self.screen, APPLE_RED, apple_center, apple_radius)
        
        # Draw highlight on apple for 3D effect
        highlight_pos = (
            apple_center[0] - apple_radius // 3,
            apple_center[1] - apple_radius // 3
        )
        pygame.draw.circle(self.screen, APPLE_HIGHLIGHT, highlight_pos, apple_radius // 3)
        
        # Draw stem
        stem_height = apple_radius // 2
        stem_width = apple_radius // 5
        pygame.draw.rect(
            self.screen,
            DARK_GREEN,
            [
                apple_center[0] - stem_width // 2,
                apple_center[1] - apple_radius - stem_height,
                stem_width,
                stem_height
            ]
        )
        
        # Draw leaf
        leaf_points = [
            (apple_center[0], apple_center[1] - apple_radius - stem_height // 2),
            (apple_center[0] + apple_radius // 2, apple_center[1] - apple_radius - stem_height),
            (apple_center[0] + apple_radius // 4, apple_center[1] - apple_radius - stem_height - apple_radius // 4)
        ]
        pygame.draw.polygon(self.screen, GREEN, leaf_points)
        
        # Draw score with shadow effect
        score_text = self.font.render(f"Score: {self.env.score}", True, WHITE)
        score_shadow = self.font.render(f"Score: {self.env.score}", True, (50, 50, 50))
        
        # Draw shadow slightly offset
        self.screen.blit(score_shadow, (grid_offset_x + 12, 12))
        self.screen.blit(score_text, (grid_offset_x + 10, 10))
        
        # Draw reward
        reward_text = self.small_font.render(f"Total Reward: {self.total_reward:.1f}", True, WHITE)
        self.screen.blit(reward_text, (grid_offset_x + 10, 50))
        
        # Draw game over message
        if self.done:
            # Semi-transparent overlay
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))  # Black with 60% opacity
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font.render("Game Over", True, RED)
            restart_text = self.small_font.render("Press SPACE to restart or ESC to exit", True, WHITE)
            
            # Center the text on the game area only (not including neural panel)
            game_area_center_x = grid_offset_x + (self.screen_width - grid_offset_x) // 2
            
            game_over_rect = game_over_text.get_rect(center=(game_area_center_x, self.screen_height // 2 - 20))
            restart_rect = restart_text.get_rect(center=(game_area_center_x, self.screen_height // 2 + 20))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def play(self):
        """Main game loop with speed control"""
        clock = pygame.time.Clock()
        running = True
        speed_multiplier = 1.0  # Default speed
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Reset the game
                        self.state = self.env.reset()
                        self.total_reward = 0
                        self.done = False
                        self.trail = []
                        
                        # Reset animation state
                        self.prev_head_pos = self.env.snake_pos.copy()
                        self.current_head_pos = self.env.snake_pos.copy()
                        self.target_head_pos = self.env.snake_pos.copy()
                        self.animation_step = self.animation_steps
                    # Speed controls
                    elif event.key == pygame.K_UP:
                        speed_multiplier = min(1000, speed_multiplier * 1.5)  # Speed up
                        print(f"Speed: {speed_multiplier}x")
                    elif event.key == pygame.K_DOWN:
                        speed_multiplier = max(0.25, speed_multiplier / 1.5)  # Slow down
                        print(f"Speed: {speed_multiplier}x")
            
            # Play and render a step
            self.play_step()
            self.render()
            
            # Control game speed with adjustable multiplier
            # Lower delay = faster game
            effective_delay = max(5, int(self.delay / speed_multiplier))
            fps = (131 / effective_delay) * self.animation_steps
            clock.tick(fps)

def get_available_models():
    """Get list of available models"""
    if not os.path.exists('models'):
        print("No 'models' directory found")
        return []
    
    models = [f for f in os.listdir('models') if f.endswith('.keras')]
    if not models:
        models = [f for f in os.listdir('models')]  # Try without extension filter
    return models

if __name__ == "__main__":
    pygame.init()
    
    parser = argparse.ArgumentParser(description="Play Snake with a trained AI")
    parser.add_argument("--model", help="Path to the model file")
    parser.add_argument("--cell_size", type=int, default=40, help="Size of each cell in pixels")
    parser.add_argument("--delay", type=int, default=50, help="Delay between moves in ms (lower = faster)")
    parser.add_argument("--animation", type=int, default=3, help="Animation steps (lower = faster)")
    
    args = parser.parse_args()
    
    # If no model specified, list available models
    if not args.model:
        models = get_available_models()
        if models:
            print("Available models:")
            for i, model in enumerate(models):
                print(f"  {i+1}. models/{model}")
            
            try:
                selection = int(input("Select a model number (or press Enter for the first one): ") or "1")
                if 1 <= selection <= len(models):
                    model_path = os.path.join("models", models[selection-1])
                else:
                    print("Invalid selection. Using the first model.")
                    model_path = os.path.join("models", models[0])
            except:
                print("Invalid input. Using the first model.")
                model_path = os.path.join("models", models[0])
        else:
            print("No models found. Please train a model first or specify a valid model path.")
            exit(1)
    else:
        model_path = args.model
    
    try:
        player = GamePlayer(model_path, cell_size=args.cell_size, delay=args.delay)
        player.animation_steps = args.animation  # Set animation smoothness
        player.delay = 30  # Reduced delay for faster game speed
        print(f"Playing Snake AI with model: {model_path}")
        print("Controls:")
        print("  ESC - Exit game")
        print("  SPACE - Reset game")
        print("  UP ARROW - Increase speed")
        print("  DOWN ARROW - Decrease speed")
        player.play()
    except Exception as e:
        print(f"Error: {e}")
    
    pygame.quit()