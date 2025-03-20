import pickle
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from src.snake_env import SnakeEnvironment
from src.agent import Agent

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Enable memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU available, enabling acceleration")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU available, using CPU")

def build_enhanced_model(state_size, action_size):
    """Build an enhanced model for late-game performance"""
    model = Sequential()
    # More complex model for handling the enhanced state
    model.add(Dense(256, input_dim=state_size, activation='relu'))
    model.add(Dropout(0.2))  # Add dropout for regularization
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def train_enhanced_model(episodes=1000):
    """Train a new model with enhanced state representation"""
    print(f"Training new model with enhanced state representation for {episodes} episodes")
    start_time = time.time()
    
    # Initialize the environment and agent
    env = SnakeEnvironment(width=10, height=10)  # Smaller grid for faster training
    
    # Create a custom agent with our enhanced model
    agent = Agent(state_size=env.state_size, action_size=env.action_size)
    
    # Replace the model with our enhanced model
    agent.model.model = build_enhanced_model(env.state_size, env.action_size)
    
    # Set training parameters optimized for exploration
    agent.epsilon = 1.0        # Start with full exploration
    agent.epsilon_min = 0.05   # Ensure ongoing exploration
    agent.epsilon_decay = 0.995 # Balanced decay rate

    print(f"Training parameters: epsilon={agent.epsilon}, decay={agent.epsilon_decay}, min={agent.epsilon_min}")
    print(f"State size: {env.state_size} features")  # Should be 22

    # Initialize tracking variables
    training_records = []
    best_reward = float('-inf')
    best_avg_reward = float('-inf')
    rewards_history = []
    
    # Create a filename-friendly timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Model prefix for saved files
    model_prefix = "enhanced_"
    
    # Memory limit
    max_memory_size = 100000
    
    try:
        for e in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 200  # Prevent infinite loops
            
            while not done and steps < max_steps:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                
                # Limit memory size
                if len(agent.memory) > max_memory_size:
                    agent.memory.pop(0)  # Remove oldest experience
                    
                state = next_state
                total_reward += reward
                steps += 1

            # Train the agent on a batch of experiences
            if len(agent.memory) > 64:
                agent.replay(batch_size=64)
            
            # Update records and history
            training_records.append((e, total_reward, env.score))
            rewards_history.append(total_reward)
            if len(rewards_history) > 100:
                rewards_history.pop(0)
            
            # Calculate moving average
            avg_reward = sum(rewards_history) / len(rewards_history)
            
            # Log progress periodically
            if e % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode: {e}/{episodes} | "
                      f"Reward: {total_reward:.1f} | Score: {env.score} | "
                      f"Avg(100): {avg_reward:.1f} | Epsilon: {agent.epsilon:.3f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Save periodically
            if e % 50 == 0 and e > 0:
                # Save checkpoint
                checkpoint_path = f"models/{model_prefix}checkpoint_{e}.keras"
                try:
                    agent.save(checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
                    
                    # Save training records
                    with open('data/enhanced_training_records.pkl', 'wb') as f:
                        pickle.dump(training_records, f)
                except Exception as ex:
                    print(f"Error saving checkpoint: {ex}")
            
            # Save best models
            if total_reward > best_reward:
                best_reward = total_reward
                try:
                    agent.save(f"models/{model_prefix}best_reward.keras")
                    print(f"New best reward: {best_reward:.1f}")
                except Exception as ex:
                    print(f"Error saving best reward model: {ex}")
            
            if len(rewards_history) >= 20 and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                try:
                    agent.save(f"models/{model_prefix}best_avg.keras")
                    print(f"New best average reward: {best_avg_reward:.1f}")
                except Exception as ex:
                    print(f"Error saving best average model: {ex}")
                    
            # Every 100 episodes, do a pure evaluation
            if e % 100 == 0:
                # Temporarily set epsilon to 0 for pure exploitation
                orig_epsilon = agent.epsilon
                agent.epsilon = 0
                
                # Run evaluation episode
                eval_state = env.reset()
                eval_done = False
                eval_reward = 0
                eval_steps = 0
                
                while not eval_done and eval_steps < max_steps:
                    eval_action = agent.act(eval_state)
                    eval_state, r, eval_done = env.step(eval_action)
                    eval_reward += r
                    eval_steps += 1
                
                print(f"Evaluation (ε=0): Score={env.score}, Reward={eval_reward:.1f}")
                
                # Restore epsilon
                agent.epsilon = orig_epsilon

        # Save final model
        try:
            final_path = f"models/{model_prefix}final.keras"
            agent.save(final_path)
            
            # Also save with timestamp for backup
            agent.save(f"models/{model_prefix}{timestamp}.keras")
            
            # Save final records
            with open('data/enhanced_training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            print(f"Training records saved to data/enhanced_training_records.pkl")
        except Exception as ex:
            print(f"Error saving final model: {ex}")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f} seconds")
        print(f"Best reward achieved: {best_reward:.1f}")
        print(f"Best average reward: {best_avg_reward:.1f}")
        print(f"Model saved as: {final_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        try:
            agent.save(f"models/{model_prefix}interrupted.keras")
            with open('data/enhanced_training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            print(f"Models and records saved")
        except Exception as ex:
            print(f"Error saving after interrupt: {ex}")

if __name__ == "__main__":
    print("\n===== SNAKE AI ENHANCED MODEL TRAINING =====")
    print("This script will train a new model with enhanced state representation")
    print("This model focuses on late-game performance with 22 state features\n")
    
    # Get episode count with validation
    try:
        episodes = int(input("How many episodes to train for? (default: 2000): ").strip() or "2000")
        if episodes <= 0:
            episodes = 2000
            print("Invalid value, using 2000 episodes")
    except ValueError:
        episodes = 2000
        print("Invalid input, using 2000 episodes")
    
    print(f"\nStarting enhanced model training for {episodes} episodes...\n")
    train_enhanced_model(episodes)