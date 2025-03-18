import pickle
import numpy as np
import os
import time
import tensorflow as tf
from src.agent import Agent
from src.snake_env import SnakeEnvironment

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

def continue_training_model(model_path="models/pasmal_ent_best_avg.keras", episodes=1000):
    """
    Continue training an existing model with the new reward system
    """
    print(f"Continuing training for model: {model_path}")
    start_time = time.time()
    
    # Check if model exists
    if not os.path.exists(model_path):
        if not model_path.endswith('.keras'):
            model_path += '.keras'
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return
        else:
            print(f"Error: Model file not found at {model_path}")
            return
    
    # Initialize the environment and agent
    env = SnakeEnvironment(width=10, height=10)  # Smaller grid for faster training
    agent = Agent(state_size=env.state_size, action_size=env.action_size)
    
    # Load the existing model
    try:
        agent.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set specific epsilon parameters for continued training
    # Adjusted for the new reward system
    agent.epsilon = 0.7        # Start with higher exploration to adapt to the new penalty
    agent.epsilon_min = 0.1    # Keep a slightly higher minimum to ensure ongoing exploration
    agent.epsilon_decay = 0.995 # Moderate decay to allow sufficient exploration

    print(f"Training parameters: epsilon={agent.epsilon}, decay={agent.epsilon_decay}, min={agent.epsilon_min}")

    # Load training records if available
    try:
        with open('data/training_records.pkl', 'rb') as f:
            training_records = pickle.load(f)
            print(f"Loaded {len(training_records)} previous training records")
            # Get the highest episode number to continue from there
            last_episode = max([record[0] for record in training_records]) + 1
    except (FileNotFoundError, ValueError):
        training_records = []
        last_episode = 0
        print("Starting new training records")

    # Initialize tracking variables
    best_reward = float('-inf')
    best_avg_reward = float('-inf')
    rewards_history = []
    
    # Create a filename-friendly timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create a new model name for the continued training (simplify names)
    model_prefix = "pasmal_ent_000000000"
    
    # Memory limit
    max_memory_size = 100000
    
    try:
        for e in range(episodes):
            current_episode = last_episode + e
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
            training_records.append((current_episode, total_reward, env.score))
            rewards_history.append(total_reward)
            if len(rewards_history) > 100:
                rewards_history.pop(0)
            
            # Calculate moving average
            avg_reward = sum(rewards_history) / len(rewards_history)
            
            # Log progress periodically
            if e % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode: {e}/{episodes} (#{current_episode}) | "
                      f"Reward: {total_reward:.1f} | Score: {env.score} | "
                      f"Avg(100): {avg_reward:.1f} | Epsilon: {agent.epsilon:.3f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Save periodically
            if e % 50 == 0 and e > 0:
                # Save checkpoint (simpler name)
                checkpoint_path = f"models/{model_prefix}checkpoint_{e}.keras"
                try:
                    agent.save(checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
                    
                    # Save training records
                    with open('data/training_records.pkl', 'wb') as f:
                        pickle.dump(training_records, f)
                except Exception as ex:
                    print(f"Error saving checkpoint: {ex}")
            
            # Save best models
            if total_reward > best_reward:
                best_reward = total_reward
                try:
                    agent.save(f"models/{model_prefix}best_reward.keras")
                    print(f"New best reward: {best_reward:.1f} - Saved as {model_prefix}best_reward.keras")
                except Exception as ex:
                    print(f"Error saving best reward model: {ex}")
            
            if len(rewards_history) >= 20 and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                try:
                    agent.save(f"models/{model_prefix}best_avg.keras")
                    print(f"New best average reward: {best_avg_reward:.1f} - Saved as {model_prefix}best_avg.keras")
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

        # Save final continued model
        try:
            # Also overwrite the original final_model.keras to keep it updated
            agent.save("models/final_model.keras")
            print(f"Updated original final_model.keras")
            
            # Also save with timestamp for backup
            agent.save(f"models/{model_prefix}{timestamp}.keras")
            print(f"Backup final model saved with timestamp")
            
            # Save final records
            with open('data/training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            print(f"Training records saved to data/training_records.pkl")
        except Exception as ex:
            print(f"Error saving final model: {ex}")
        
        total_time = time.time() - start_time
        print(f"Continued training completed in {total_time:.1f} seconds")
        print(f"Best reward achieved: {best_reward:.1f}")
        print(f"Best average reward: {best_avg_reward:.1f}")
        print(f"Model saved as: models/final_model.keras")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        try:
            agent.save(f"models/{model_prefix}interrupted.keras")
            agent.save("models/final_model.keras")  # Also update the original
            with open('data/training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            print(f"Models and records saved")
        except Exception as ex:
            print(f"Error saving after interrupt: {ex}")

if __name__ == "__main__":
    print("\n===== SNAKE AI TRAINING CONTINUATION =====")
    print("This script will continue training your final model with improved rewards")
    print("Default: Continue training 'models/final_model.keras' for 1000 episodes\n")
    
    # List available models
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
        if model_files:
            print("Available models:")
            for i, model_file in enumerate(model_files):
                print(f"  {i+1}. {model_file}")
            print()  # Empty line
    
    # Get model path with better default and validation
    model_to_continue = input("Enter path to model to continue (default: models/final_model.keras): ").strip()
    if not model_to_continue:
        model_to_continue = "models/final_model.keras"
    
    if not os.path.exists(model_to_continue):
        if not model_to_continue.startswith("models/"):
            model_to_continue = f"models/{model_to_continue}"
        if not model_to_continue.endswith(".keras"):
            model_to_continue = f"{model_to_continue}.keras"
    
    # Get episode count with validation
    try:
        episodes = int(input("How many episodes to train for? (default: 1000): ").strip() or "1000")
        if episodes <= 0:
            episodes = 1000
            print("Invalid value, using 1000 episodes")
    except ValueError:
        episodes = 1000
        print("Invalid input, using 1000 episodes")
    
    print(f"\nStarting continued training of {model_to_continue} for {episodes} episodes...\n")
    continue_training_model(model_to_continue, episodes)