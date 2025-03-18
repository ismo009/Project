import pickle
import numpy as np
import os
import time
import tensorflow as tf
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

def train():
    print("Starting Snake AI training...")
    start_time = time.time()

    # Initialize the environment and agent
    env = SnakeEnvironment(width=10, height=10)  # Smaller grid for faster training
    agent = Agent(state_size=env.state_size, action_size=env.action_size)

    # Modify agent's epsilon parameters directly after initialization
    agent.epsilon = 1.0                # Start with full exploration
    agent.epsilon_min = 0.05           # Keep minimum exploration higher
    agent.epsilon_decay = 0.995        # Decay exploration more slowly
    print(f"Initial epsilon: {agent.epsilon}, decay: {agent.epsilon_decay}, min: {agent.epsilon_min}")

    # Load training records if available
    try:
        with open('data/training_records.pkl', 'rb') as f:
            training_records = pickle.load(f)
            print(f"Loaded {len(training_records)} previous training records")
    except FileNotFoundError:
        training_records = []
        print("Starting new training records")

    # At the beginning
    full_reward_history = []

    # Training parameters
    episodes = 1000
    best_reward = float('-inf')
    best_avg_reward = float('-inf')
    rewards_history = []

    # Early stopping parameters
    patience = 200  # Episodes to wait before stopping if no improvement
    no_improvement_count = 0

    # At the beginning of your train function:
    max_memory_size = 100000

    try:
        for e in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 200  # Reduce max steps to prevent getting stuck

            while not done and steps < max_steps:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                # When remembering experiences:
                if len(agent.memory) > max_memory_size:
                    agent.memory.pop(0)  # Remove oldest experience
                state = next_state
                total_reward += reward
                steps += 1

            # During training
            full_reward_history.append((e, total_reward, env.score))

            # Train the agent with experiences - use smaller batches for better stability
            if len(agent.memory) > 64:
                agent.replay(batch_size=64)

            # Update records
            training_records.append((e, total_reward))
            rewards_history.append(total_reward)
            if len(rewards_history) > 100:
                rewards_history.pop(0)
            
            # Calculate moving average
            avg_reward = sum(rewards_history) / len(rewards_history)

            # Log progress periodically
            if e % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode: {e}/{episodes} | Reward: {total_reward:.1f} | "
                      f"Avg(100): {avg_reward:.1f} | Epsilon: {agent.epsilon:.3f} | "
                      f"Time: {elapsed:.1f}s | Score: {env.score}")
            
            # Save checkpoint models
            if e % 50 == 0 and e > 0:  # Save more frequently
                checkpoint_path = f"models/checkpoint_episode_{e}.keras"
                try:
                    agent.save(checkpoint_path)
                    print(f"Checkpoint model saved: {checkpoint_path}")
                    # Also save training records periodically
                    with open('data/training_records.pkl', 'wb') as f:
                        pickle.dump(training_records, f)
                    print("Training records saved")
                except Exception as ex:
                    print(f"Error saving checkpoint: {ex}")

            # Save best model based on individual episode reward
            if total_reward > best_reward:
                best_reward = total_reward
                no_improvement_count = 0  # Reset counter
                try:
                    agent.save("models/best.keras")
                    print(f"New best reward: {best_reward:.1f} - Saved best_reward_model.keras")
                except Exception as ex:
                    print(f"Error saving best reward model: {ex}")
            else:
                no_improvement_count += 1

            # Save best model based on average reward (more stable)
            if len(rewards_history) >= 50 and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                try:
                    agent.save("models/best_avg.keras")
                    print(f"New best average reward: {best_avg_reward:.1f} - Saved best_avg_model.keras")
                except Exception as ex:
                    print(f"Error saving best average model: {ex}")
            
            # Early stopping if no improvement for a while
            if no_improvement_count >= patience:
                print(f"Early stopping after {e} episodes due to no improvement")
                break

            # Try to adapt learning rate over time if possible
            if e > 0 and e % 200 == 0 and hasattr(agent.model, 'reduce_learning_rate'):
                try:
                    agent.model.reduce_learning_rate(0.5)  # Reduce learning rate by half
                    print(f"Reduced learning rate at episode {e}")
                except Exception as ex:
                    # Just continue if this fails
                    pass

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

        # Save the final trained model
        try:
            agent.save("models/final_model.keras")
            print(f"Final model saved after {episodes} episodes")

            # Save final training records
            with open('data/training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            print("Final training records saved")

            # When saving
            with open('data/full_history.pkl', 'wb') as f:
                pickle.dump(full_reward_history, f)
            print("Full reward history saved")
        except Exception as ex:
            print(f"Error saving final model: {ex}")
            
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f} seconds")
        print(f"Best reward achieved: {best_reward:.1f}")
        print(f"Best average reward: {best_avg_reward:.1f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        try:
            # Save the model on interrupt
            agent.save("models/interrupted_model.keras")
            with open('data/interrupted_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            print("Model and training records saved")
        except Exception as ex:
            print(f"Error saving model after interrupt: {ex}")

if __name__ == "__main__":
    train()