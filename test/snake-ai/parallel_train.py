import numpy as np
import tensorflow as tf
import os
import pickle
import time
import logging
from multiprocessing import Pool, cpu_count, set_start_method
from src.environment import SnakeEnvironment
from src.agent import Agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# Try to use 'spawn' method for better compatibility across platforms
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, ignore
    pass

def run_episode(env_config):
    """Run a single episode in a separate process"""
    # Create a new environment for this process
    env = SnakeEnvironment(width=env_config['width'], height=env_config['height'])
    state = env.reset()
    done = False
    experiences = []
    total_reward = 0
    epsilon = env_config['epsilon']
    steps = 0
    max_steps = 200  # Prevent infinite loops
    
    # Since we can't share the model across processes, we'll use a combination of
    # epsilon-greedy exploration and a simple heuristic for action selection
    while not done and steps < max_steps:
        steps += 1
        
        # Use epsilon-greedy policy with heuristic backup
        if np.random.random() < epsilon:
            # Pure exploration
            action = np.random.randint(0, env.action_size)
        else:
            # When not exploring, use a simple heuristic based on the state values
            state_values = np.array(state)
            danger = state_values[0:4]  # First 4 values represent danger in 4 directions
            food_dir = state_values[8:12]  # Last 4 values represent food direction
            
            # Prefer directions where there's food and no danger
            combined = food_dir - (danger * 100)  # Heavily penalize danger directions
            action = np.argmax(combined)
            
        next_state, reward, done = env.step(action)
        experiences.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
    
    return experiences, total_reward, env.score

def main():
    # Directory setup
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Try to determine the best number of parallel environments based on CPU cores
    available_cores = cpu_count()
    n_envs = max(1, available_cores - 1)  # Leave one CPU core free
    
    # Cap at a reasonable maximum to prevent overhead
    n_envs = min(n_envs, 16)
    logging.info(f"Training with {n_envs} parallel environments on {available_cores} cores")
    
    # Initialize agent
    state_size = 12  # Match state size from environment
    action_size = 4  # up, right, down, left
    agent = Agent(state_size=state_size, action_size=action_size)
    
    # Training parameters
    episodes = 1000
    batch_size = 128
    save_frequency = 50
    
    # Try to load previous training state if it exists
    try:
        with open('data/training_records.pkl', 'rb') as f:
            training_records = pickle.load(f)
            last_episode = max([record[0] for record in training_records]) + 1
            logging.info(f"Resuming training from episode {last_episode}")
            
            # Try to load the most recent model
            try:
                latest_model = f"models/dqn_model_episode_{(last_episode // save_frequency) * save_frequency}.keras"
                if os.path.exists(latest_model):
                    agent.load(latest_model)
                    logging.info(f"Loaded model from {latest_model}")
                else:
                    logging.warning(f"Could not find latest model at {latest_model}")
            except Exception as ex:
                logging.error(f"Error loading model: {ex}")
    except FileNotFoundError:
        training_records = []
        last_episode = 0
        logging.info("Starting new training session")
    
    # For tracking progress
    start_time = time.time()
    rewards_history = []
    best_reward = float('-inf')
    best_avg_reward = float('-inf')
    
    try:
        # Run training for the specified number of episodes
        for e in range(last_episode, episodes, n_envs):
            # Create environment configurations for parallel execution
            env_configs = [
                {
                    'width': 10, 
                    'height': 10, 
                    'epsilon': max(0.01, agent.epsilon)
                }
                for i in range(min(n_envs, episodes - e))
            ]
            
            # Run episodes in parallel
            with Pool(len(env_configs)) as pool:
                results = pool.map(run_episode, env_configs)
                
            # Process results
            all_experiences = []
            episode_rewards = []
            max_score = 0
            
            for experiences, total_reward, score in results:
                all_experiences.extend(experiences)
                episode_rewards.append(total_reward)
                training_records.append((e + len(episode_rewards) - 1, total_reward, score))
                max_score = max(max_score, score)
            
            # Print progress
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            rewards_history.append(avg_reward)
            if len(rewards_history) > 100:
                rewards_history = rewards_history[-100:]
            
            moving_avg = sum(rewards_history) / len(rewards_history)
            elapsed = time.time() - start_time
            
            logging.info(f"Episodes {e}-{e+len(episode_rewards)-1}/{episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | Moving Avg: {moving_avg:.2f} | "
                      f"Max Score: {max_score} | Epsilon: {agent.epsilon:.4f} | Time: {elapsed:.1f}s")
            
            # Train on all experiences
            if len(all_experiences) > 0:
                for exp in all_experiences:
                    agent.remember(*exp)
                agent.replay(batch_size=min(batch_size, len(all_experiences)))
                
            # Save model periodically
            if e % save_frequency == 0 or e + n_envs >= episodes:
                try:
                    model_path = f"models/dqn_model_episode_{e}.keras"
                    agent.save(model_path)
                    logging.info(f"Model saved to {model_path}")
                    
                    # Save training records
                    with open('data/training_records.pkl', 'wb') as f:
                        pickle.dump(training_records, f)
                    
                except Exception as ex:
                    logging.error(f"Error during saving: {ex}")
            
            # Save best model based on average reward
            if moving_avg > best_avg_reward and len(rewards_history) >= 10:
                best_avg_reward = moving_avg
                try:
                    agent.save("models/best_avg_model.keras")
                    logging.info(f"New best average reward: {best_avg_reward:.2f} - Saved best_avg_model.keras")
                except Exception as ex:
                    logging.error(f"Error saving best average model: {ex}")
            
            # Save best model based on single episode reward
            current_best = max(episode_rewards)
            if current_best > best_reward:
                best_reward = current_best
                try:
                    agent.save("models/best_reward_model.keras")
                    logging.info(f"New best reward: {best_reward:.2f} - Saved best_reward_model.keras")
                except Exception as ex:
                    logging.error(f"Error saving best reward model: {ex}")
        
        # Save final model
        try:
            agent.save("models/dqn_model_final.keras")
            logging.info(f"Final model saved. Total training time: {time.time() - start_time:.1f}s")
            
            # Final save of training records
            with open('data/training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
        except Exception as ex:
            logging.error(f"Error during final model saving: {ex}")
    
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
        # Save the model on interrupt too
        try:
            agent.save("models/dqn_model_interrupted.keras")
            with open('data/training_records_interrupted.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            logging.info("Model and training records saved on interrupt")
        except Exception as ex:
            logging.error(f"Error during interrupt saving: {ex}")
    
    # Save training summary
    try:
        # Calculate final statistics
        final_rewards = rewards_history[-min(100, len(rewards_history)):]
        final_avg = sum(final_rewards) / len(final_rewards)
        
        # Find best performing models
        scores = [record[2] for record in training_records]
        max_score = max(scores) if scores else 0
        
        with open('data/training_summary.txt', 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"---------------\n")
            f.write(f"Episodes trained: {min(episodes, len(training_records))}\n")
            f.write(f"Best single reward: {best_reward:.2f}\n")
            f.write(f"Best average reward: {best_avg_reward:.2f}\n")
            f.write(f"Final average reward: {final_avg:.2f}\n")
            f.write(f"Maximum score achieved: {max_score}\n")
            f.write(f"Total training time: {time.time() - start_time:.1f} seconds\n")
        
        logging.info("Training summary saved to data/training_summary.txt")
    except Exception as ex:
        logging.error(f"Error saving training summary: {ex}")

if __name__ == "__main__":
    try:
        # Set GPU memory growth to avoid memory allocation issues
        physical_devices = tf.config.list_physical_devices('GPU') 
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                logging.info(f"GPU memory growth enabled for {device}")
        
        main()
    except Exception as ex:
        logging.error(f"Unhandled exception: {ex}")
        raise