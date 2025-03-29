import pickle
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
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

def continue_enhanced_training(model_path="models/enhanced_checkpoint_2950.keras", episodes=3000):
    """Continue training enhanced model with focus on late-game performance"""
    print(f"Continuing training for enhanced model: {model_path}")
    print("Optimizing for late-game performance...")
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
    env = SnakeEnvironment(width=10, height=10)
    agent = Agent(state_size=env.state_size, action_size=env.action_size)
    
    # Load the existing model
    try:
        agent.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Extract episode number from filename
    try:
        start_episode = int(model_path.split('_')[-1].split('.')[0])
        print(f"Continuing from episode {start_episode}")
    except:
        start_episode = 3000
        print(f"Could not extract episode number, starting from {start_episode}")
    
    # Set specific epsilon parameters for late-game focus
    agent.epsilon = 0.3        # Lower epsilon - we want to exploit more but still explore
    agent.epsilon_min = 0.05   # Small min epsilon for ongoing exploration
    agent.epsilon_decay = 0.998 # Slow decay to maintain some exploration
    
    # Try to adjust learning rate to fine-tune
    try:
        # Get current optimizer and learning rate
        current_lr = agent.model.model.optimizer.learning_rate.numpy()
        # Reduce learning rate for fine-tuning
        new_lr = current_lr * 0.5
        # Create new optimizer with reduced learning rate
        new_optimizer = Adam(learning_rate=new_lr)
        # Recompile with new optimizer
        agent.model.model.compile(optimizer=new_optimizer, loss='mse')
        print(f"Reduced learning rate from {current_lr} to {new_lr} for fine-tuning")
    except Exception as e:
        print(f"Could not adjust learning rate: {e}")
    
    print(f"Training parameters: epsilon={agent.epsilon}, decay={agent.epsilon_decay}, min={agent.epsilon_min}")

    # Load training records if available
    try:
        with open('data/enhanced_training_records.pkl', 'rb') as f:
            training_records = pickle.load(f)
            print(f"Loaded {len(training_records)} previous training records")
    except (FileNotFoundError, ValueError):
        training_records = []
        print("Starting new training records")

    # Initialize tracking variables
    best_reward = float('-inf')
    best_avg_reward = float('-inf')
    rewards_history = []
    best_late_game_score = 0  # Track best performance in late game
    
    # Create a filename-friendly timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create a new model name for the continued training
    model_prefix = "enhanced_latev2_"
    
    # Memory limit
    max_memory_size = 100000
    
    # Prioritize late-game experiences (with score > 15)
    late_game_memory = []
    late_game_memory_size = 20000  # Size limit for late-game experiences
    
    try:
        for e in range(episodes):
            current_episode = start_episode + e
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 300  # Increased for longer games
            episode_memory = []  # Store experiences for this episode
            
            while not done and steps < max_steps:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                # Store experience for later processing
                experience = (state, action, reward, next_state, done)
                episode_memory.append(experience)
                
                state = next_state
                total_reward += reward
                steps += 1

            # Add episode experiences to memory
            for experience in episode_memory:
                agent.remember(*experience)
                
            # Also add to late-game memory if score is high
            if env.score > 15:
                for experience in episode_memory:
                    late_game_memory.append(experience)
                print(f"Added {len(episode_memory)} late-game experiences (score: {env.score})")
                
            # Limit memory sizes
            if len(agent.memory) > max_memory_size:
                # Remove oldest general experiences
                excess = len(agent.memory) - max_memory_size
                agent.memory = agent.memory[excess:]
                
            if len(late_game_memory) > late_game_memory_size:
                # Remove oldest late-game experiences
                excess = len(late_game_memory) - late_game_memory_size
                late_game_memory = late_game_memory[excess:]
            
            # Train on general experiences
            if len(agent.memory) > 64:
                agent.replay(batch_size=64)
            
            # Also train on late-game experiences if available
            if len(late_game_memory) > 32:
                # Sample batch from late-game memory
                mini_batch = np.random.choice(len(late_game_memory), 32, replace=False)
                late_game_batch = [late_game_memory[i] for i in mini_batch]
                states = np.array([experience[0] for experience in late_game_batch])
                actions = np.array([experience[1] for experience in late_game_batch])
                rewards = np.array([experience[2] for experience in late_game_batch])
                next_states = np.array([experience[3] for experience in late_game_batch])
                dones = np.array([experience[4] for experience in late_game_batch])
                
                # Get Q values for current states
                q_values = agent.model.model.predict(states, verbose=0)
                next_q_values = agent.model.model.predict(next_states, verbose=0)
                
                # Update Q values with rewards and next state values
                for i in range(len(late_game_batch)):
                    if dones[i]:
                        q_values[i][actions[i]] = rewards[i]
                    else:
                        q_values[i][actions[i]] = rewards[i] + agent.gamma * np.max(next_q_values[i])
                
                # Train model on late-game experiences
                agent.model.model.fit(states, q_values, epochs=1, verbose=0)
                
                # Double the weight for really late games (score > 20)
                if env.score > 20:
                    agent.model.model.fit(states, q_values, epochs=1, verbose=0)
            
            # Update records and history
            training_records.append((current_episode, total_reward, env.score))
            rewards_history.append(total_reward)
            if len(rewards_history) > 100:
                rewards_history.pop(0)
            
            # Calculate moving average
            avg_reward = sum(rewards_history) / len(rewards_history)
            
            # Track best late-game score
            if env.score > best_late_game_score:
                best_late_game_score = env.score
                try:
                    agent.save(f"models/{model_prefix}best_score.keras")
                    print(f"New best score: {best_late_game_score} - Saved")
                except Exception as ex:
                    print(f"Error saving best score model: {ex}")
            
            # Log progress periodically
            if e % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode: {e}/{episodes} (#{current_episode}) | "
                      f"Reward: {total_reward:.1f} | Score: {env.score} | "
                      f"Avg(100): {avg_reward:.1f} | Epsilon: {agent.epsilon:.3f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Save periodically
            if e % 50 == 0 and e > 0:
                checkpoint_path = f"models/{model_prefix}checkpoint_{current_episode}.keras"
                try:
                    agent.save(checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
                    
                    # Save training records
                    with open('data/enhanced_continued_records.pkl', 'wb') as f:
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
                    
            # Every 100 episodes, do pure evaluations of different lengths
            if e % 100 == 0:
                # Temporarily set epsilon to 0 for pure exploitation
                orig_epsilon = agent.epsilon
                agent.epsilon = 0
                
                # Run standard evaluation episode
                eval_reward, eval_score = run_evaluation(agent, env, max_steps)
                print(f"Standard Eval (ε=0): Score={eval_score}, Reward={eval_reward:.1f}")
                
                # Run long evaluation (more steps allowed)
                long_reward, long_score = run_evaluation(agent, env, 500)
                print(f"Long Eval (ε=0): Score={long_score}, Reward={long_reward:.1f}")
                
                # Restore epsilon
                agent.epsilon = orig_epsilon

        # Save final continued model
        try:
            final_path = f"models/{model_prefix}final.keras"
            agent.save(final_path)
            print(f"Final enhanced model saved as {final_path}")
            
            # Also save with timestamp for backup
            agent.save(f"models/{model_prefix}{timestamp}.keras")
            
            # Save final records
            with open('data/enhanced_continued_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            print(f"Training records saved")
        except Exception as ex:
            print(f"Error saving final model: {ex}")
        
        total_time = time.time() - start_time
        print(f"Continued training completed in {total_time:.1f} seconds")
        print(f"Best reward achieved: {best_reward:.1f}")
        print(f"Best average reward: {best_avg_reward:.1f}")
        print(f"Best late-game score: {best_late_game_score}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        try:
            agent.save(f"models/{model_prefix}interrupted.keras")
            with open('data/enhanced_continued_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            print(f"Models and records saved")
        except Exception as ex:
            print(f"Error saving after interrupt: {ex}")

def run_evaluation(agent, env, max_steps):
    """Run an evaluation episode with no exploration"""
    eval_state = env.reset()
    eval_done = False
    eval_reward = 0
    eval_steps = 0
    
    while not eval_done and eval_steps < max_steps:
        eval_action = agent.act(eval_state)
        eval_state, r, eval_done = env.step(eval_action)
        eval_reward += r
        eval_steps += 1
    
    return eval_reward, env.score

if __name__ == "__main__":
    print("\n===== ENHANCED SNAKE AI CONTINUED TRAINING =====")
    print("This script will continue training your enhanced model with focus on late-game performance")
    
    # List available models
    models_dir = "models"
    if os.path.exists(models_dir):
        enhanced_models = [f for f in os.listdir(models_dir) if f.startswith('enhanced_') and f.endswith('.keras')]
        if enhanced_models:
            print("\nAvailable enhanced models:")
            for i, model_file in enumerate(enhanced_models):
                print(f"  {i+1}. {model_file}")
            print()  # Empty line
    
    # Get model path
    model_to_continue = input("Enter enhanced model to continue (default: models/enhanced_checkpoint_2950.keras): ").strip()
    if not model_to_continue:
        model_to_continue = "models/enhanced_checkpoint_2950.keras"
    
    if not model_to_continue.startswith("models/"):
        model_to_continue = f"models/{model_to_continue}"
    if not model_to_continue.endswith(".keras"):
        model_to_continue = f"{model_to_continue}.keras"
    
    # Get episode count
    try:
        episodes = int(input("How many episodes to train for? (default: 3000): ").strip() or "3000")
        if episodes <= 0:
            episodes = 3000
            print("Invalid value, using 3000 episodes")
    except ValueError:
        episodes = 3000
        print("Invalid input, using 3000 episodes")
    
    print(f"\nStarting continued training of {model_to_continue} for {episodes} episodes...\n")
    continue_enhanced_training(model_to_continue, episodes)