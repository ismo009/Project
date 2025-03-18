import numpy as np
import tensorflow as tf
import os
import pickle
import time
from src.environment import SnakeEnvironment
from src.agent import Agent

def train():
    # Directory setup
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Environment and agent
    env = SnakeEnvironment(width=10, height=10)  # Smaller grid for faster learning
    agent = Agent(state_size=env.state_size, action_size=env.action_size)
    agent.epsilon_decay = 0.99  # Slightly faster exploration decay
    
    # Training parameters
    episodes = 2000
    batch_size = 128
    train_every = 4  # Train every 4 episodes
    
    # For tracking progress
    start_time = time.time()
    rewards_history = []
    training_records = []
    best_reward = float('-inf')
    patience = 100
    no_improvement = 0
    
    # Load previous training if available
    try:
        with open('data/training_records.pkl', 'rb') as f:
            training_records = pickle.load(f)
            print(f"Loaded {len(training_records)} previous training records")
    except FileNotFoundError:
        training_records = []
    
    try:
        for e in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 200  # Limit episode length to prevent very long episodes
            
            while not done and steps < max_steps:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
            
            # Train every few episodes for efficiency
            if e % train_every == 0:
                agent.replay(batch_size=batch_size)
                
            # Track rewards
            rewards_history.append(total_reward)
            if len(rewards_history) > 100:
                rewards_history = rewards_history[-100:]
            moving_avg = sum(rewards_history) / len(rewards_history)
            training_records.append((e, total_reward))
            
            # Print progress
            if e % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode {e}/{episodes} | Reward: {total_reward:.2f} | "
                      f"Moving Avg: {moving_avg:.2f} | Epsilon: {agent.epsilon:.4f} | "
                      f"Steps: {steps} | Time: {elapsed:.1f}s")
            
            # Early stopping
            if total_reward > best_reward:
                best_reward = total_reward
                no_improvement = 0
                # Save best model
                agent.save("models/best_dqn_model.h5")
            else:
                no_improvement += 1
                
            if no_improvement >= patience and e > 500:  # Give at least 500 episodes before early stop
                print(f"Early stopping at episode {e} with best reward: {best_reward}")
                break
                
            # Save model periodically
            if e % 100 == 0 and e > 0:
                agent.save(f"models/dqn_model_episode_{e}.h5")
                # Save training records
                with open('data/training_records.pkl', 'wb') as f:
                    pickle.dump(training_records, f)
        
        # Save final model and records
        agent.save("models/dqn_model_final.h5")
        with open('data/training_records.pkl', 'wb') as f:
            pickle.dump(training_records, f)
            
        print(f"Training complete. Best reward: {best_reward}")
        print(f"Total training time: {time.time() - start_time:.1f}s")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save on interrupt
        agent.save("models/dqn_model_interrupted.h5")
        with open('data/training_records_interrupted.pkl', 'wb') as f:
            pickle.dump(training_records, f)
        print("Model and training records saved")

if __name__ == "__main__":
    train()