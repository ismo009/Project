import numpy as np
import tensorflow as tf
import os
import pickle
import time
from src.environment import SnakeEnvironment
from src.agent import Agent

def train():
    # Setup directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Create environment and agent
    env = SnakeEnvironment(width=10, height=10)  # Smaller grid for faster training
    agent = Agent(state_size=env.state_size, action_size=env.action_size)
    
    # Training parameters 
    episodes = 2000
    batch_size = 128
    
    # For tracking progress
    start_time = time.time()
    training_records = []
    rewards_window = []
    best_reward = -float('inf')
    
    try:
        print("Starting training...")
        for e in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            step = 0
            max_steps = 100  # Limit steps to prevent infinite episodes
            
            # Run a single episode
            while not done and step < max_steps:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step += 1
            
            # Train the agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            # Update records
            training_records.append((e, total_reward))
            rewards_window.append(total_reward)
            if len(rewards_window) > 100:
                rewards_window.pop(0)
            
            # Calculate moving average
            avg_reward = sum(rewards_window) / len(rewards_window)
            
            # Track best model
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save("models/best_model.keras")
            
            # Print progress
            if e % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode {e}/{episodes} | Reward: {total_reward:.1f} | "
                      f"Moving Avg: {avg_reward:.1f} | Epsilon: {agent.epsilon:.3f} | "
                      f"Steps: {step} | Time: {elapsed:.1f}s")
            
            # Save checkpoint
            if e % 100 == 0 and e > 0:
                agent.save(f"models/checkpoint_episode_{e}.keras")
                with open('data/training_records.pkl', 'wb') as f:
                    pickle.dump(training_records, f)
                print(f"Checkpoint saved at episode {e}")
        
        # Save final model and records
        agent.save("models/final_model.keras")
        with open('data/final_training_records.pkl', 'wb') as f:
            pickle.dump(training_records, f)
        print(f"Training completed in {time.time() - start_time:.1f}s")
        print(f"Best reward achieved: {best_reward}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save("models/interrupted_model.keras")
        with open('data/interrupted_records.pkl', 'wb') as f:
            pickle.dump(training_records, f)
        print("Progress saved")

if __name__ == "__main__":
    train()