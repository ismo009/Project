import numpy as np
import random
from collections import deque
from models.dqn_model import DQN

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.priorities = deque(maxlen=2000)
        self.alpha = 0.6  # Priority exponent
        self.epsilon_mem = 0.01  # Small constant to avoid zero priority
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        # Create DQN model
        model = DQN(self.state_size, self.action_size, self.learning_rate)
        return model
        
    def remember(self, state, action, reward, next_state, done):
        # Calculate TD error as priority (simplified version)
        state_tensor = np.array([state], dtype=np.float32)
        next_state_tensor = np.array([next_state], dtype=np.float32)
        
        current_val = self.model.predict(state_tensor)[0][action]
        next_val = np.max(self.model.predict(next_state_tensor)[0])
        target = reward + self.gamma * next_val * (1 - done)
        priority = abs(target - current_val) + self.epsilon_mem
        
        # Store experience and priority
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def act(self, state):
        # Decide action: explore or exploit
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.array([state], dtype=np.float32)  # Ensure proper shape and dtype
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=128):  # Increased from 32 to 128
        # Train on batch from memory
        if len(self.memory) < batch_size:
            return
            
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample batch based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        minibatch = [self.memory[idx] for idx in indices]
        
        states = np.array([i[0] for i in minibatch], dtype=np.float32)
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch], dtype=np.float32)
        dones = np.array([i[4] for i in minibatch])
        
        # Get predictions
        targets = rewards + self.gamma * (np.amax(self.model.predict(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict(states)
        
        # Update targets for actions taken
        for i, action in enumerate(actions):
            targets_full[i][action] = targets[i]
        
        # Train the model
        self.model.train_on_batch(states, targets_full)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, file_path):
        """Save the agent's model"""
        success = self.model.save(file_path)
        if success:
            print(f"Model successfully saved to {file_path}")
        else:
            print(f"Failed to save model to {file_path}")
        return success
    
    def load(self, file_path):
        """Load the agent's model"""
        success = self.model.load(file_path)
        if success:
            print(f"Model successfully loaded from {file_path}")
        else:
            print(f"Failed to load model from {file_path}")
        return success