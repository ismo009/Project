import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQN:
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))  # Increased from 64 to 256
        model.add(Dense(256, activation='relu'))  # Increased from 64 to 256
        # You could even add another layer for more capacity
        model.add(Dense(128, activation='relu'))  # New layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        from random import sample
        minibatch = sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def predict_on_batch(self, state):
        return self.model.predict(state, verbose=0)

    def train_on_batch(self, state, target):
        return self.model.fit(state, target, epochs=1, verbose=0, batch_size=len(state))

    def load(self, file_path):
        # Make sure the file path has the correct extension
        if not file_path.endswith('.keras'):
            file_path += '.keras'
        try:
            # Use tf.keras.models.load_model instead
            self.model = tf.keras.models.load_model(file_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def save(self, file_path):
        # Make sure the file path has the correct extension
        if not file_path.endswith('.keras'):
            file_path += '.keras'
        try:
            # Use tf.keras.models.save_model instead
            tf.keras.models.save_model(self.model, file_path)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False