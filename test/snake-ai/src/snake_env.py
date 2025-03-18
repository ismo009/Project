import numpy as np
import random

class SnakeEnvironment:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.snake_pos = None
        self.snake_body = None
        self.food_pos = None
        self.score = 0
        self.done = False
        self.direction = None
        self.steps_taken = 0
        self.steps_without_food = 0
        self.first_apple_eaten = False  # Flag to track if the first apple has been eaten
        
        # Define state and action dimensions
        self.state_size = 12
        self.action_size = 4  # up, down, left, right
        
        self.reset()
    
    def reset(self):
        # Initialize snake at the center
        self.snake_pos = [self.width//2, self.height//2]
        self.snake_body = [[self.width//2, self.height//2], 
                           [self.width//2-1, self.height//2], 
                           [self.width//2-2, self.height//2]]
        self.food_pos = self._place_food()
        self.score = 0
        self.done = False
        self.direction = 'RIGHT'
        self.steps_taken = 0
        self.steps_without_food = 0
        self.first_apple_eaten = False  # Reset first apple flag
        
        return self._get_state()
    
    def _place_food(self):
        # Place food at a random position that's not occupied by the snake
        food = [random.randrange(1, self.width-1), random.randrange(1, self.height-1)]
        while food in self.snake_body:
            food = [random.randrange(1, self.width-1), random.randrange(1, self.height-1)]
        return food
    
    def _get_state(self):
        # Create state representation
        head_x, head_y = self.snake_pos
        food_x, food_y = self.food_pos
        
        # Danger positions (immediate neighbors)
        point_l = [head_x - 1, head_y]
        point_r = [head_x + 1, head_y]
        point_u = [head_x, head_y - 1]
        point_d = [head_x, head_y + 1]
        
        # Current direction one-hot encoded
        dir_l = self.direction == 'LEFT'
        dir_r = self.direction == 'RIGHT'
        dir_u = self.direction == 'UP'
        dir_d = self.direction == 'DOWN'
        
        # Danger detected (collision would happen)
        danger_l = point_l in self.snake_body or point_l[0] < 0 or point_l[0] >= self.width
        danger_r = point_r in self.snake_body or point_r[0] < 0 or point_r[0] >= self.width
        danger_u = point_u in self.snake_body or point_u[1] < 0 or point_u[1] >= self.height
        danger_d = point_d in self.snake_body or point_d[1] < 0 or point_d[1] >= self.height
        
        # Food direction
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        
        # Convert to numpy array - this has 12 elements
        state = np.array([
            danger_l, danger_r, danger_u, danger_d,
            dir_l, dir_r, dir_u, dir_d,
            food_left, food_right, food_up, food_down
        ], dtype=np.float32)  # Using float32 for better compatibility with TensorFlow
        
        return state
    
    def step(self, action):
        # Increment step counters
        self.steps_taken += 1
        self.steps_without_food += 1
        
        # Map action (0,1,2,3) to direction
        if action == 0:   # UP
            self.direction = 'UP'
        elif action == 1: # RIGHT
            self.direction = 'RIGHT'
        elif action == 2: # DOWN
            self.direction = 'DOWN'
        elif action == 3: # LEFT
            self.direction = 'LEFT'
        
        # Move snake
        if self.direction == 'UP':
            self.snake_pos[1] -= 1
        elif self.direction == 'DOWN':
            self.snake_pos[1] += 1
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= 1
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += 1
        
        # Check if snake ate food
        reward = 0
        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos == self.food_pos:
            self.food_pos = self._place_food()
            reward = 10
            self.score += 1
            
            # Mark that the first apple has been eaten
            if not self.first_apple_eaten:
                self.first_apple_eaten = True
                # Give bonus reward for eating first apple
                reward += 5
            
            self.steps_without_food = 0  # Reset counter when food is eaten
        else:
            self.snake_body.pop()
            reward = -0.1  # Small penalty for each step to encourage finding food quickly
            
            # Calculate distance-based reward component
            head_x, head_y = self.snake_pos
            food_x, food_y = self.food_pos
            distance = abs(head_x - food_x) + abs(head_y - food_y)
            prev_distance = abs(self.snake_body[0][0] - food_x) + abs(self.snake_body[0][1] - food_y)
            
            # Small reward for getting closer to food
            if distance < prev_distance:
                reward += 0.1
        
        # Check if game is over (snake hit wall or itself)
        hit_wall = (self.snake_pos[0] < 0 or 
                   self.snake_pos[0] >= self.width or
                   self.snake_pos[1] < 0 or 
                   self.snake_pos[1] >= self.height)
        
        hit_self = list(self.snake_pos) in self.snake_body[1:]
        
        if hit_wall or hit_self:
            self.done = True
            
            # Different penalties based on how the snake died
            if hit_self:
                # Check if the snake enclosed itself
                enclosed_area = self._calculate_enclosed_area()
                if enclosed_area > 0.5:  # If more than 50% of the surrounding area is occupied
                    reward = -50  # Large penalty for enclosing itself
                    print(f"Snake enclosed itself! Penalty applied.")
                else:
                    reward = -25  # Standard penalty for self-collision
            else:
                reward = -10  # Standard penalty for hitting wall
                
            # Additional penalty based on score to discourage early deaths
            if self.score < 3:
                reward -= 5  # Extra penalty for dying with a low score
        
        # Major penalty ONLY if the snake hasn't eaten the first apple within 20 steps
        if not self.first_apple_eaten and self.steps_without_food >= 20:
            reward -= 50  # Large penalty for not finding first food quickly
            self.done = True
            print(f"Episode terminated: Failed to eat first apple within 20 steps")
        
        return self._get_state(), reward, self.done

    def _calculate_enclosed_area(self):
        """
        Calculate the percentage of the 8 surrounding cells around the snake's head
        that are occupied by the snake's body.
        """
        head_x, head_y = self.snake_pos
        surrounding_positions = [
            [head_x - 1, head_y],  # Left
            [head_x + 1, head_y],  # Right
            [head_x, head_y - 1],  # Up
            [head_x, head_y + 1],  # Down
            [head_x - 1, head_y - 1],  # Top-left
            [head_x + 1, head_y - 1],  # Top-right
            [head_x - 1, head_y + 1],  # Bottom-left
            [head_x + 1, head_y + 1],  # Bottom-right
        ]

        # Count how many of these positions are occupied by the snake's body
        occupied_count = sum(1 for pos in surrounding_positions if pos in self.snake_body)

        # Calculate the percentage of occupied cells
        return occupied_count / len(surrounding_positions)