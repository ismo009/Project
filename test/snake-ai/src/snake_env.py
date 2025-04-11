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
        self.first_apple_eaten = False
        
        # Updated state size to include new features
        self.state_size = 23  # Was 12 before our changes
        self.action_size = 4  # up, right, down, left
        
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
        """Enhanced state representation with 22 features for better late-game performance"""
        head_x, head_y = self.snake_pos
        food_x, food_y = self.food_pos
        
        # Current danger detection (immediate neighbors)
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
        
        # NEW: Look ahead danger detection (2 steps away)
        danger_l2 = False
        if not danger_l:  # Only check if first step is safe
            point_l2 = [point_l[0] - 1, point_l[1]]
            danger_l2 = point_l2 in self.snake_body or point_l2[0] < 0 or point_l2[0] >= self.width
        
        danger_r2 = False
        if not danger_r:
            point_r2 = [point_r[0] + 1, point_r[1]]
            danger_r2 = point_r2 in self.snake_body or point_r2[0] < 0 or point_r2[0] >= self.width
        
        danger_u2 = False
        if not danger_u:
            point_u2 = [point_u[0], point_u[1] - 1]
            danger_u2 = point_u2 in self.snake_body or point_u2[1] < 0 or point_u2[1] >= self.height
        
        danger_d2 = False
        if not danger_d:
            point_d2 = [point_d[0], point_d[1] + 1]
            danger_d2 = point_d2 in self.snake_body or point_d2[1] < 0 or point_d2[1] >= self.height
        
        # NEW: Path freedom metrics (critical for late game)
        # Check how many free spaces are around potential next positions
        free_space_l = self._count_free_neighbors(point_l) if not danger_l else 0
        free_space_r = self._count_free_neighbors(point_r) if not danger_r else 0
        free_space_u = self._count_free_neighbors(point_u) if not danger_u else 0
        free_space_d = self._count_free_neighbors(point_d) if not danger_d else 0
        
        # Normalize free space to [0, 1] range
        free_space_l = free_space_l / 8
        free_space_r = free_space_r / 8
        free_space_u = free_space_u / 8
        free_space_d = free_space_d / 8
        
        # NEW: Distance to tail (helps with path planning)
        tail_x, tail_y = self.snake_body[-1]
        tail_dist = abs(head_x - tail_x) + abs(head_y - tail_y)
        # Normalize by max possible distance on board
        max_dist = self.width + self.height
        tail_dist_normalized = tail_dist / max_dist
        
        # NEW: Board density metric (how crowded the board is)
        board_density = len(self.snake_body) / (self.width * self.height)
        
        # Add late game flag (changes strategy when snake is long)
        is_late_game = 1.0 if len(self.snake_body) > 20 else 0.0
        
        # Combine all features - this now has 22 elements
        state = np.array([
            # Original features
            danger_l, danger_r, danger_u, danger_d,
            dir_l, dir_r, dir_u, dir_d,
            food_left, food_right, food_up, food_down,
            
            # New features for late game
            danger_l2, danger_r2, danger_u2, danger_d2,
            free_space_l, free_space_r, free_space_u, free_space_d,
            tail_dist_normalized, board_density, is_late_game
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        # Track previous state for reward calculation
        prev_head_x, prev_head_y = self.snake_pos.copy()
        prev_tail = self.snake_body[-1].copy() if self.snake_body else None
        
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
            if not self.first_apple_eaten:
                self.first_apple_eaten = True
            
            # Progressive milestone rewards system
            if self.score == 5:
                reward += 15  # Small milestone
                print(f"Milestone reached: 5 apples! Bonus: +15")
            elif self.score == 10:
                reward += 25  # Medium milestone
                print(f"Milestone reached: 10 apples! Bonus: +25")
            elif self.score == 15: 
                reward += 40  # Big milestone
                print(f"Milestone reached: 15 apples! Bonus: +40")
            elif self.score == 20:
                reward += 60  # Major milestone 
                print(f"Milestone reached: 20 apples! Bonus: +60")
            elif self.score == 25:
                reward += 100  # Super milestone
                print(f"Milestone reached: 25 apples! Bonus: +100")
            elif self.score > 25 and self.score % 5 == 0:
                milestone_reward = 120  # Massive rewards for late-game milestones
                reward += milestone_reward
                print(f"Late-game milestone: {self.score} apples! Bonus: +{milestone_reward}")
            
            # Base reward with exponential scaling
            base_reward = 10
            # Exponential increase based on score
            reward += base_reward * (1.01 ** (self.score))
            
            self.score += 1
            self.steps_without_food = 0
        else:
            self.snake_body.pop()
            
            # IMPROVED LATE GAME: Use more sophisticated movement rewards
            if len(self.snake_body) > 15:  # Only apply these for longer snakes
                # Calculate space efficiency - reward for moving toward open areas
                head_x, head_y = self.snake_pos
                space_around = self._count_free_neighbors(self.snake_pos)
                
                # Reward more for staying in open areas in late game
                open_space_reward = space_around * 0.05
                reward += open_space_reward
                
                # Add tail-following behavior for very long snakes
                # This encourages the snake to follow its own tail in late game
                if len(self.snake_body) > 20:
                    tail_x, tail_y = self.snake_body[-1]
                    curr_tail_dist = abs(head_x - tail_x) + abs(head_y - tail_y)
                    prev_tail_dist = abs(prev_head_x - prev_tail[0]) + abs(prev_head_y - prev_tail[1])
                    
                    # Reward getting closer to tail in late game
                    if self.steps_without_food > 10 and curr_tail_dist < prev_tail_dist:
                        reward += 0.2
            else:
                # Original movement rewards for early game
                reward = -0.1  # Small penalty per step
                
                # Distance-based reward component
                head_x, head_y = self.snake_pos
                food_x, food_y = self.food_pos
                distance = abs(head_x - food_x) + abs(head_y - food_y)
                prev_distance = abs(prev_head_x - food_x) + abs(prev_head_y - food_y)
                
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
                if enclosed_area > 0.5:  # If more than 50% of surrounding area is occupied
                    reward = -50  # Large penalty for enclosing itself
                    print(f"Snake enclosed itself! Penalty applied.")
                else:
                    # Scale penalty based on score - less harsh for long snakes
                    # This acknowledges that longer snakes are harder to manage
                    if self.score > 20:
                        reward = -15  # Reduced penalty for self-collision in late game
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
        
        # Add penalty for taking too many steps without food
        # But make this less strict in late game where navigation is harder
        max_steps_without_food = 100 if len(self.snake_body) > 15 else 50
        if self.steps_without_food > max_steps_without_food:
            reward -= 25
            self.done = True
            print(f"Episode terminated: Too many steps without food ({self.steps_without_food})")
        
        # Update state and return
        next_state = self._get_state()
        return next_state, reward, self.done

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
    
    def _count_free_neighbors(self, position):
        """Count number of free neighboring cells around a position"""
        if not (0 <= position[0] < self.width and 0 <= position[1] < self.height):
            return 0
            
        x, y = position
        neighbors = [
            [x-1, y], [x+1, y], [x, y-1], [x, y+1],
            [x-1, y-1], [x+1, y-1], [x-1, y+1], [x+1, y+1]
        ]
        
        free_count = 0
        for nx, ny in neighbors:
            # Check if position is valid and not occupied by snake
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                [nx, ny] not in self.snake_body):
                free_count += 1
        
        return free_count