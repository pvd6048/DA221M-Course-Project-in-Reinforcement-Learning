import random # Used for randomly sampling from the buffer
import numpy as np # Used for stacking sampled batch elements
import os # Used for checking/creating directories when saving the buffer (needed for save_buffer)
import pickle # Used for serializing/deserializing the buffer for saving/loading (needed for save/load_buffer)

# Defines a class for storing and sampling experience tuples (s, a, r, s', done)
class ReplayMemory:
    # Constructor for the ReplayMemory class
    def __init__(self, capacity, seed):
        # Set the random seed for reproducibility of sampling
        random.seed(seed)
        # Maximum number of transitions to store in the buffer
        self.capacity = capacity
        # Initialize an empty list to store the experience tuples
        self.buffer = []
        # Initialize the position pointer for inserting new experiences (circular buffer index)
        self.position = 0

    # Method to add a new experience tuple to the buffer
    def push(self, state, action, reward, next_state, done):
        # If the buffer is not yet full, expand it by one slot
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) # Append a placeholder
        # Store the new experience tuple at the current position, potentially overwriting an old one
        self.buffer[self.position] = (state, action, reward, next_state, done)
        # Update the position pointer, wrapping around if capacity is reached (circular buffer logic)
        self.position = (self.position + 1) % self.capacity

    # Method to sample a random batch of experiences from the buffer
    def sample(self, batch_size):
        # Randomly select `batch_size` experiences from the buffer without replacement
        batch = random.sample(self.buffer, batch_size)
        # Unzip the batch of tuples into separate lists for states, actions, rewards, etc.
        # Then, use np.stack to convert these lists into NumPy arrays (batch dimensions first)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        # Return the stacked NumPy arrays for each component of the transitions
        return state, action, reward, next_state, done

    # Method to return the current number of experiences stored in the buffer
    def __len__(self):
        # The length is simply the current size of the internal list `self.buffer`
        return len(self.buffer)

    # Method to save the entire buffer content to a file using pickle
    def save_buffer(self, env_name, suffix="", save_path=None):
        # Check if the 'checkpoints' directory exists, create it if not
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        # Define the default save path if none is provided
        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        # Print the save location
        print('Saving buffer to {}'.format(save_path))

        # Open the specified file in binary write mode ('wb')
        with open(save_path, 'wb') as f:
            # Use pickle to dump the entire buffer list into the file
            pickle.dump(self.buffer, f)

    # Method to load the buffer content from a previously saved file
    def load_buffer(self, save_path):
        # Print the load location
        print('Loading buffer from {}'.format(save_path))

        # Open the specified file in binary read mode ('rb')
        with open(save_path, "rb") as f:
            # Load the buffer list from the pickle file
            self.buffer = pickle.load(f)
            # Reset the position pointer based on the loaded buffer size (for circular buffer logic)
            self.position = len(self.buffer) % self.capacity