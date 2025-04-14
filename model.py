import torch
import torch.nn as nn # Base class for all neural network modules
import torch.nn.functional as F # Functional interface, e.g., activation functions
from torch.distributions import Normal # Normal distribution for Gaussian policy

# --- Constants for Gaussian Policy ---
# Maximum value for log standard deviation (to prevent std from becoming too large)
LOG_SIG_MAX = 2
# Minimum value for log standard deviation (to prevent std from becoming too small/zero)
LOG_SIG_MIN = -20
# Small epsilon value for numerical stability (e.g., to avoid log(0))
epsilon = 1e-6

# --- Weight Initialization Function ---
# Defines how to initialize weights for linear layers
def weights_init_(m):
    # Check if the module is an instance of a Linear layer
    if isinstance(m, nn.Linear):
        # Apply Xavier uniform initialization to the weight matrix
        # Gain=1 is suitable for ReLU activations (used later)
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # Initialize the bias term to a constant value of 0
        torch.nn.init.constant_(m.bias, 0)


# --- Value Network (V(s)) ---
# Not typically used directly in the main SAC update loop, but can be used for approximations or older SAC variants
class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        # Call the constructor of the parent class (nn.Module)
        super(ValueNetwork, self).__init__()

        # Define the first linear layer (input: state dim, output: hidden dim)
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        # Define the second linear layer (input: hidden dim, output: hidden dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # Define the output layer (input: hidden dim, output: 1, representing the value V(s))
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Apply the custom weight initialization function to all modules in this network
        self.apply(weights_init_)

    # Defines the forward pass of the network
    def forward(self, state):
        # Pass state through first linear layer and apply ReLU activation
        x = F.relu(self.linear1(state))
        # Pass through second linear layer and apply ReLU activation
        x = F.relu(self.linear2(x))
        # Pass through the output layer (no activation, output is the value estimate)
        x = self.linear3(x)
        # Return the estimated state value V(s)
        return x


# --- Q-Network (Critic: Q(s, a)) ---
# Implements the Critic network, estimating the action-value function Q(s, a)
# Contains TWO Q-networks internally (Q1 and Q2) for Clipped Double-Q Learning
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        # Call the constructor of the parent class (nn.Module)
        super(QNetwork, self).__init__()

        # --- Q1 architecture ---
        # First linear layer for Q1 (input: state dim + action dim, output: hidden dim)
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        # Second linear layer for Q1 (input: hidden dim, output: hidden dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer for Q1 (input: hidden dim, output: 1, representing Q1(s, a))
        self.linear3 = nn.Linear(hidden_dim, 1)

        # --- Q2 architecture (independent layers for the second Q-network) ---
        # First linear layer for Q2 (input: state dim + action dim, output: hidden dim)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        # Second linear layer for Q2 (input: hidden dim, output: hidden dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer for Q2 (input: hidden dim, output: 1, representing Q2(s, a))
        self.linear6 = nn.Linear(hidden_dim, 1)

        # Apply the custom weight initialization function to all modules in this network
        self.apply(weights_init_)

    # Defines the forward pass of the network
    def forward(self, state, action):
        # Concatenate state and action tensors along the feature dimension (dimension 1)
        xu = torch.cat([state, action], 1)

        # --- Forward pass for Q1 ---
        # Pass concatenated state-action through Q1's first layer and apply ReLU
        x1 = F.relu(self.linear1(xu))
        # Pass through Q1's second layer and apply ReLU
        x1 = F.relu(self.linear2(x1))
        # Pass through Q1's output layer
        x1 = self.linear3(x1)

        # --- Forward pass for Q2 ---
        # Pass concatenated state-action through Q2's first layer and apply ReLU
        x2 = F.relu(self.linear4(xu))
        # Pass through Q2's second layer and apply ReLU
        x2 = F.relu(self.linear5(x2))
        # Pass through Q2's output layer
        x2 = self.linear6(x2)

        # Return the outputs of both Q-networks (Q1(s, a), Q2(s, a))
        return x1, x2


# --- Gaussian Policy Network (Actor: π(a|s)) ---
# Implements the Actor network using a stochastic Gaussian policy for continuous action spaces
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        # Call the constructor of the parent class (nn.Module)
        super(GaussianPolicy, self).__init__()

        # --- Network Architecture ---
        # First linear layer (input: state dim, output: hidden dim)
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        # Second linear layer (input: hidden dim, output: hidden dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer for the mean (μ) of the Gaussian distribution (input: hidden dim, output: action dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        # Output layer for the log standard deviation (log σ) of the Gaussian distribution (input: hidden dim, output: action dim)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # Apply the custom weight initialization function
        self.apply(weights_init_)

        # --- Action Rescaling ---
        # If action space info is not provided, assume actions are already in [-1, 1]
        if action_space is None:
            self.action_scale = torch.tensor(1.) # No scaling
            self.action_bias = torch.tensor(0.) # No bias
        else:
            # Calculate the action scale and bias to map tanh output from [-1, 1] to [action_space.low, action_space.high]
            # action_scale = (high - low) / 2
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            # action_bias = (high + low) / 2
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    # Defines the forward pass to compute distribution parameters
    def forward(self, state):
        # Pass state through first layer and apply ReLU
        x = F.relu(self.linear1(state))
        # Pass through second layer and apply ReLU
        x = F.relu(self.linear2(x))
        # Compute the mean (μ) using the mean output layer
        mean = self.mean_linear(x)
        # Compute the log standard deviation (log σ) using the log_std output layer
        log_std = self.log_std_linear(x)
        # Clamp the log_std values to be within [LOG_SIG_MIN, LOG_SIG_MAX] for numerical stability
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        # Return the computed mean and log_std
        return mean, log_std

    # Method to sample an action and compute its log probability
    def sample(self, state):
        # Get the mean (μ) and log standard deviation (log σ) from the forward pass
        mean, log_std = self.forward(state)
        # Calculate the standard deviation (σ = exp(log σ))
        std = log_std.exp()
        # Create a Normal (Gaussian) distribution object with the computed mean and std
        normal = Normal(mean, std)
        # Sample an action `x_t` using the reparameterization trick (x_t = μ + σ * ε, where ε ~ N(0, I))
        # This allows gradients to flow back through the sampling process
        x_t = normal.rsample()
        # Apply the tanh squashing function to bound the action sample `x_t` to the range [-1, 1]
        y_t = torch.tanh(x_t)
        # Rescale the squashed action `y_t` to the environment's action range using precomputed scale and bias
        action = y_t * self.action_scale + self.action_bias
        # Calculate the log probability of the *unsquashed* action `x_t` under the Gaussian distribution
        log_prob = normal.log_prob(x_t)
        # --- Enforcing Action Bound Correction ---
        # Correct the log probability due to the tanh transformation.
        # This is derived from the change of variables formula for probability distributions.
        # log π(a|s) = log ρ(u|s) - Σ log(1 - tanh(u_i)²) where a = tanh(u) and u ~ N(μ, σ²)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        # Sum the log probabilities across the action dimensions (if action space is multi-dimensional)
        log_prob = log_prob.sum(1, keepdim=True)
        # Calculate the mean action, also squashed and rescaled (used for evaluation)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # Return the final action, its corrected log probability, and the mean action
        return action, log_prob, mean

    # Override the .to() method to ensure action_scale and action_bias are moved to the correct device
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        # Call the parent class's .to() method to move the network parameters
        return super(GaussianPolicy, self).to(device)


# --- Deterministic Policy Network (Actor: a = π(s)) ---
# Implements the Actor network using a deterministic policy
# Used for comparison or in algorithms like DDPG
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        # Call the constructor of the parent class (nn.Module)
        super(DeterministicPolicy, self).__init__()
        # First linear layer
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        # Second linear layer
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer for the deterministic action (mean)
        self.mean = nn.Linear(hidden_dim, num_actions)
        # Tensor to store noise (used for exploration during training)
        self.noise = torch.Tensor(num_actions)

        # Apply the custom weight initialization
        self.apply(weights_init_)

        # --- Action Rescaling ---
        # Same logic as in GaussianPolicy to map tanh output to environment's action range
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    # Defines the forward pass to compute the deterministic action
    def forward(self, state):
        # Pass state through first layer and apply ReLU
        x = F.relu(self.linear1(state))
        # Pass through second layer and apply ReLU
        x = F.relu(self.linear2(x))
        # Compute the action using the output layer, apply tanh squashing, and rescale
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        # Return the deterministic action
        return mean

    # Method to sample an action (adds noise for exploration during training)
    def sample(self, state):
        # Compute the deterministic mean action using the forward pass
        mean = self.forward(state)
        # Generate noise from a normal distribution (mean 0, std 0.1)
        noise = self.noise.normal_(0., std=0.1)
        # Move noise to the same device as the mean tensor
        noise = noise.to(mean.device)
        # Clamp the noise to a fixed range [-0.25, 0.25]
        noise = noise.clamp(-0.25, 0.25)
        # Add noise to the mean action for exploration
        action = mean + noise
        # Return the noisy action, a dummy log probability (0, as policy is deterministic), and the mean action
        return action, torch.tensor(0.), mean

    # Override the .to() method to also move action_scale, action_bias, and noise tensor to the device
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        # Call the parent class's .to() method
        return super(DeterministicPolicy, self).to(device)