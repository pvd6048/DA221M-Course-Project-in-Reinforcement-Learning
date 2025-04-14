import os
import torch
import torch.nn.functional as F # Functional interface for neural network modules (like loss functions)
from torch.optim import Adam # Adam optimizer
from utils import soft_update, hard_update # Utility functions for updating target networks
from model import GaussianPolicy, QNetwork, DeterministicPolicy # Neural network models for policy and Q-functions

# Define the Soft Actor-Critic agent class
class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        # --- Hyperparameters ---
        self.gamma = args.gamma # Discount factor (γ) for future rewards
        self.tau = args.tau # Target smoothing coefficient (τ) for soft target updates
        self.alpha = args.alpha # Entropy regularization coefficient (α)

        # --- Configuration ---
        self.policy_type = args.policy # Type of policy network ("Gaussian" or "Deterministic")
        self.target_update_interval = args.target_update_interval # Frequency of target network updates
        self.automatic_entropy_tuning = args.automatic_entropy_tuning # Flag to automatically tune alpha

        # --- Device Setup ---
        self.device = torch.device("cuda" if args.cuda else "cpu") # Set device to GPU or CPU

        # --- Critic Network Initialization ---
        # Initialize the Critic network (QNetwork contains two Q-functions internally)
        # Input: state dim, action dim, hidden layer size. Output: Q-value(s) Q(s, a)
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        # Initialize the optimizer for the Critic network parameters (θ)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        # Initialize the Target Critic network (same architecture as the main critic)
        # Target networks provide stable targets during training (parameters θ')
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        # Initialize target critic weights to be the same as the main critic weights initially (θ' ← θ)
        hard_update(self.critic_target, self.critic)

        # --- Policy Network Initialization (depends on policy_type) ---
        if self.policy_type == "Gaussian": # If using a stochastic Gaussian policy (standard SAC)
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as suggested in the paper
            if self.automatic_entropy_tuning is True: # If alpha should be learned
                # Calculate the target entropy H₀ = -dim(Action Space)
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                # Initialize the log of alpha as a learnable parameter (log α). Using log ensures α stays positive.
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                # Initialize a separate optimizer for log_alpha
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            # Initialize the Actor network (Gaussian Policy πφ(a|s))
            # Input: state dim. Output: parameters (mean, log_std) for Gaussian distribution over actions
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            # Initialize the optimizer for the Actor network parameters (φ)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else: # If using a deterministic policy (e.g., for comparison or DDPG-like variant)
            self.alpha = 0 # No entropy term if policy is deterministic
            self.automatic_entropy_tuning = False # Disable automatic tuning for deterministic policy
            # Initialize the Actor network (Deterministic Policy a = πφ(s))
            # Input: state dim. Output: specific action
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            # Initialize the optimizer for the Actor network parameters (φ)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    # Method to select an action based on the current state
    def select_action(self, state, evaluate=False):
        # Convert state to PyTorch tensor, add batch dimension, and move to device
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False: # During training: sample action for exploration
            # Sample action, log probability, and mean action from the policy πφ(a|s)
            # Uses reparameterization trick internally for Gaussian policy
            action, _, _ = self.policy.sample(state)
        else: # During evaluation: use the mean action for deterministic behavior
            # Sample action, log probability, and mean action from the policy πφ(a|s)
            _, _, action = self.policy.sample(state) # Take the mean action (third return value)
        # Detach action from computation graph, move to CPU, convert to NumPy array, remove batch dim
        return action.detach().cpu().numpy()[0]

    # Method to update network parameters using a batch of experiences
    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch of transitions (s, a, r, s', mask) from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        # Convert batch elements to PyTorch tensors and move to the appropriate device
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1) # Add dimension for broadcasting
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1) # Mask is 1 for non-terminal, 0 for terminal

        # --- Begin Critic Update ---
        # Target computations should not be part of gradient calculation for the main critic
        with torch.no_grad():
            # Sample next actions (a') and their log probabilities (log π(a'|s')) from the *current* policy πφ for the next states (s')
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            # Compute the Q-values for the next state-action pairs (s', a') using the *target* critic networks Qθ'
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            # Apply Clipped Double-Q trick: take the minimum of the two target Q-values
            # Subtract the scaled entropy term α * log π(a'|s') to get the "soft" Q-value target
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # Compute the final target Q-value (y) using the Bellman equation: y = r + γ * mask * (min(Qθ') - α log π)
            # The mask ensures the future value is zero for terminal states
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # Get the Q-values for the current state-action pairs (s, a) from the *main* critic networks Qθ
        # Using two Q-functions helps mitigate positive bias in policy improvement steps
        qf1, qf2 = self.critic(state_batch, action_batch)
        # Calculate the Mean Squared Error loss for the first Q-network: JQ(θ1) = E[(Qθ1(s,a) - y)²]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        # Calculate the Mean Squared Error loss for the second Q-network: JQ(θ2) = E[(Qθ2(s,a) - y)²]
        qf2_loss = F.mse_loss(qf2, next_q_value)
        # Total critic loss is the sum of the two losses
        qf_loss = qf1_loss + qf2_loss

        # --- Perform Critic Optimization Step ---
        self.critic_optim.zero_grad() # Reset gradients before backpropagation
        qf_loss.backward() # Compute gradients of the critic loss w.r.t. critic parameters (θ)
        self.critic_optim.step() # Update critic parameters (θ) using the optimizer

        # --- Begin Policy Update ---
        # Sample actions (a) and log probabilities (log π(a|s)) from the *current* policy πφ for the current states (s)
        # This uses the reparameterization trick, allowing gradients to flow back to the policy network
        pi, log_pi, _ = self.policy.sample(state_batch)

        # Compute the Q-values for these *policy* actions (s, a) using the *main* critic networks Qθ
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        # Take the minimum of the two Q-values for the policy loss calculation (part of Clipped Double-Q)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Calculate the policy loss Jπ(φ) = E [α * log π(a|s) - min(Qθ(s,a))]
        # The goal is to maximize Q-value while maximizing entropy (acting randomly)
        # Minimizing this expression achieves that goal. Expectation E is approximated by mean over the batch.
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # --- Perform Policy Optimization Step ---
        self.policy_optim.zero_grad() # Reset gradients before backpropagation
        policy_loss.backward() # Compute gradients of the policy loss w.r.t. policy parameters (φ)
        self.policy_optim.step() # Update policy parameters (φ) using the optimizer

        # --- Begin Alpha (Entropy Temperature) Update (if automatic tuning is enabled) ---
        if self.automatic_entropy_tuning:
            # Calculate the loss for alpha (log_alpha). J(α) = E [-α * (log π(a|s) + H₀)]
            # This encourages the policy entropy (measured by -log π) to match the target entropy H₀
            # .detach() is crucial to prevent gradients from flowing into the policy network during this step
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            # --- Perform Alpha Optimization Step ---
            self.alpha_optim.zero_grad() # Reset gradients for the alpha optimizer
            alpha_loss.backward() # Compute gradients of the alpha loss w.r.t. log_alpha
            self.alpha_optim.step() # Update log_alpha using its optimizer

            # Update the scalar alpha value used in calculations (α = exp(log α))
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # Store current alpha value for logging (TensorboardX)
        else: # If alpha is fixed
            alpha_loss = torch.tensor(0.).to(self.device) # No loss calculated
            alpha_tlogs = torch.tensor(self.alpha).to(self.device) # Store the fixed alpha value for logging


        # --- Update Target Networks ---
        # Periodically update the target critic network using polyak averaging (soft update)
        if updates % self.target_update_interval == 0:
            # θ' ← τ * θ + (1 - τ) * θ'
            soft_update(self.critic_target, self.critic, self.tau)

        # Return loss values and alpha for monitoring/logging purposes
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # --- Model Saving ---
    # Save model parameters (weights) and optimizer states to a checkpoint file
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        # Save state dictionaries of policy, critic, target critic, and their optimizers
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # --- Model Loading ---
    # Load model parameters and optimizer states from a checkpoint file
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path) # Load the checkpoint dictionary
            # Load the state dictionaries into the respective models and optimizers
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            # Set models to evaluation mode (.eval()) or training mode (.train())
            # This affects layers like Dropout and BatchNorm
            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()