import argparse
import datetime
# Recommended: Use Gymnasium, the maintained fork of Gym
import gymnasium as gym
# If you must use Gym, ensure it's version 0.26+
# import gym
import numpy as np
import itertools
import torch
from sac import SAC  # Assuming sac.py is in the same directory and compatible
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory # Assuming replay_memory.py is compatible

# --- Add Imports for Logging, Saving, and Plotting ---
import pandas as pd
import matplotlib.pyplot as plt
import os
# ---

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# Updated default environment name to a v4 version
parser.add_argument('--env-name', default="Humanoid-v4", # Example v4 environment
                    help='Mujoco Gym environment (default: Humanoid-v4)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
# --- Add argument for output directory ---
parser.add_argument('--output_dir', default='results',
                    help='Directory to save results and plots (default: results)')

args = parser.parse_args()

# --- Create Output Directory ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"sac_{args.env_name}_seed{args.seed}_{timestamp}"
output_path = os.path.join(args.output_dir, run_name)
os.makedirs(output_path, exist_ok=True)
print(f"Saving results to: {output_path}")
# ---

# Environment
# Ensure any wrappers used (like NormalizedActions if uncommented) are compatible with Gymnasium API
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)

# --- CHANGE 1: Remove old env.seed() call ---
# env.seed(args.seed) # This line is removed

# --- Seed action space (still valid and recommended) ---
env.action_space.seed(args.seed)

# --- Seed libraries ---
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
# Ensure SAC class is compatible with Gymnasium's observation/action spaces if using Gymnasium
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Tensorboard (corrected typo) - Save TensorBoard logs within the run directory
writer = SummaryWriter(log_dir=os.path.join(output_path, 'tensorboard'))

# Memory
# Ensure ReplayMemory is compatible with data types and structure
memory = ReplayMemory(args.replay_size, args.seed)

# --- Initialize Lists to Store Log Data ---
train_rewards_log = []
train_steps_log = []
test_rewards_log = []
test_steps_log = []
# ---

# Training Loop
total_numsteps = 0
updates = 0

print("Starting training...")
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    # done = False # Old 'done' flag

    # --- CHANGE 2.1: Modify env.reset() call ---
    # state = env.reset() # Old reset call
    # New reset call: pass seed, receive tuple (observation, info)
    # Seeding only the first episode's reset ensures run reproducibility.
    state, info = env.reset(seed=args.seed if i_episode == 1 else None)

    # Use terminated/truncated flags from Gymnasium API
    terminated = False
    truncated = False

    # Loop condition uses combined terminated or truncated
    while not (terminated or truncated):
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                # Ensure agent.update_parameters is compatible
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                # Log losses to TensorBoard
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                # Corrected spelling for temperature
                writer.add_scalar('entropy_temperature/alpha', alpha, updates)
                updates += 1

        # --- CHANGE 2.2: Modify env.step() call ---
        next_state, reward, terminated, truncated, info = env.step(action)

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # --- CHANGE 3: Update mask calculation based on 'terminated' flag ---
        mask = float(not terminated)

        # Ensure memory.push expects the correct data types and order
        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        # The loop condition `while not (terminated or truncated):` handles exiting the loop

    # --- Log Training Episode Results ---
    train_rewards_log.append(episode_reward)
    train_steps_log.append(total_numsteps)
    writer.add_scalar('reward/train', episode_reward, i_episode) # Log to TensorBoard as well
    # Reduce console output frequency (optional)
    if i_episode % 100 == 0: # Print every 100 episodes
         print(f"Episode: {i_episode}, Total Numsteps: {total_numsteps}, Episode Steps: {episode_steps}, Training Reward: {episode_reward:.2f}")
    # ---

    # Check termination condition
    if total_numsteps > args.num_steps:
        print(f"Reached maximum number of steps: {args.num_steps}. Stopping training.")
        break

    # Evaluation loop
    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        eval_episodes = 10 # Use a distinct variable name
        for eval_ep in range(eval_episodes):
            # --- CHANGE 4.1: Modify env.reset() call in eval loop ---
            state, info = env.reset(seed=args.seed + eval_ep) # Seed each eval episode differently

            eval_terminated = False
            eval_truncated = False
            eval_episode_reward = 0 # Use a distinct variable name

            # Evaluation loop condition
            while not (eval_terminated or eval_truncated):
                # Ensure select_action is compatible
                action = agent.select_action(state, evaluate=True)

                # --- CHANGE 4.2: Modify env.step() call in eval loop ---
                next_state, reward, eval_terminated, eval_truncated, info = env.step(action)

                eval_episode_reward += reward
                state = next_state

            avg_reward += eval_episode_reward
        avg_reward /= eval_episodes # Calculate average over evaluation episodes

        # --- Log Test Results ---
        test_rewards_log.append(avg_reward)
        test_steps_log.append(total_numsteps)
        writer.add_scalar('avg_reward/test', avg_reward, i_episode) # Log to TensorBoard
        print("----------------------------------------")
        print(f"Test Episodes: {eval_episodes}, Avg. Reward: {avg_reward:.2f} at Step: {total_numsteps}")
        print("----------------------------------------")
        # ---

env.close()
writer.close() # Close the TensorBoard writer
print("Training finished.")

# --- Saving Test Results to CSV ---
print("Saving test results...")
results_df = pd.DataFrame({
    'timesteps': test_steps_log,
    'average_reward': test_rewards_log
})
csv_filename = os.path.join(output_path, "test_results.csv")
results_df.to_csv(csv_filename, index=False)
print(f"Saved test results to {csv_filename}")

# --- Optionally save full training log ---
# print("Saving training log...")
# train_df = pd.DataFrame({
#     'timesteps': train_steps_log,
#     'episode_reward': train_rewards_log
# })
# train_csv_filename = os.path.join(output_path, "train_log.csv")
# train_df.to_csv(train_csv_filename, index=False)
# print(f"Saved training log to {train_csv_filename}")
# ---

# --- Plotting Results ---
print("Generating plots...")
plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style

fig, ax = plt.subplots(1, 2, figsize=(16, 6)) # Create 2 subplots

# Plot Training Rewards (Optional: Plot rolling average for smoother curve)
# Calculate rolling average
train_rewards_series = pd.Series(train_rewards_log)
rolling_window = 50 # Adjust window size as needed
train_rewards_rolling = train_rewards_series.rolling(window=rolling_window, min_periods=1).mean()

# ax[0].plot(train_steps_log, train_rewards_log, alpha=0.3, label='Episode Reward') # Raw data
ax[0].plot(train_steps_log, train_rewards_rolling, label=f'Training Reward (Rolling Avg {rolling_window})')
ax[0].set_xlabel("Total Timesteps")
ax[0].set_ylabel("Reward")
ax[0].set_title("Training Episode Rewards")
ax[0].legend()
ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # Use scientific notation for x-axis if numbers are large

# Plot Test Rewards
ax[1].plot(test_steps_log, test_rewards_log, marker='o', linestyle='-', label='Avg Test Reward')
ax[1].set_xlabel("Total Timesteps")
ax[1].set_ylabel("Average Reward")
ax[1].set_title(f"Average Test Reward ({eval_episodes} Episodes)")
ax[1].legend()
ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # Use scientific notation for x-axis

plt.suptitle(f'SAC Training Progress - {args.env_name} (Seed: {args.seed})', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

# Save the plot
plot_filename = os.path.join(output_path, "training_plot.png")
plt.savefig(plot_filename)
print(f"Saved plot to {plot_filename}")

# Display the plot (optional, comment out if running in non-interactive environment)
# plt.show()

print("Script finished.")