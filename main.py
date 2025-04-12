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
args = parser.parse_args()

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

# Tensorboard (corrected typo)
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
# Ensure ReplayMemory is compatible with data types and structure
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    # done = False # Old 'done' flag

    # --- CHANGE 2.1: Modify env.reset() call ---
    # state = env.reset() # Old reset call
    # New reset call: pass seed, receive tuple (observation, info)
    # Seeding only the first episode's reset ensures run reproducibility.
    # Seeding every reset makes each episode start identically if run standalone.
    # Choose based on desired reproducibility level. Here, seeding first reset:
    state, info = env.reset(seed=args.seed if i_episode == 1 else None)
    # Alternatively, seed every reset:
    # state, info = env.reset(seed=args.seed)

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

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                # Corrected spelling for temperature
                writer.add_scalar('entropy_temperature/alpha', alpha, updates)
                updates += 1

        # --- CHANGE 2.2: Modify env.step() call ---
        # next_state, reward, done, _ = env.step(action) # Old step call (4 return values)
        # New step call (5 return values)
        next_state, reward, terminated, truncated, info = env.step(action)

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # --- CHANGE 3: Update mask calculation based on 'terminated' flag ---
        # Old mask logic might rely on internal _max_episode_steps or combined 'done'
        # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        # New mask logic: mask is 0 if terminated (final state), 1 otherwise (including truncation)
        mask = float(not terminated)

        # Ensure memory.push expects the correct data types and order
        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        # The loop condition `while not (terminated or truncated):` handles exiting the loop

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    # Evaluation loop
    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for eval_ep in range(episodes): # Use a different loop variable name
            # --- CHANGE 4.1: Modify env.reset() call in eval loop ---
            # state = env.reset() # Old reset call
            # New reset call for evaluation: Seed for consistency, get tuple
            # Seed each evaluation episode reset for consistent eval comparison
            state, info = env.reset(seed=args.seed + eval_ep) # Use different seeds for each eval ep
            # state, info = env.reset(seed=args.seed) # Or use the same seed for all eval eps

            # Reset flags for evaluation episode
            eval_terminated = False
            eval_truncated = False
            episode_reward = 0 # Renamed from eval_episode_reward for clarity

            # Evaluation loop condition
            while not (eval_terminated or eval_truncated):
                # Ensure select_action is compatible
                action = agent.select_action(state, evaluate=True)

                # --- CHANGE 4.2: Modify env.step() call in eval loop ---
                # next_state, reward, done, _ = env.step(action) # Old step call
                # New step call
                next_state, reward, eval_terminated, eval_truncated, info = env.step(action)

                episode_reward += reward
                state = next_state
                # Loop condition handles exit

            avg_reward += episode_reward
        avg_reward /= episodes # Calculate average over evaluation episodes

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()