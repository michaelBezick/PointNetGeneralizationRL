import random
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dm_control import suite
from torch.utils.tensorboard import SummaryWriter

from point_cloud_generator import PointCloudGenerator
import pdb

env = suite.load(domain_name="walker", task_name="walk")
physics = env.physics

point_cloud_generator = PointCloudGenerator(physics)

# Hyperparameters
GAMMA = 0.99  # Discount factor
LR = 3e-4  # Learning rate
BUFFER_SIZE = 100000
BATCH_SIZE = 256
TAU = 0.005  # Target network soft update
ALPHA = 0.2  # Entropy temperature
TOTAL_EPISODES = 1000
MAX_STEPS = 1000
HIDDEN_DIM = 64
LATENT_DIM=64

# TensorBoard Logger
writer = SummaryWriter("runs/sac_walker")

# Load the walker walk environment
env = suite.load(domain_name="walker", task_name="walk")
action_spec = env.action_spec()
obs_spec = env.observation_spec()

obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
action_dim = action_spec.shape[0]
action_min = torch.tensor(action_spec.minimum, dtype=torch.float32)
action_max = torch.tensor(action_spec.maximum, dtype=torch.float32)

class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.h = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
        )

        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh()
        )

    def forward(self, x):

        """I don't think I need to transform because it is the same"""

        """Input size: [b, n, 3]"""

        # transform here

        x = torch.permute(x, (0, 2, 1))  # [b,3,n]

        x = self.h(x)  # x -> [b,64,n]

        # transform here
        x = self.mlp2(x)  # x -> [b,128,n]

        x = torch.max(x, dim=2)  # x -> [b, 128]

        x = self.mlp3(x)

        return x

# Neural Networks
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.mu = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)  # Clamping to avoid NaNs
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self(state)
        normal = torch.distributions.Normal(mu, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        action = torch.tanh(action)  # Squash actions to (-1,1)
        scaled_action = action_min + 0.5 * (action + 1) * (action_max - action_min)
        return scaled_action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.q = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE, frame_stack=3):
        self.buffer = deque(maxlen=max_size)
        self.frame_stack = frame_stack

    def push(self, frames, action, reward, next_frames, done):
        """
        Stores a transition where `frames` contains 3 consecutive observations.
        Each observation consists of 2 depth images from different cameras.
        `frames`: Shape (3, 2, H, W)
        `next_frames`: Shape (3, 2, H, W)
        """
        self.buffer.append((frames, action, reward, next_frames, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        # Unzip batch into separate lists
        frames_seq, actions, rewards, next_frames_seq, dones = zip(*batch)

        # Convert lists to tensors
        states = torch.FloatTensor(np.array(frames_seq))  # Shape: (batch, 3, 2, H, W)
        next_states = torch.FloatTensor(np.array(next_frames_seq))
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# SAC Agent
class SACAgent:
    def __init__(self, latent_dim, action_dim):

        self.encoder = PointNetEncoder(latent_dim)
        self.actor = Actor(latent_dim, action_dim)
        self.critic1 = Critic(latent_dim, action_dim)
        self.critic2 = Critic(latent_dim, action_dim)
        self.target_critic1 = Critic(latent_dim, action_dim)
        self.target_critic2 = Critic(latent_dim, action_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=LR
        )

        self.replay_buffer = ReplayBuffer()

        self.log_alpha = torch.tensor(np.log(ALPHA), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.squeeze(0).numpy()

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            BATCH_SIZE
        )

        # Compute next actions and log probabilities
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - torch.exp(
                self.log_alpha
            ) * next_log_probs.unsqueeze(-1)
            q_target = rewards + (1 - dones) * GAMMA * min_q_next

        # Compute current Q-values
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (torch.exp(self.log_alpha) * log_probs - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (entropy coefficient) update
        alpha_loss = -torch.mean(self.log_alpha * (log_probs + action_dim).detach())
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        # Log losses
        writer.add_scalar("Loss/Critic", critic_loss.item(), len(self.replay_buffer))
        writer.add_scalar("Loss/Actor", actor_loss.item(), len(self.replay_buffer))
        writer.add_scalar("Loss/Alpha", alpha_loss.item(), len(self.replay_buffer))


# Training Loop
agent = SACAgent(LATENT_DIM, action_dim)

# Function to collect a sequence of frames with frame skipping
def get_depth_observation_sequence(env, num_frames=3, frame_skip=1, height=84, width=84, camera_ids=[0, 1], last_action=None):
    """
    Collects a sequence of `num_frames`, each containing 2 depth images (from different cameras).
    Applies frame skipping by repeating the last chosen action.
    
    Returns shape: (num_frames, 2, height, width) -> 3 frames, 2 images per frame (total 6 images).
    """
    frames = []
    for _ in range(num_frames):
        depth_images = []
        for cam_id in camera_ids:
            depth_image = env.physics.render(height, width, camera_id=cam_id, depth=True)
            depth_image = depth_image.astype(np.float32)  # Convert to float32
            depth_image = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX)  # Normalize
            depth_images.append(depth_image)
        frames.append(np.array(depth_images))  # Shape: (2, height, width)

        # Apply frame skipping: Reuse the last chosen action
        for _ in range(frame_skip):
            env.step(last_action)

    return np.array(frames)  # Shape: (3, 2, height, width)

def point_cloud_to_tensor(point_cloud):
    points = np.asarray(point_cloud.points, dtype=np.float32)
    points_tensor = torch.tensor(points)
    return points_tensor

FRAME_SKIP = 1  # Repeat each action for 1 extra frame

for episode in range(TOTAL_EPISODES):
    timestep = env.reset()
    last_action = np.zeros(action_dim)  # Initialize with a neutral action
    frames = get_depth_observation_sequence(env, num_frames=3, frame_skip=FRAME_SKIP, last_action=last_action)  # Shape: (3, 2, H, W)
    total_reward = 0

    for step in range(MAX_STEPS):
        # Encode 3-frame sequence (2 depth images per frame) into a latent vector
        breakpoint()
        if len(np.shape(frames)) == 4:
            frames = np.expand_dims(frames, axis=0)
        point_clouds = point_cloud_generator.convertDepthImagesToPointcloud(frames, num_cameras=2, num_frames=3)

        latent_vector = agent.encoder(point_clouds)  # Shape: (latent_dim)

        action = agent.select_action(latent_vector)
        
        # Execute the chosen action for `frame_skip` steps
        timestep = env.step(action)

        next_frames = get_depth_observation_sequence(env, num_frames=3, frame_skip=FRAME_SKIP, last_action=action)  # Shape: (3, 2, H, W)

        reward = timestep.reward
        done = timestep.last()

        agent.replay_buffer.push(frames, action, reward, next_frames, done)
        agent.update()

        frames = next_frames
        last_action = action  # Store the last action for frame skipping
        total_reward += reward

        if done:
            break

    writer.add_scalar("Reward/Episode", total_reward, episode)
    print(f"Episode {episode}, Reward: {total_reward}")

