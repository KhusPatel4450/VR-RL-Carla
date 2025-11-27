"""
Variational Reinforcement Learning (VR-RL) Training Script
==========================================================
This script implements the VR-RL algorithm for autonomous driving in CARLA.

Key Components:
1. CVAE (Conditional Variational Autoencoder): Compresses images into a robust latent representation.
   - Uses an architectural "Bottleneck" to force the model to use the latent variable 'z'.
2. PPO (Proximal Policy Optimization): The RL agent that learns to drive using the CVAE features.
3. Two-Phase Training:
   - Phase 1: Unsupervised Representation Learning (Training the Vision System).
   - Phase 2: Reinforcement Learning (Training the Driving Policy).
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from dataclasses import dataclass
import os

# Import the environment wrapper
from carla_env_wrapper import CarlaEnv

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Hyperparameters for Training"""
    # --- Environment ---
    img_h: int = 80        # Input image height (resized)
    img_w: int = 160       # Input image width (resized)
    channels: int = 3      # RGB = 3 channels
    nav_dim: int = 5       # Number of navigation features (speed, angle, etc.)
    action_dim: int = 2    # Steering [-1, 1], Throttle [0, 1]
    
    # --- CVAE (Vision System) ---
    z_dim: int = 64        # Size of the Latent Vector (Random Noise)
    rep_dim: int = 256     # Size of the Representation (Output to PPO)
    beta_target: float = 1.0 # Weight of KL Divergence term (Regularization)
    cvae_lr: float = 3e-4  # Learning Rate for CVAE
    
    # --- PPO (Driving Agent) ---
    gamma: float = 0.99    # Discount factor (importance of future rewards)
    lam: float = 0.95      # GAE parameter (bias-variance trade-off)
    clip_range: float = 0.2 # PPO clipping (prevents drastic policy changes)
    entropy_coef: float = 0.01 # Initial exploration randomness
    value_coef: float = 0.5  # Weight of Critic loss
    ppo_lr: float = 3e-4   # Learning Rate for PPO
    ppo_epochs: int = 10   # Number of updates per batch
    ppo_batch_size: int = 64
    
    # --- Training Schedule ---
    pretrain_steps: int = 15000 # Steps to train CVAE (Phase 1)
    rl_episodes: int = 2000     # Episodes to train PPO (Phase 2)
    eval_interval: int = 25     # How often to run a Deterministic Test
    buffer_size: int = 50000    # Replay buffer capacity
    batch_size: int = 128       # Batch size for CVAE training
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# ============================================================
# CVAE MODELS
# ============================================================

class Encoder(nn.Module):
    """
    Encodes the raw image into a Latent Distribution (Mean, StdDev).
    Input: Image (3, 80, 160) -> Output: mu (64), logvar (64)
    """
    def __init__(self):
        super().__init__()
        # Standard Convolutional Neural Network
        self.net = nn.Sequential(
            nn.Conv2d(cfg.channels, 32, 4, 2), nn.ReLU(), # Downsample
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),           # Downsample
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),          # Downsample
            nn.Flatten()
        )
        # Calculate the size of the flattened vector dynamically
        with torch.no_grad():
            d = torch.zeros(1, cfg.channels, cfg.img_h, cfg.img_w)
            self.flat_dim = self.net(d).shape[1] # Approx 25,600 features

        # Heads for Mu (Mean) and LogVar (Variance)
        self.fc_mu = nn.Linear(self.flat_dim, cfg.z_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, cfg.z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    """
    The core of VR-RL.
    Takes an Observation (Image) AND a Latent Code (z).
    Outputs a Representation (r) for the Agent, and Reconstructs the Image.
    """
    def __init__(self):
        super().__init__()
        dummy_enc = Encoder()
        self.flat_dim = dummy_enc.flat_dim # 25,600
        
        # 1. Visual Feature Extractor (Deterministic Path)
        self.obs_processor = nn.Sequential(
            nn.Conv2d(cfg.channels, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
            nn.Flatten()
        )
        
        # 2. [CRITICAL] Compression Bottleneck
        # We compress the 25,600 visual features down to 128.
        # Why? If we don't, the massive visual signal drowns out the tiny z signal (64),
        # causing "Posterior Collapse" (KL=0). This forces the model to use z.
        self.compress_obs = nn.Sequential(
            nn.Linear(self.flat_dim, 128),
            nn.ReLU()
        )
        
        # Dropout forces the model to rely on z when visual features are noisy
        self.obs_dropout = nn.Dropout(p=0.5) 
        
        # 3. Fusion Layer (Visuals + Latent)
        # Combines Compressed Visuals (128) + Latent z (64)
        self.fc_combine = nn.Linear(128 + cfg.z_dim, 512)
        
        # 4. Representation Layer (The Output for PPO)
        self.fc_rep = nn.Linear(512, cfg.rep_dim) 
        
        # 5. Reconstruction Head (Only used for CVAE training)
        self.fc_recon = nn.Linear(cfg.rep_dim, self.flat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2, padding=1), nn.ReLU(), 
            nn.ConvTranspose2d(64, 32, 5, 2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, cfg.channels, 6, 2, padding=1), nn.Sigmoid()
        )

    def get_rep(self, obs, z):
        """
        Fast inference method for Phase 2 (RL).
        Does NOT reconstruct the image, only calculates 'r'.
        """
        obs_feat = self.obs_processor(obs)
        obs_feat = self.compress_obs(obs_feat) # Apply bottleneck
        # Note: No dropout during inference!
        h = torch.cat([obs_feat, z], dim=1)
        h = F.relu(self.fc_combine(h))
        r = torch.tanh(self.fc_rep(h))
        return r

    def forward(self, obs, z):
        """Full forward pass for Phase 1 (CVAE Training)."""
        obs_feat = self.obs_processor(obs)
        obs_feat = self.compress_obs(obs_feat)
        
        # Apply dropout during training to force z usage
        if self.training:
            obs_feat = self.obs_dropout(obs_feat)
            
        h = torch.cat([obs_feat, z], dim=1)
        h = F.relu(self.fc_combine(h))
        r = torch.tanh(self.fc_rep(h)) # The Representation
        
        # Reconstruction logic
        recon_h = F.relu(self.fc_recon(r))
        recon_h = recon_h.view(-1, 128, 8, 18) # Reshape to image tensor
        out = self.deconv(recon_h)
        recon_img = F.interpolate(out, size=(cfg.img_h, cfg.img_w), mode='bilinear')
        
        return r, recon_img

class VRRL_Model(nn.Module):
    """Container for Encoder and Decoder"""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward_train(self, x):
        """Standard VAE Forward Pass: Encode -> Sample -> Decode"""
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) # Reparameterization Trick
        r, x_hat = self.decoder(x, z)        # Conditional Decoding
        return x_hat, mu, logvar

# ============================================================
# PPO AGENT
# ============================================================

class ActorCritic(nn.Module):
    """The Driver (Policy) and the Judge (Critic)"""
    def __init__(self):
        super().__init__()
        input_dim = cfg.rep_dim + cfg.nav_dim
        
        # Actor: Decides Steering/Throttle
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, cfg.action_dim)
        )
        # Exploration Noise Parameter
        # Initialized to 0.0 (High Noise) to prevent early stagnation ("Parking")
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))
        
        # Critic: Estimates Value of current state
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )

    def get_action(self, state, deterministic=False):
        """Selects an action based on state"""
        mu = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        
        # Deterministic = Evaluation Mode (Best behavior)
        if deterministic:
            return mu, None, None
            
        # Stochastic = Training Mode (Exploration)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), self.critic(state)

    def evaluate(self, state, action):
        """Evaluates actions for PPO Update"""
        mu = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        return dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1), self.critic(state).squeeze(-1)

# ============================================================
# BUFFER (Experience Replay)
# ============================================================

class Buffer:
    """Stores training data in RAM. Supports Save/Load."""
    def __init__(self):
        self.obs_img = []
        self.obs_nav = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []
        self.ptr = 0
        self.max_size = cfg.buffer_size

    def add(self, img, nav, a, r, d, logp, v):
        # Grow buffer if needed
        if len(self.obs_img) < self.max_size:
            self.obs_img.append(None); self.obs_nav.append(None); self.actions.append(None)
            self.rewards.append(None); self.dones.append(None); self.logprobs.append(None); self.values.append(None)
            
        # Store as uint8 to save massive amounts of RAM (4x savings vs float)
        self.obs_img[self.ptr] = (img * 255).astype(np.uint8)
        self.obs_nav[self.ptr] = nav
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = d
        self.logprobs[self.ptr] = logp
        self.values[self.ptr] = v
        self.ptr = (self.ptr + 1) % self.max_size

    def sample_images(self, batch_size):
        """Randomly samples images for CVAE training"""
        indices = np.random.randint(0, len(self.obs_img), size=batch_size)
        batch = [self.obs_img[i] for i in indices]
        return torch.tensor(np.array(batch), dtype=torch.float32).permute(0, 3, 1, 2).to(cfg.device) / 255.0

    def get_all(self):
        """Returns all data for PPO update"""
        imgs = torch.tensor(np.array(self.obs_img), dtype=torch.float32).permute(0, 3, 1, 2).to(cfg.device) / 255.0
        navs = torch.tensor(np.array(self.obs_nav), dtype=torch.float32).to(cfg.device)
        acts = torch.tensor(np.array(self.actions), dtype=torch.float32).to(cfg.device)
        rews = torch.tensor(np.array(self.rewards), dtype=torch.float32).to(cfg.device)
        dones = torch.tensor(np.array(self.dones), dtype=torch.float32).to(cfg.device)
        logps = torch.tensor(np.array(self.logprobs), dtype=torch.float32).to(cfg.device)
        vals = torch.tensor(np.array(self.values), dtype=torch.float32).to(cfg.device)
        return imgs, navs, acts, rews, dones, logps, vals

    def save(self, path):
        """Saves buffer to disk to skip collection on restart"""
        data = {
            'obs_img': self.obs_img, 'obs_nav': self.obs_nav, 'actions': self.actions,
            'rewards': self.rewards, 'dones': self.dones, 'ptr': self.ptr
        }
        torch.save(data, path)
        print(f"[Buffer] Saved {len(self.obs_img)} steps to {path}")
        
    def load(self, path):
        """Loads buffer from disk"""
        print(f"[Buffer] Loading from {path}...")
        # weights_only=False is required for loading numpy arrays in newer PyTorch
        data = torch.load(path, weights_only=False)
        self.obs_img = data['obs_img']
        self.obs_nav = data['obs_nav']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        self.ptr = data['ptr']
        print(f"[Buffer] Loaded {len(self.obs_img)} steps.")

    def clear(self):
        self.__init__()

# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train():
    

    env = CarlaEnv(visual_display=False)
    
    # Initialize Networks
    vr_model = VRRL_Model().to(cfg.device)
    cvae_opt = torch.optim.Adam(vr_model.parameters(), lr=cfg.cvae_lr)
    
    policy = ActorCritic().to(cfg.device)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=cfg.ppo_lr)
    
    # LR Scheduler: Linearly decays Learning Rate to 0 over training
    # This improves stability in later episodes.
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        policy_opt, start_factor=1.0, end_factor=0.01, total_iters=cfg.rl_episodes
    )
    
    buffer = Buffer()
    writer = SummaryWriter(log_dir="runs/vr_rl_final")
    buffer_path = "buffer_dump.pt"

    # ---------------------------------------------------------
    # PHASE 1: DATA COLLECTION (Once)
    # ---------------------------------------------------------
    if os.path.exists(buffer_path):
        buffer.load(buffer_path)
        print("[Phase 1] Skipped collection (loaded from file).")
    else:
        print("\n[Phase 1] Collecting data (20,000 frames)...")
        obs, _ = env.reset()
        for i in range(20000):
            action = env.action_space.sample()
            next_obs, _, done, _, _ = env.step(action)
            buffer.add(obs[0], obs[1], action, 0, done, 0, 0)
            obs = next_obs
            if done: obs, _ = env.reset()
            if i % 1000 == 0: print(f"  Collected {i} frames...")
        buffer.save(buffer_path)

    # ---------------------------------------------------------
    # PHASE 1: CVAE TRAINING (Representation Learning)
    # ---------------------------------------------------------
    print("[Phase 1] Training CVAE...")
    vr_model.train()
    for i in range(cfg.pretrain_steps):
        img_batch = buffer.sample_images(cfg.batch_size)
        recon_batch, mu, logvar = vr_model.forward_train(img_batch)
        
        # Anneal Beta: Gradually increase regularization to prevent collapse
        beta = min(1.0, i / 5000.0) * cfg.beta_target
        
        # Loss = Reconstruction + Beta * KL Divergence
        recon_loss = F.mse_loss(recon_batch, img_batch, reduction='sum') / cfg.batch_size
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / cfg.batch_size
        loss = recon_loss + beta * kl_loss
        
        cvae_opt.zero_grad()
        loss.backward()
        cvae_opt.step()
        
        if i % 500 == 0:
            print(f"  Step {i} | Loss: {loss.item():.4f} | KL: {kl_loss.item():.4f}")

    print("[Phase 1] Complete. Encoder Discarded.")

    # ---------------------------------------------------------
    # PHASE 2: REINFORCEMENT LEARNING (PPO)
    # ---------------------------------------------------------
    print("\n[Phase 2] Starting Policy Optimization...")
    buffer.clear() # Clear old random data
    
    for p in vr_model.parameters(): p.requires_grad = False
    vr_model.eval() # Freeze CVAE for stable features

    for ep in range(cfg.rl_episodes):
        # Decay Entropy: Reduce randomness as agent gets smarter
        current_ent_coef = max(0.001, cfg.entropy_coef * (1.0 - ep / cfg.rl_episodes))
        
        obs, _ = env.reset()
        ep_reward = 0
        
        # --- Rollout (Data Collection) ---
        while True:
            img = torch.tensor(obs[0]).permute(2,0,1).unsqueeze(0).to(cfg.device).float() / 255.0
            nav = torch.tensor(obs[1]).unsqueeze(0).to(cfg.device).float()
            
            # VR-RL Core: Sample random Z (stochastic augmentation)
            with torch.no_grad():
                z = torch.randn(1, cfg.z_dim, device=cfg.device)
                rep = vr_model.decoder.get_rep(img, z)
            
            state = torch.cat([rep, nav], dim=1)
            action, log_prob, val = policy.get_action(state)
            np_act = action.squeeze().cpu().numpy()
            
            next_obs, reward, done, _, _ = env.step(np_act)
            
            # Scale Reward: Helps PPO learn from small signals
            scaled_reward = reward * 10.0 
            
            buffer.add(obs[0], obs[1], np_act, scaled_reward, done, log_prob.item(), val.item())
            obs = next_obs
            ep_reward += reward
            
            # PPO Standard: Update on fixed batch size or episode end
            if done or len(buffer.obs_img) >= 1024:
                break
        
        # --- PPO Update ---
        if len(buffer.obs_img) >= 1024:
            imgs, navs, acts, rews, dones, old_logps, vals = buffer.get_all()
            
            # Bootstrap value for GAE
            with torch.no_grad():
                last_img = torch.tensor(next_obs[0]).permute(2,0,1).unsqueeze(0).to(cfg.device).float()/255.0
                last_nav = torch.tensor(next_obs[1]).unsqueeze(0).to(cfg.device).float()
                z = torch.randn(1, cfg.z_dim, device=cfg.device)
                last_rep = vr_model.decoder.get_rep(last_img, z)
                last_val = policy.critic(torch.cat([last_rep, last_nav], dim=1)).item()
            
            # Compute GAE (Generalized Advantage Estimation)
            advantages = torch.zeros_like(rews)
            gae = 0
            for t in reversed(range(len(rews))):
                if t == len(rews) - 1:
                    next_val = last_val
                    next_non_terminal = 1.0 - float(done)
                else:
                    next_val = vals[t+1]
                    next_non_terminal = 1.0 - dones[t]
                delta = rews[t] + cfg.gamma * next_val * next_non_terminal - vals[t]
                gae = delta + cfg.gamma * cfg.lam * next_non_terminal * gae
                advantages[t] = gae
            returns = advantages + vals
            
            # SGD Updates (Multiple Epochs)
            for _ in range(cfg.ppo_epochs):
                with torch.no_grad():
                    # Augment data: Re-sample Z for every batch
                    batch_z = torch.randn(imgs.size(0), cfg.z_dim, device=cfg.device)
                    batch_reps = vr_model.decoder.get_rep(imgs, batch_z)
                    batch_states = torch.cat([batch_reps, navs], dim=1)

                new_logps, entropy, new_vals = policy.evaluate(batch_states, acts)
                ratio = torch.exp(new_logps - old_logps)
                
                # PPO Clipped Loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-cfg.clip_range, 1+cfg.clip_range) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_vals, returns)
                
                loss = actor_loss + cfg.value_coef * critic_loss - current_ent_coef * entropy.mean()
                
                policy_opt.zero_grad()
                loss.backward()
                policy_opt.step()
            
            lr_scheduler.step()
            buffer.clear()
            
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"Episode {ep} | Reward: {ep_reward:.2f} | LR: {current_lr:.6f}")
        writer.add_scalar("RL/Reward", ep_reward, ep)
        writer.add_scalar("RL/LearningRate", current_lr, ep)

        # ---------------------------------------------------------
        # EVALUATION LOOP (Deterministic Test)
        # ---------------------------------------------------------
        if ep > 0 and ep % cfg.eval_interval == 0:
            eval_scores = []
            for _ in range(3): 
                e_obs, _ = env.reset()
                e_rew = 0
                while True:
                    e_img = torch.tensor(e_obs[0]).permute(2,0,1).unsqueeze(0).to(cfg.device).float() / 255.0
                    e_nav = torch.tensor(e_obs[1]).unsqueeze(0).to(cfg.device).float()
                    
                    with torch.no_grad():
                        e_z = torch.randn(1, cfg.z_dim, device=cfg.device)
                        e_rep = vr_model.decoder.get_rep(e_img, e_z)
                        e_state = torch.cat([e_rep, e_nav], dim=1)
                        # Deterministic = True (No noise)
                        e_action, _, _ = policy.get_action(e_state, deterministic=True)
                    
                    e_np_act = e_action.squeeze().cpu().numpy()
                    e_next_obs, r, d, _, _ = env.step(e_np_act)
                    e_rew += r
                    e_obs = e_next_obs
                    if d: break
                eval_scores.append(e_rew)
            
            avg_eval = np.mean(eval_scores)
            print(f"  [EVAL] Episode {ep} | Avg Deterministic Reward: {avg_eval:.2f}")
            writer.add_scalar("RL/Eval_Reward", avg_eval, ep)

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Interrupted.")