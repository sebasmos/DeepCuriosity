import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import gymnasium as gym
import imageio
from models import PPOAgent_Raw, PPOAgent_ICM  
import os

def play_model_from_weights(ckpt_path, seed=0, render=True):
    """
    Load a model (baseline or ICM) from checkpoint and play one episode.
    Automatically detects if ICM weights are present.
    """
    import gymnasium as gym
    import imageio
    import torch
    from models import PPOAgent_Raw, PPOAgent_ICM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["config"]

    env = gym.make(cfg["environment"], render_mode="rgb_array" if render else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # Detect model type
    is_icm = any(k.startswith("icm.") for k in checkpoint["model_state_dict"].keys())
    agent_class = PPOAgent_ICM if is_icm else PPOAgent_Raw
    agent = agent_class(obs_dim, act_dim, cfg["actor_sizes"], cfg["critic_sizes"], act_limit).to(device)

    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    print(f"[✓] Loaded model from update {checkpoint.get('update', 'N/A')}, step {checkpoint.get('total_steps', 'N/A')}")

    obs, _ = env.reset(seed=seed)
    frames, score, done = [], 0.0, False
    ep_len = 0
    max_steps = env.spec.max_episode_steps or 1000

    while not done and ep_len < max_steps:
        try:
            if render:
                frames.append(env.render())
        except Exception as e:
            print(f"[⚠️] Skipping rendering due to: {e}")
            render = False

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = agent.act(obs_tensor.squeeze(0))
        obs, r, term, trunc, _ = env.step(action)
        score += r
        ep_len += 1
        done = term or trunc

    env.close()

    vid_path = None
    if frames:
        run_dir = ckpt_path.parent.parent
        vid_path = run_dir / f"video_ckpt_{ckpt_path.stem}_seed{seed}.mp4"
        try:
            imageio.mimsave(vid_path, frames, fps=30, format='FFMPEG')
            print(f"[📹] Video saved to: {vid_path}")
        except Exception as e:
            print(f"[❌] Could not save video: {e}")
            vid_path = None

    print(f"Episode length: {ep_len}")
    print(f"Total score: {score:.2f}")
    return score, vid_path
    
def plot_metrics(run_dir):
    """Plot reward and losses from training logs in a given run directory."""
    run_dir = Path(run_dir)
    log_file = run_dir / "logs" / "metrics.csv"
    rewards_file = run_dir / "episode_rewards.npy"

    plt.figure(figsize=(12, 5))

    # --- Plot Losses ---
    try:
        import pandas as pd
        data = pd.read_csv(log_file)
        steps = data["step"]
        total_loss = data["total_loss"]
        loss_pi = data["loss_pi"]
        loss_v = data["loss_v"]

        plt.subplot(1, 2, 1)
        plt.plot(steps, total_loss, label="Total Loss")
        plt.plot(steps, loss_pi, label="Policy Loss", alpha=0.7)
        plt.plot(steps, loss_v, label="Value Loss", alpha=0.7)
        plt.title("Losses during Training")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
    except Exception as e:
        print(f"[Warning] Could not plot losses: {e}")

    # --- Plot Rewards ---
    try:
        rewards = np.load(rewards_file)
        window = 50
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid") if len(rewards) >= window else rewards

        plt.subplot(1, 2, 2)
        plt.plot(rewards, label="Raw Reward", alpha=0.3)
        if len(rewards) >= window:
            plt.plot(np.arange(window - 1, len(rewards)), moving_avg, label=f"Moving Avg ({window})", color="red")
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
    except Exception as e:
        print(f"[Warning] Could not plot rewards: {e}")

    plt.tight_layout()
    plt.show()


def compare_rewards_plot(raw_dir, icm_dir):
    """Compare reward curves between baseline PPO and PPO + ICM."""
    raw_rewards = np.load(Path(raw_dir) / "episode_rewards.npy")
    icm_rewards = np.load(Path(icm_dir) / "episode_rewards.npy")

    plt.figure(figsize=(10, 5))
    plt.plot(raw_rewards, label="PPO Raw", alpha=0.3)
    plt.plot(np.convolve(raw_rewards, np.ones(50)/50, mode='valid'), label="PPO Raw (MA50)", color='blue')

    plt.plot(icm_rewards, label="PPO + ICM", alpha=0.3)
    plt.plot(np.convolve(icm_rewards, np.ones(50)/50, mode='valid'), label="ICM (MA50)", color='orange')

    plt.title("PPO vs PPO + ICM (Episode Rewards)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()