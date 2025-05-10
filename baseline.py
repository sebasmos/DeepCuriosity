# baseline.py
import hydra
from omegaconf import DictConfig
from utils import set_seed, save_checkpoint, log_metrics, prepare_directories,get_swimmer_xml
from models import PPOAgent_Raw
from buffer import PPOBuffer
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time, csv
import gymnasium as gym

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def train(cfg: DictConfig):
    set_seed(cfg.seed)
    xml_file = get_swimmer_xml(cfg)
    env = gym.make(cfg.environment, render_mode='rgb_array', xml_file=xml_file.as_posix())
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.device = cfg.device.lower()
        if cfg.device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {cfg.device}. Use 'cpu' or 'cuda'.")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use 'cpu' instead.")
    print(f"Using device: {cfg.device}")

    agent = PPOAgent_Raw(obs_dim, act_dim, cfg.actor_sizes, cfg.critic_sizes, act_limit).to(cfg.device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    buffer = PPOBuffer(obs_dim, act_dim, cfg.rollout_steps, cfg.gamma, device=cfg.device)

    run_dir, ckpt_dir, log_dir = prepare_directories("raw_pytorch", cfg)
    log_file_handle = open(log_dir / "metrics.csv", 'w', newline='')
    log_writer = csv.writer(log_file_handle)
    log_writer.writerow(['update', 'step', 'avg_ep_reward', 'loss_pi', 'loss_v', 'total_loss'])

    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, ep_len = 0.0, 0
    total_steps = 0
    update_num = 0
    all_episode_rewards = []
    start_time = time.time()

    while total_steps < cfg.steps:
        agent.eval()
        for t in range(cfg.rollout_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            value = agent.get_value(obs_tensor).item()
            action, logp = agent.act(obs_tensor.squeeze(0))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            ep_len += 1
            total_steps += 1

            buffer.store(obs, action, reward, value, logp)
            obs = next_obs

            terminal = done or ep_len == env.spec.max_episode_steps
            epoch_ended = t == cfg.rollout_steps - 1

            if terminal or epoch_ended:
                last_val = agent.get_value(torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)).item() if not done else 0.0
                buffer.finish_path(last_val)
                if terminal:
                    all_episode_rewards.append(ep_ret)
                    obs, _ = env.reset(seed=cfg.seed)
                    ep_ret, ep_len = 0.0, 0

        agent.train()
        data = buffer.get()
        avg_pi_loss, avg_v_loss = 0.0, 0.0

        dataset = TensorDataset(data['obs'], data['act'], data['adv'], data['ret'], data['logp'])
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        for _ in range(cfg.epochs_per_update):
            for batch_obs, batch_act, batch_adv, batch_ret, batch_logp_old in dataloader:
                optimizer.zero_grad()
                dist = agent(batch_obs)
                logp_new = dist.log_prob(batch_act).sum(axis=-1)
                values_new = agent.get_value(batch_obs)

                ratio = torch.exp(logp_new - batch_logp_old)
                clip_adv = torch.clamp(ratio, 1 - cfg.clip_param, 1 + cfg.clip_param) * batch_adv
                loss_pi = -(torch.min(ratio * batch_adv, clip_adv)).mean()
                loss_v = ((values_new - batch_ret) ** 2).mean()

                loss = loss_pi + cfg.value_loss_coef * loss_v
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                avg_pi_loss += loss_pi.item()
                avg_v_loss += loss_v.item()

        update_num += 1
        num_batches = len(dataloader) * cfg.epochs_per_update
        avg_pi_loss /= num_batches
        avg_v_loss /= num_batches
        avg_total_loss = avg_pi_loss + cfg.value_loss_coef * avg_v_loss

        log_metrics(update_num, total_steps, all_episode_rewards, avg_pi_loss, avg_v_loss, avg_total_loss, start_time, log_writer)

        if update_num % cfg.save_interval == 0 or total_steps >= cfg.steps:
            save_checkpoint(ckpt_dir, update_num, total_steps, agent, optimizer, cfg, all_episode_rewards)

    env.close()
    log_file_handle.close()
    np.save(run_dir / "episode_rewards.npy", np.array(all_episode_rewards))
    print(f"Training complete. Results in: {run_dir}")


if __name__ == "__main__":
    train()