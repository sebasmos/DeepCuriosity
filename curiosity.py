# curiosity.py  – train PPO + ICM, saves in data/ppo_icm/…

import hydra
from omegaconf import DictConfig

from utils   import set_seed, prepare_directories, save_checkpoint, log_metrics
from models  import PPOAgent_ICM
from buffer  import PPOBuffer

import gymnasium as gym
import torch, numpy as np, csv, time
from torch.utils.data import DataLoader, TensorDataset


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def train(cfg: DictConfig):
    # ───── set-up ────────────────────────────────────────────────────────────────
    set_seed(cfg.seed)
    env        = gym.make(cfg.environment, render_mode="rgb_array")
    obs_dim    = env.observation_space.shape[0]
    act_dim    = env.action_space.shape[0]
    act_limit  = env.action_space.high[0]

    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.device = cfg.device.lower()
        if cfg.device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {cfg.device}. Use 'cpu' or 'cuda'.")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use 'cpu' instead.")
    print(f"Using device: {cfg.device}")

    agent      = PPOAgent_ICM(obs_dim, act_dim,
                              cfg.actor_sizes, cfg.critic_sizes,
                              act_limit).to(cfg.device)
    optimizer  = torch.optim.Adam(
        list(agent.parameters()) + list(agent.icm.parameters()), lr=cfg.lr
    )
    buffer     = PPOBuffer(obs_dim, act_dim, cfg.rollout_steps,
                           cfg.gamma, device=cfg.device)

    run_dir, ckpt_dir, log_dir = prepare_directories("ppo_icm", cfg)

    # ───── csv logger ───────────────────────────────────────────────────────────
    log_f = open(log_dir / "metrics.csv", "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(
        ["update", "step", "avg_ep_reward", "loss_pi", "loss_v", "total_loss"]
    )

    # ───── training loop ────────────────────────────────────────────────────────
    obs, _          = env.reset(seed=cfg.seed)
    ep_ret, ep_len  = 0.0, 0
    total_steps     = 0
    update_num      = 0
    all_rewards     = []
    start_time      = time.time()

    while total_steps < cfg.steps:
        # ─ rollouts ────────────────────────────────────────────────────────────
        agent.eval()
        for t in range(cfg.rollout_steps):
            obs_t   = torch.as_tensor(obs, dtype=torch.float32,
                                      device=cfg.device).unsqueeze(0)
            value   = agent.get_value(obs_t).item()
            action, logp = agent.act(obs_t.squeeze(0))

            nxt_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_steps += 1
            ep_ret  += reward
            ep_len  += 1

            # intrinsic reward
            act_t    = torch.as_tensor(action, dtype=torch.float32,
                                       device=cfg.device).unsqueeze(0)
            nxt_t    = torch.as_tensor(nxt_obs, dtype=torch.float32,
                                       device=cfg.device).unsqueeze(0)
            i_rew    = agent.icm.compute_intrinsic_reward(obs_t, nxt_t, act_t).item()
            buffer.store(obs, action, reward + i_rew, value, logp)

            obs = nxt_obs

            terminal   = done or ep_len == env.spec.max_episode_steps
            epoch_done = t == cfg.rollout_steps - 1
            if terminal or epoch_done:
                last_val = 0.0 if done else agent.get_value(
                    torch.as_tensor(obs, dtype=torch.float32,
                                    device=cfg.device).unsqueeze(0)).item()
                buffer.finish_path(last_val)

                if terminal:
                    all_rewards.append(ep_ret)
                    obs, _ = env.reset(seed=cfg.seed)
                    ep_ret, ep_len = 0.0, 0

        # ─ updates ─────────────────────────────────────────────────────────────
        agent.train()
        data      = buffer.get()
        pi_loss_m = v_loss_m = 0.0

        ds  = TensorDataset(data["obs"], data["act"], data["adv"],
                            data["ret"], data["logp"])
        dl  = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

        for _ in range(cfg.epochs_per_update):
            for ob, ac, adv, ret, logp_old in dl:
                optimizer.zero_grad()
                dist      = agent(ob)
                logp_new  = dist.log_prob(ac).sum(-1)
                values    = agent.get_value(ob)

                ratio     = torch.exp(logp_new - logp_old)
                clip_adv  = torch.clamp(ratio,
                                        1 - cfg.clip_param,
                                        1 + cfg.clip_param) * adv
                loss_pi   = -(torch.min(ratio * adv, clip_adv)).mean()
                loss_v    = ((values - ret) ** 2).mean()

                _, pred_nxt, tgt_nxt = agent.icm(ob, ob, ac)
                loss_icm = 0.5 * ((pred_nxt - tgt_nxt) ** 2).sum(-1).mean()

                loss = loss_pi + cfg.value_loss_coef * loss_v + 0.01 * loss_icm
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pi_loss_m += loss_pi.item()
                v_loss_m  += loss_v.item()

        update_num += 1
        nb = len(dl) * cfg.epochs_per_update
        pi_loss_m /= nb
        v_loss_m  /= nb
        tot_loss   = pi_loss_m + cfg.value_loss_coef * v_loss_m

        log_metrics(update_num, total_steps, all_rewards,
                    pi_loss_m, v_loss_m, tot_loss, start_time, writer)

        if update_num % cfg.save_interval == 0 or total_steps >= cfg.steps:
            save_checkpoint(ckpt_dir, update_num, total_steps,
                            agent, optimizer, cfg, all_rewards)

    # ─ clean-up ────────────────────────────────────────────────────────────────
    env.close()
    log_f.close()
    np.save(run_dir / "episode_rewards.npy", np.array(all_rewards))
    print(f"ICM training complete. Results in: {run_dir}")


if __name__ == "__main__":
    train()