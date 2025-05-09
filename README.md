[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sebasmos/RL-Curiosity-is-All-You-Need/blob/main/LICENSE)

# 🧐 DeepCuriosity

**Simple PyTorch implementation of PPO and Intrinsic Curiosity Module (ICM)** for fast training and visualization in [MuJoCo](https://github.com/google-deepmind/mujoco) environments like `Swimmer-v4`.

> ✅ **Quick demo?** Open [Demo.ipynb](https://github.com/sebasmos/DeepCuriosity/blob/main/Demo.ipynb) to try it out interactively.

---

## 🔧 What’s Inside

- ✅ PPO baseline (MLP actor-critic)
- 🔍 PPO + Curiosity (ICM)
- 📊 Logs, reward plots, and training metrics
- 🎞️ Video generation of trained agents
- 🗂️ Organized folder structure for easy experiments

---

## 📦 Installation

```bash
conda create -n ppo_icm python=3.11.11 -y
conda activate ppo_icm

pip install -r requirements.txt
```

---

## 🚀 How to Run

### Train PPO:

```bash
python baseline.py
```

### Train PPO + ICM:

```bash
python curiosity.py 
```

---

## 📈 Visualize Results

Use the `plot.ipynb` notebook to:

- Compare PPO vs PPO+ICM
- Plot reward curves
- Render videos from saved episodes

---

## 🗂️ Project & Output Structure

```bash
RL/
├── baseline.py         # PPO training
├── icm.py              # PPO + ICM training
├── utils.py            # Video, plotting, loading helpers
├── models.py           # Agents and ICM
├── buffer.py           # PPO buffer
├── plot.ipynb          # Notebook to view results
├── configs/config.yaml # Training settings (Hydra)
└── data/               # <== ALL training artefacts live here
    ├── raw_pytorch/    # Baseline PPO runs
    │   └── <env>/                 # e.g. Swimmer-v4
    │       └── <run>_<timestamp>/ # e.g. mlp_256_raw_pytorch_20250501_101010
    │           ├── checkpoints/          # model_update_<n>_steps_<k>.pt
    │           ├── logs/                 # metrics.csv per update
    │           ├── episode_rewards.npy   # per-episode returns
    │           └── video_*.mp4           # optional rendered roll-outs
    └── ppo_icm/       # PPO + ICM runs
        └── <env>/
            └── <run>_<timestamp>/        # same layout as above
                ├── checkpoints/
                ├── logs/
                ├── episode_rewards.npy
                └── video_*.mp4
```

---

## 🤝 Contribute or Reach Out

Feel free to fork, open issues, or submit PRs.  
If you have questions or want to collaborate, [reach out to me](https://github.com/sebasmos) — happy to chat!

---

### ⚡ Missing / To-Do

- [ ] Add new metrics
- [ ] dm_control for adding more joints
- [ ] Optional curiosity bonuses (RND, Disagreement, etc.)
- [ ] Add support for more MuJoCo environments (e.g. Hopper, HalfCheetah)
- [ ] Add TensorBoard logging
- [ ] Add WandB support
- [ ] Include policy evaluation script
- [ ] Add support for model saving/loading via CLI

---

### 📚 References

- Inspired by [Curiosity-Driven Exploration](https://arxiv.org/abs/2109.08603)
- Historical foundation: [Artificial Curiosity since 1990](https://people.idsia.ch/~juergen/artificial-curiosity-since-1990.html)
