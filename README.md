[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sebasmos/DeepCuriosity/blob/main/LICENSE)

# 🧠 DeepCuriosity

**Simple PyTorch implementation of PPO and Intrinsic Curiosity Module (ICM)** for fast training and visualization in [MuJoCo](https://github.com/google-deepmind/mujoco) environments such as `Swimmer-v4`.

---

## 🔧 What’s Inside

- ✅ **PPO baseline** (MLP actor-critic)
- 🔍 **PPO + ICM** (intrinsic curiosity)
- 📊 CSV logs & reward plots
- 🎞️ Automatic episode-video rendering
- 🗂️ Clean directory layout → everything lands under `data/`

---

## 📦 Installation

```bash
# create and activate environment
conda create -n ppo_icm python=3.11 -y
conda activate ppo_icm

# install dependencies
pip install -r requirements.txt
```

*`requirements.txt` already pins the exact versions we tested (Gymnasium 1.1.1, Torch 2.5.1, NumPy 2.2.5, …).*  
*If you prefer a one-liner:*
```bash
pip install torch==2.5.1 gymnasium==1.1.1 mujoco glfw \
            matplotlib==3.10.1 imageio==2.33.1 imageio[ffmpeg] \
            hydra-core==1.3.2 pyyaml pandas==2.2.3 numpy==2.2.5 ipython==8.12.3
```

---

## 🚀 How to Run

### 1. Train PPO baseline
```bash
python baseline.py                # uses configs/config.yaml
```

### 2. Train PPO + ICM (curiosity)
```bash
python curiosity.py               # same config file
```

*(Both scripts rely on Hydra; you can override parameters on the CLI, e.g. `python baseline.py steps=1_000_000`)*

---

## 📈 Visualise Results

Open **`plot.ipynb`** and run the cells to

- plot reward curves for both agents,
- compare moving averages,
- embed rendered videos.

---

## 🗂️ Project & Output Structure

```bash
RL/
├── baseline.py          # PPO training script
├── curiosity.py         # PPO + ICM training script
├── utils.py             # logging, plotting, video helpers
├── models.py            # agent & ICM networks
├── buffer.py            # PPO buffer with GAE
├── plot.ipynb           # notebook for analysis/visuals
├── configs/config.yaml  # shared Hydra config
└── data/                # ← all artefacts saved here
    ├── raw_pytorch/     #   baseline runs
    │   └── Swimmer-v4/
    │       └── mlp_256_raw_pytorch_<timestamp>/
    │           ├── checkpoints/   # model_update_<n>_steps_<k>.pt
    │           ├── logs/          # metrics.csv per update
    │           ├── episode_rewards.npy
    │           └── video_*.mp4    # optional roll-out videos
    └── ppo_icm/        #   curiosity runs
        └── Swimmer-v4/
            └── mlp_256_icm_pytorch_<timestamp>/
                ├── checkpoints/
                ├── logs/
                ├── episode_rewards.npy
                └── video_*.mp4
```

---

## 🤝 Contribute / Contact

Feel free to **fork, open issues or send PRs**.  
Questions or ideas? → [@sebasmos](https://github.com/sebasmos)

---

## ⚡ To-Do / Ideas

- [ ] Extra curiosity bonuses (RND, RIDE, Disagreement)
- [ ] Support more MuJoCo tasks (Hopper, HalfCheetah, Ant…)
- [ ] TensorBoard & Weights-and-Biases logging
- [ ] CLI policy-evaluation script
- [ ] dm_control wrapper to play with custom joint counts

---

## 📚 References

* **Curiosity-Driven Exploration** – <https://arxiv.org/abs/2109.08603>  
* **Artificial Curiosity since 1990** – <https://people.idsia.ch/~juergen/artificial-curiosity-since-1990.html>
