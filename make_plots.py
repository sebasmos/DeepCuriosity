from utils import plot_metrics, compare_rewards_plot, prepare_directories
import hydra
from omegaconf import DictConfig

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    raw_name = "alg-ppobase_env-swimmerv4_arch-mlp256"
    icm_name = "alg-ppoicm_env-swimmerv4_arch-mlp256"
    cfgraw = cfg.copy()
    cfgraw.run_name = raw_name
    cfgicm = cfg.copy()
    cfgicm.run_name = icm_name
    raw_dir, ckpt_dir, log_dir = prepare_directories("raw_pytorch", cfgraw)
    icm_dir, ckpt_dir, log_dir = prepare_directories("ppo_icm", cfgicm)

    print(f"Raw directory: {raw_dir}")
    print(f"ICM directory: {icm_dir}")

    try:
        plot_metrics(raw_dir)
        plot_metrics(icm_dir)

        compare_rewards_plot(raw_dir, icm_dir)
    except Exception as e:
        print(f"Error during plotting: {e}")

if __name__ == "__main__":
    main()