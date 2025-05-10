#! /bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --job-name=curiosity
#SBATCH --output=%j-curiosity.out
#SBATCH --error=%j-curiosity.err
#SBATCH --mail-user=yjmantilla@gmail.com
#SBATCH --mail-type=ALL

module load python/3.11.5
module load mujoco/3.3.0
## virtualenv --no-download $SLURM_TMPDIR/env
virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
cd /home/yorguin/scratch/DeepCuriosity
pip install --no-cache-dir --no-index -r requirements_cc.txt

## TO use
## sbatch --export=MODE=curiosity --job-name=curiosity ccjob_train.sh
## sbatch --export=MODE=baseline --job-name=baseline ccjob_train.sh

# Dispatch based on mode
if [ "$MODE" == "baseline" ]; then
    python -u baseline.py run_name=alg-ppobase_env-swimmerv4_arch-mlp256
elif [ "$MODE" == "curiosity" ]; then
    python -u curiosity.py run_name=alg-ppoicm_env-swimmerv4_arch-mlp256
else
    echo "Invalid MODE specified. Use MODE=baseline or MODE=curiosity"
    exit 1
fi