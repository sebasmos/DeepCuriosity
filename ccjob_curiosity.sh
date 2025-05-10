#! /bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=0-03:00:00
#SBATCH --job-name=baseline
#SBATCH --output=%j-baseline.out
#SBATCH --error=%j-baseline.err
#SBATCH --mail-user=yjmantilla@gmail.com
#SBATCH --mail-type=ALL

module load python/3.11.5
module load mujoco/3.3.0
#virtualenv --no-download $SLURM_TMPDIR/env
virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
cd /home/yorguin/scratch/DeepCuriosity
pip install --no-cache-dir --no-index -r requirements_cc.txt
python curiosity.py