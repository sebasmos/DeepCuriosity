#! /bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:05:00
#SBATCH --job-name=example
#SBATCH --output=%j-example.out
#SBATCH --error=%j-example.err
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
python baseline.py --config-name config --config-path configs
python icm.py --config-name config --config-path configs
