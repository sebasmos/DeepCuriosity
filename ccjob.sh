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
#virtualenv --no-download $SLURM_TMPDIR/env
virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
cd /home/yorguin/scratch/DeepCuriosity
##pip install --no-index -r requirements.txt
pip install -r requirements_cc.txt
python baseline.py --config configs/config.yaml --variant default
python icm.py --config configs/config.yaml --variant icm