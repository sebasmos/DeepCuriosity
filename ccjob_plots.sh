## salloc --time=1:0:0 --mem-per-cpu=16G --ntasks=1 --account=def-kjerbi

module purge
module load python/3.11.5
module load mujoco/3.3.0

cd /home/yorguin/envs

##virtualenv --no-download $SLURM_TMPDIR/env
##virtualenv --no-download swimmer_env

##source $SLURM_TMPDIR/env/bin/activate
source /home/yorguin/envs/swimmer_env/bin/activate

##pip install --no-index --upgrade pip
cd /home/yorguin/scratch/DeepCuriosity

##pip install --no-index --upgrade pip
##pip install --no-cache-dir --no-index -r requirements_cc.txt
python -u make_plots.py