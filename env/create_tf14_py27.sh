qsub -I -l select=1:ncpus=16:mem=100gb:ngpus=1:gpu_model=v100,walltime=72:00:00
module load cuda-toolkit/9.0.176 cuDNN/9.0v7 anaconda3/5.0.1
conda create -n py27_env pip python=2.7
source activate py27_env
pip install -r requirements.txt
conda install jupyter
python -m ipykernel install --user --name tf14_py27 --display-name tf14_py27