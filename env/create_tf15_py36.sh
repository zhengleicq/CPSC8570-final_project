qsub -I -l select=1:ncpus=16:mem=100gb:ngpus=1:gpu_model=p100,walltime=72:00:00
module load cuda-toolkit/9.0.176 cuDNN/9.0v7 anaconda3/5.0.1
conda create -n tf15_py36 pip python=3.6
source activate tf15_py36
pip install tensorflow-gpu==1.15
conda install jupyter
python -m ipykernel install --user --name tf15_py36 --display-name tf15_py36