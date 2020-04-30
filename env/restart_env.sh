qsub -I -l select=1:ncpus=16:mem=100gb:ngpus=1:gpu_model=p100,walltime=72:00:00
module load cuda-toolkit/9.0.176 cuDNN/9.0v7 anaconda3/5.0.1
source activate 8570_env
cd /home/yuxinc/8570/team_project/Robustness-Aware-Pruning-ADMM