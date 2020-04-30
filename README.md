Security_project_adversarial_robust_model_compression
use ADMM pruning and knowledge distillation to compress and adversarial training models


  
CPSC-8570 Spring 2020

Lei Zheng:
Yuxin Cui: Apply adversarial training and knowledge distillation compression on MNIST
Rui Cao:
Huixin Deng:

Team Project: Adversarial Robust Model Compression


## Table of Contents

- [Contents](#contents)
- [Environment](#environment)
- [Usage](#usage)



## Contents:
It's a Python3.5 program. It can compress deep networks(LeNet on MNIST and ResNet on CIFAR10) using ADMM and GAN-TSC algorithm.


## Environment: 
Clemson Palmetto, 1 gpu, 120 gb mem.

Module: Anaconda3/5.1.0 cuDNN/10.0v7.4.2 cuda-toolkit/10.0.130 openblas/0.3.5

Python: 3.6.6

Packages needed: tensorflow-gpu, keras, numpy, jupyter, torch, torchvision, tensorboard, pyyaml, scipy, sklearn, pathlib2

In folder 'env', we give creat_env.sh files to help setting up enviroment. 
    
May need other packages, just run "pip install package_name" if required.



## Usage:
* download the project using Unix command line:
       
        $ git clone https://github.com/zhengleicq/CPSC8570-final_project.git
       
        $ cd GAN_TSC_MNIST



* You can train teacher model using Unix command line:
        
        $ python kd_compression.py --model_type teacher --checkpoint_dir teacher_models/checkpoint_adv_16 --num_steps 5000 --temperature 5


* To pretrain student model, you can using command line:
    
        $ python kd_compression.py --model_type student --checkpoint_dir student_models/checkpoint_adv_8 --num_steps 5000


* You can compress the teacher model to student model using command line:
        
        $ python kd_compression.py --model_type student --checkpoint_dir student_models/checkpoint_adv_8 --load_teacher_from_checkpoint true --load_teacher_checkpoint_dir  teacher_models/checkpoint_adv_16 --num_steps 5000 --temperature 5