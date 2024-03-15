#!/bin/bash

#SBATCH --job-name=unida_basline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=28G
#SBATCH --partition batch_grad
#SBATCH -o train_out_1227/MIC_naive_case0_%j.out
#SBATCH --time=7-0


#source /data/opt/anaconda3/bin/conda init
source /data/dragoon0905/init.sh
conda activate hrda2

python run_experiments.py --config configs/mic/pascalHR2csHR_mic_hrda.py


#python run_experiments.py --config configs/hrda/gtaHR2csHR_hrda.py
#python run_experiments.py --config configs/hrda/pascalHR2csHR_hrda.py

#python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
#python run_experiments.py --config configs/daformer/pascal2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py

#python tools/convert_datasets/gta_11.py data/gta --nproc 8
#python tools/convert_datasets/cityscapes_16.py data/cityscapes --nproc 8
#python tools/convert_datasets/cityscapes_12.py data/cityscapes --nproc 8
#python tools/convert_datasets/pascal_context.py data/gta --nproc 8