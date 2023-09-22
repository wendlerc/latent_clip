#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --job-name=openclip-vitb32-laion400m-1epoch
ml purge
source /p/project/ccstdl/laion/miniconda/bin/activate laion
source .env/bin/activate
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR
export PYTHONPATH="$PYTHONPATH:$PWD/src"
LOGS=/p/scratch/ccstdl/wendler3/logs
NAME=vitb32
srun --cpu_bind=v --accel-bind=gn --threads-per-core=1 python -u src/training/main.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --train-data="/p/fastdata/mmlaion/laion-400m/LAION-400m-webdataset/data/{00000..41455}.tar"  --dataset-type webdataset\
    --train-num-samples=40733208 \
    --dataset-resampled \
    --warmup 2000 \
    --batch-size=896 \
    --report-to=tensorboard \
    --epochs=10 \
    --workers=8 \
    --model ViT-B-32 \
    --name $NAME \
    --logs $LOGS \
    --seed 1337 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --lr 0.001 \
    --save-most-recent \
    --grad-checkpoint \
    --resume latest
