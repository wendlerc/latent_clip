#!/usr/bin/bash -x
#SBATCH --account=laion
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --partition=g40x
#SBATCH --job-name=openclip-vitb32-toy-1epoch
#SBATCH --output=logs/openclip-vitb32-toy-1epoch-%j.out

module load cuda/11.8
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export PYTHONWARNINGS="ignore"
export CXX=g++
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(echo $MASTER_ADDR | sed 's/ip-//; s/-/./g')
export MASTER_PORT=12802
echo "$MASTER_ADDR:$MASTER_PORT" > dist_url.txt
source .env2/bin/activate
export PYTHONPATH=src:$PYTHONPATH
export NLTK_DATA=cache

NAME=vitb32
LOGS=/scratch/wendlerc/logs

#  --threads-per-core=1 leads to error:
#  
# srun --cpu_bind=v --accel-bind=gn --threads-per-core=1 python -u src/training/main.py \
srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --train-data="/admin/home-wendlerc/data/toy/00000.tar"  --dataset-type webdataset\
    --train-num-samples=10000 \
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
