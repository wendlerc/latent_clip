#!/usr/bin/bash -x
#SBATCH --account=laion
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --partition=g40x
#SBATCH --job-name=debug-latentclip-decoderlayers
#SBATCH --output=logs/debug/latentclip-plus-decoding-%j.out
#SBATCH --error=logs/debug/latentclip-plus-decoding-%j.err
#SBATCH --exclusive
##SBATCH --exclude=ip-26-0-157-217,ip-26-0-144-140,ip-26-0-147-120,ip-26-0-149-101


# Check for command-line arguments
if [ -z "$1" ]; then
    echo "Model/config name required."
    exit 1
fi

if [ -z "$2" ]; then
    echo "wandb run name required."
    exit 1
fi

MODEL=$1
NAME=$2
#LOGS=/scratch/wendlerc/logs
LOGS=/admin/home-wendlerc/models

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
# this is apparently breaking things and without it it seems to work: 
#export MASTER_ADDR=$(echo $MASTER_ADDR | sed 's/ip-//; s/-/./g')
export MASTER_PORT=12802
echo "$MASTER_ADDR:$MASTER_PORT" > dist_url.txt
cat ~/.aws/config > awsconfig.txt
source .env2/bin/activate
export PYTHONPATH=src:$PYTHONPATH
export NLTK_DATA=cache

## 3584 is too large

srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --report-to "tensorboard" \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --train-data="pipe:aws s3 cp s3://stability-west/laioco-latents-split-full/{000..500}/{000000..000100}.tar -" \
    --dataset-type webdataset-latent \
    --dataset-resampled \
    --train-num-samples=100000000 \
    --warmup 100 \
    --batch-size=3072 \
    --epochs=1 \
    --workers=12 \
    --model $MODEL \
    --name $NAME \
    --logs $LOGS \
    --seed 1337 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --lr 0.001 \
    --beta1 0.9 \
    --beta2 0.98 \
    --eps 1e-6 \
    --save-most-recent \
    --grad-checkpoint \
    --precision amp_bfloat16 \


