#!/usr/bin/bash -x
#SBATCH --account=laion
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=6:00:00
#SBATCH --partition=g40x
#SBATCH --job-name=openclip-vitb32-toy-1epoch
#SBATCH --output=logs/evaluate-openclip-latentvitb32-toy-1epoch-%j.out

# Check for command-line arguments
if [ -z "$1" ]; then
    echo "Model/config name required."
    exit 1
fi

if [ -z "$2" ]; then
    echo "wandb run name required."
    exit 1
fi

if [ -z "$3" ]; then
    echo "checkpoint project name required."
    exit 1
fi

MODEL=$1
NAME=$2
#LOGS=/scratch/wendlerc/logs
LOGS=/admin/home-wendlerc/models
CKPT=$3

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
export MASTER_PORT=12802
source .env2/bin/activate
export PYTHONPATH=src:$PYTHONPATH
export NLTK_DATA=cache

#     --report-to "wandb,tensorboard" \
#    --wandb-project-name "open-clip-eval_welcomewest.hpc.stability.ai" \
#    --logs $LOGS \
#    --precision amp_bfloat16 \
#     --name ${NAME}_val \


srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --model $MODEL \
    --pretrained $LOGS/$NAME/checkpoints/$CKPT \
    --imagenet-val /admin/home-wendlerc/data/imagenet-1k-val \
    --precision amp_bfloat16

