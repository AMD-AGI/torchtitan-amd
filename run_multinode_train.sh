#!/usr/bin/bash

set -ex
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-"2"}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-"1"}
export NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE:-"1"}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-"0"}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-"1"}
export NCCL_PROTO=${NCCL_PROTO:-"Simple"}
export RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE:-"0"}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-"1"}
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-"0"}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-"262144"}
# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_70b.toml"}

GPUS_PER_NODE=$NGPU
echo "getting SLURM_MASTER_ADDR:" ${SLURM_MASTER_ADDR}
echo "getting SLURM_MASTER_PORT:" ${SLURM_MASTER_PORT}
MASTER_ADDR=${SLURM_MASTER_ADDR}
MASTER_PORT=${SLURM_MASTER_PORT}
echo "SLURM_PROCID:$SLURM_PROCID"
echo "SLURM_NODEID:$SLURM_NODEID"
echo "SLURM_NNODES: $SLURM_NNODES"
NNODES=$SLURM_NNODES
NODE_RANK=${SLURM_NODEID}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export REBUILD_PRIMUS_TURBO=${REBUILD_PRIMUS_TURBO:-0}
# install primus turbo from source
if [ "$REBUILD_PRIMUS_TURBO" == "1" ]; then
    echo "Rebuilding Primus Turbo from source..."
    mkdir -p "/workspace/turbo"
    cd "/workspace/turbo"
    git clone https://github.com/AMD-AGI/Primus-Turbo.git --recursive
    cd Primus-Turbo
    git checkout main
    pip3 install -r requirements.txt
    pip3 install --no-build-isolation .
    # Set GPU_ARCHS to compile Turbo for multiple AMD GPU architectures.
    GPU_ARCHS="gfx942;gfx950" pip3 install --no-build-isolation .
    cd "$CONTAINER_MOUNT/torchtitan-amd"
    echo "Rebuilding Primus Turbo from source done."
else
    echo "Skip Primus Turbo rebuild. REBUILD_PRIMUS_TURBO=$REBUILD_PRIMUS_TURBO"
fi

if [ "$USING_AINIC" == "1" ]; then
    # Setup Pollara specific args
    echo "Using AINIC"
    echo "RCCL_HOME_DIR: $RCCL_HOME_DIR"
    echo "ANP_HOME_DIR: $ANP_HOME_DIR"
    export NCCL_MAX_P2P_CHANNELS=56
    export NCCL_IB_TC=104      # traffic class 
    export NCCL_IB_FIFO_TC=192 # match network cts  
    export NET_OPTIONAL_RECV_COMPLETION=1
    export NCCL_IB_USE_INLINE=1
    export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
    export NCCL_GDR_FLUSH_DISABLE=1
    export NCCL_DMABUF_ENABLE=0
    export NCCL_IGNORE_CPU_AFFINITY=1
    export NCCL_IB_QPS_PER_CONNECTION=1
    #export NCCL_IB_ROCE_VERSION_NUM=2 #
    export NCCL_IB_GID_INDEX=1 # suggest to use this value in vultr cluster
    export LD_LIBRARY_PATH=${RCCL_HOME_DIR}/build/release:${ANP_HOME_DIR}/build:${ANP_HOME_DIR}/build/lib:$LD_LIBRARY_PATH
    export LD_PRELOAD=${ANP_HOME_DIR}/build/librccl-net.so:${RCCL_HOME_DIR}/build/release/librccl.so.1.0
else
    export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-"3"} 
fi


overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=${NNODES} \
         --node_rank ${NODE_RANK} \
         --nproc_per_node=${NGPU} \
         --rdzv_backend c10d \
         --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --local-ranks-filter ${LOG_RANK} \
        --role rank --tee 3 \
        torchtitan/train.py --job.config_file ${CONFIG_FILE}
