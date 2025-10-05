#!/usr/bin/bash
# How to use entryPoint.sh
# Example job: http://127.0.0.1:30183/training/detail?id=john-titan-dsv2-2node-template-mfbql

set -ex
# export TORCH_TITAN_PATH=/home/yanyuan.qin@amd.com/meta/torchtitan-amd
cd $TORCH_TITAN_PATH

MODEL_NAME=deepseek-16b # llama4-scout, llama4-maverick, deepseek-16b, llama3, deepseek-236b, deepseek-671b
export WANDB_RUN_NAME="titian-deepseek-16b"
# Set wandb ENV
export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan-PTC"}
mkdir -p $TORCH_TITAN_PATH/logs
export WANDB_DIR=$TORCH_TITAN_PATH/logs
export LOGS_DIR=$TORCH_TITAN_PATH/logs

# Select config file and repo id based on MODEL_NAME
if [ "$MODEL_NAME" = "llama4-scout" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/experiments/llama4/train_configs/llama4_17bx16e.toml"}
  export REPO_ID=${REPO_ID:="meta-llama/Llama-4-Scout-17B-16E"}
elif [ "$MODEL_NAME" = "llama4-maverick" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/experiments/llama4/train_configs/llama4_17bx128e.toml"}
  export REPO_ID=${REPO_ID:="meta-llama/Llama-4-Maverick-17B-128E"}
elif [ "$MODEL_NAME" = "deepseek-16b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml"}
  export REPO_ID=${REPO_ID:="deepseek-ai/deepseek-moe-16b-base"}
elif [ "$MODEL_NAME" = "deepseek-236b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/deepseek_v3/train_configs/deepseek_v3_236b.toml"}
  export REPO_ID=${REPO_ID:="deepseek-ai/DeepSeek-V2"}
elif [ "$MODEL_NAME" = "deepseek-671b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b.toml"}
  export REPO_ID=${REPO_ID:="deepseek-ai/DeepSeek-V3.1-Base"}
elif [ "$MODEL_NAME" = "llama3-70b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/llama3/train_configs/llama3_70b.toml"}
  export REPO_ID=${REPO_ID:="meta-llama/Llama-3.1-70B"}
elif [ "$MODEL_NAME" = "llama3-8b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/llama3/train_configs/llama3_8b.toml"}
  export REPO_ID=${REPO_ID:="meta-llama/Llama-3.1-8B"}
elif [ "$MODEL_NAME" = "llama3-405b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/llama3/train_configs/llama3_405b.toml"}
  export REPO_ID=${REPO_ID:="meta-llama/Llama-3.1-405B"}
else
  echo "Please add new mode config in entryPoint.sh"
  exit 1
fi

# set up NCCL  
export NCCL_IB_HCA=rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1
export NCCL_DEBUG=WARN # INFO, WARN, ERROR, DEBUG, TRACE, OFF
# build rccl library
tar xzf "/home/primus/data/libbnxt/libbnxt_re-234.0.154.0.tar.gz" -C /tmp/ && \
mv /tmp/libbnxt_re-* /tmp/libbnxt && \
mv /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.inbox && \
cd /tmp/libbnxt/ && sh ./autogen.sh && ./configure && \
make -C /tmp/libbnxt clean all install && \
echo '/usr/local/lib' > /etc/ld.so.conf.d/libbnxt_re.conf && \
ldconfig && \
cp -f /tmp/libbnxt/bnxt_re.driver /etc/libibverbs.d/ && \
echo "Rebuilding libbnxt done."

# set up torch and primus-turbo libraries
cd $TORCH_TITAN_PATH 
# Setup the turbo wheel file and torch version
export TORCH_VERSION=${TORCH_VERSION:="2.9.0.dev20250825+rocm6.3"}                                   # torch version to install in the container
export PRIMUS_TURBO_WHEEL=${PRIMUS_TURBO_WHEEL:="3rdparty/primus_turbo-0.1.0+2e40784-cp310-cp310-linux_x86_64.whl"} 
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-"2"}

HOST_NAME=$(hostname) ; 
export TMP_BUILD_DIR=3rdparty/build/$HOST_NAME ; 
mkdir -p $TMP_BUILD_DIR ; 
export CACHE_TAG="ubuntu"
#export AITER_JIT_DIR=$TMP_BUILD_DIR/${CACHE_TAG}_aiter_cache ; 
#echo AITER_JIT_DIR is $AITER_JIT_DIR ; 
pip3 install torch==$TORCH_VERSION torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --force-reinstall ; 
#pip3 install /home/yanyuan.qin@amd.com/meta/torch-2.9.0.dev20250825+rocm6.3-cp310-cp310-manylinux_2_28_x86_64.whl --force-reinstall ; 
pip3 install -r requirements.txt ; 
pip3 install -e . ; 
pip3 install -qq hip-python --extra-index-url https://test.pypi.org/simple ; 
pip3 install --extra-index-url https://test.pypi.org/simple $PRIMUS_TURBO_WHEEL ; 
pip3 install torchao ; 
pip3 uninstall numpy -y && pip3 install numpy==1.26.4; 
python scripts/download_hf_assets.py --assets tokenizer --repo_id $REPO_ID --hf_token=$HF_TOKEN

# Set cluster ENV, those info is from k8s by default
export MASTER_ADDR=${MASTER_ADDR:-localhost} # those info is from k8s by default
export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NGPU=${NGPU:-"8"}

echo "IP_INTERFACE=${IP_INTERFACE}"
echo "NCCL_IB_HCA=${NCCL_IB_HCA}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

export LOG_RANK=${LOG_RANK:-0}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-"4"}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-"1"}
export NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE:-"1"}
# export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-"3"}
export NCCL_IB_ROCE_VERSION_NUM=${NCCL_IB_ROCE_VERSION_NUM:-"2"}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-"0"}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-"1"}
export NCCL_PROTO=${NCCL_PROTO:-"Simple"}
export RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE:-"0"}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-"1"}
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-"0"}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-"262144"}
TIME_STAMP=$(date +%Y%m%d%H%M%S)

# start training
torchrun --nnodes=${NNODES} \
         --node_rank ${NODE_RANK} \
         --nproc_per_node=${NGPU} \
         --master_addr "${MASTER_ADDR}" \
         --master_port "${MASTER_PORT}" \
         --local-ranks-filter ${LOG_RANK} \
         --role rank --tee 3 \
         torchtitan/train.py --job.config_file ${CONFIG_FILE} "$@" 2>&1 | tee ${LOGS_DIR}/${WANDB_RUN_NAME}-${TIME_STAMP}.log
