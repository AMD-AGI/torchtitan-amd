#!/bin/bash
#SBATCH --job-name=deepseek-671b
#SBATCH --output=logs/deepseek-671b-1019/deepseek-671b-lbs32-64N-FSDP.%j.out
#SBATCH --nodes=64                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=96                   # assign all CPUs to the job
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=23:00:00                      # Adjust as necessary
#SBATCH --exclude=chi[2871,2869,2894,2827,2830,2854,2893]
##SBATCH --nodelist=chi[2743,2772,2774,2779,2798,2800-2804,2810-2813,2819-2825,2835-2838,2854-2857,2859-2861,2863-2865,2867-2870,2872,2874-2875,2877-2885,2888-2890,2893-2902]
export SLURM_TREE_WIDTH=128

srun docker login -u rocmshared -p password 

# Setup your keys for HF and WADNB
export HF_TOKEN=${HF_TOKEN:="your_hf_token"}    # please set your HF token here or via environment variable
export WANDB_API_KEY=${WANDB_API_KEY:="wandb_api_key"}
# export WANDB_API_KEY=${WANDB_API_KEY:="your_wandb_token"}    # please set your WANDB token here
# Setup the mount points for the host and container
export HOST_MOUNT=${HOST_MOUNT:="/mnt/models"}     # change this path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${CONTAINER_MOUNT:="/mnt/models"}      # change this path to development workspace path inside the docker
export REBUILD_PRIMUS_TURBO=0
MODEL_NAME=deepseek-671b # llama4-scout, llama4-maverick, deepseek-16b, llama3, deepseek-236b, deepseek-671b
export SAVE_TRACES_FOLDER=${SAVE_TRACES_FOLDER:="deepseek-671b-lbs32-64N-FSDP-traces"}

# Setup the config file and repo id for the model
if [ "$MODEL_NAME" == "llama4-scout" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/experiments/llama4/train_configs/llama4_17bx16e.toml"}     
  export REPO_ID=${REPO_ID:="meta-llama/Llama-4-Scout-17B-16E"} 
elif [ "$MODEL_NAME" == "llama4-maverick" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/experiments/llama4/train_configs/llama4_17bx128e.toml"}     
  export REPO_ID=${REPO_ID:="meta-llama/Llama-4-Maverick-17B-128E"} 
elif [ "$MODEL_NAME" == "deepseek-16b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml"}  
  export REPO_ID=${REPO_ID:="deepseek-ai/deepseek-moe-16b-base"}   
elif [ "$MODEL_NAME" == "deepseek-236b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/deepseek_v3/train_configs/deepseek_v3_236b.toml"}  
  export REPO_ID=${REPO_ID:="deepseek-ai/DeepSeek-V2"}
elif [ "$MODEL_NAME" == "deepseek-671b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b_mi355.toml"}  
  export REPO_ID=${REPO_ID:="deepseek-ai/DeepSeek-V3.1-Base"}
elif [ "$MODEL_NAME" == "llama3-70b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/llama3/train_configs/llama3_70b.toml"}  
  export REPO_ID=${REPO_ID:="meta-llama/Llama-3.1-70B"}
elif [ "$MODEL_NAME" == "llama3-8b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/llama3/train_configs/llama3_8b.toml"}  
  export REPO_ID=${REPO_ID:="meta-llama/Llama-3.1-8B"}
elif [ "$MODEL_NAME" == "llama3-405b" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/llama3/train_configs/llama3_405b.toml"}  
  export REPO_ID=${REPO_ID:="meta-llama/Llama-3.1-405B"}
else
  echo "Please add new mode confing in the run_slurm_pretrain.sh file"
  exit 1
fi
                           
# Setup the turbo wheel file and torch version
export TORCH_VERSION=${TORCH_VERSION:="2.9.0.dev20250825+rocm6.3"} # torch version to install in the container
export PRIMUS_TURBO_WHEEL=${PRIMUS_TURBO_WHEEL:="3rdparty/dist/fp8_opt/primus_turbo-0.1.0+acf2d3c-cp310-cp310-linux_x86_64.whl"} # path to your local bulid turbo wheel file

export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-"2"}

echo "get first node"
# Get the list of nodes and the first node (master node)
# node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo "node list: $(scontrol show hostnames $SLURM_JOB_NODELIST)" 
COORDINATOR_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
master_node=$COORDINATOR_IP

# Set environment variables for distributed training
export SLURM_MASTER_ADDR=${SLURM_MASTER_ADDR:="$master_node"}
export SLURM_MASTER_PORT=${SLURM_MASTER_PORT:-"29565"}

# Optional: Print out the values for debugging
echo "MASTER_ADDR=$SLURM_MASTER_ADDR"
echo "MASTER_PORT=$SLURM_MASTER_PORT"


export ANP_HOME_DIR="/mnt/models/apps/amd-anp" # need to build it
export RCCL_HOME_DIR="/mnt/models/apps/rccl" # need to build it

export USING_AINIC=${USING_AINIC:="1"}  # set to 1 if using AINIC, otherwise 0
# export NCCL_DEBUG=${NCCL_DEBUG:="INFO"}

if [ "$USING_AINIC" == "1" ]; then
    # Define the Docker image
    export NCCL_IB_HCA=${NCCL_IB_HCA:="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"} # modify based on the GPU NiC settings
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:="enp193s0f1np1"}
    export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/pytorch-private:titan-mi355-10.16"}
else
    # Define the Docker image
    export NCCL_IB_HCA=${NCCL_IB_HCA:="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8"} # modify based on the GPU NiC settings
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:="enp193s0f1np1"}
    export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/pytorch-private:titan-mi355-10.16"}
fi 
echo $NCCL_IB_HCA

# Pull docker image
srun docker pull $DOCKER_IMAGE

export TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Current time: $TIME_STAMP"

export TITAN_DIR=${PWD}           

# Setup the IB mount options
if [ -e "/etc/libibverbs.d/bnxt_re.driver" ]; then
  echo "/etc/libibverbs.d exists and using broadcom or aininc."
  export IB_MOUNT_OPTIONS=" -v /etc/libibverbs.d/:/etc/libibverbs.d  \
  -v /usr/lib/x86_64-linux-gnu/libibverbs/:/usr/lib/x86_64-linux-gnu/libibverbs/ \
  "
else
  echo "/etc/libibverbs.d does not exist not using ."
  export IB_MOUNT_OPTIONS=""
fi
echo $IB_MOUNT_OPTIONS

export USE_ROCM_AITER_ROPE_BACKEND=0 # accelate aiter compile

srun bash -c 'echo 0 | sudo tee /proc/sys/kernel/numa_balancing; '


# Run the Docker container with the script
srun bash -c "docker ps -aq | xargs -r docker rm -f ; \
docker run --rm \
 --env SLURM_MASTER_ADDR=\$SLURM_MASTER_ADDR \
 --env SLURM_MASTER_PORT=\$SLURM_MASTER_PORT \
 --env SLURM_PROCID=\$SLURM_PROCID \
 --env SLURM_NODEID=\$SLURM_NODEID \
 --env SLURM_NNODES=\$SLURM_NNODES \
 --env USING_AINIC=\$USING_AINIC \
 --env NCCL_DEBUG=\$NCCL_DEBUG \
 --env ANP_HOME_DIR=\$ANP_HOME_DIR \
 --env RCCL_HOME_DIR=\$RCCL_HOME_DIR \
 --env NCCL_IB_HCA=\$NCCL_IB_HCA \
 --env NCCL_SOCKET_IFNAME=\$NCCL_SOCKET_IFNAME \
 --env CACHE_TAG=\$CACHE_TAG \
 --env CONFIG_FILE=\$CONFIG_FILE \
 --env REPO_ID=\$REPO_ID \
 --env HF_TOKEN=\$HF_TOKEN \
 --env TORCH_VERSION=\$TORCH_VERSION \
 --env WANDB_API_KEY=\$WANDB_API_KEY \
 --env PRIMUS_TURBO_WHEEL=\$PRIMUS_TURBO_WHEEL \
 --env CONTAINER_MOUNT=\$CONTAINER_MOUNT \
 --env SAVE_TRACES_FOLDER=\$SAVE_TRACES_FOLDER \
 --env REBUILD_PRIMUS_TURBO=\$REBUILD_PRIMUS_TURBO \
 --env USE_ROCM_AITER_ROPE_BACKEND=\$USE_ROCM_AITER_ROPE_BACKEND \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 -v \$HOST_MOUNT:\$CONTAINER_MOUNT \
 -v /mnt/vfs/dataset:/mnt/vfs/dataset \
 \${IB_MOUNT_OPTIONS} \
 \$DOCKER_IMAGE /bin/bash -c \
 'echo \$(date) ; \
   export AITER_TARGET_ARCH=gfx950 ; \
   export PYTORCH_ROCM_ARCH=gfx950 ; \
   export ROCM_TARGET_ARCH=gfx950 ; \
   export AITER_JIT_COMPILE_THREADS=32 ; \
   export CMAKE_BUILD_PARALLEL_LEVEL=32 ; \
   export PYTORCH_TUNABLEOP_ENABLED=0 ; \
   export PYTORCH_DISABLE_FLASH_ATTENTION_TUNABLE=1 ; \
   export MAX_JOBS=32 ; \
   export GPU_ARCHS=gfx950  ; \
   cd \$CONTAINER_MOUNT/john/torchtitan-amd ; \
   wandb login \$WANDB_API_KEY; \
    HOST_NAME=\$(hostname) ; \
    pwd ; ls -l ; \
    pip3 install -r requirements.txt ; \
    pip3 install -e . ; \
    pip3 install torchao ; \
    pip3 uninstall numpy -y && pip install numpy==1.26.4; \ 
    pip3 install -qq hip-python --extra-index-url https://test.pypi.org/simple ; \
    pip3 install --extra-index-url https://test.pypi.org/simple \$PRIMUS_TURBO_WHEEL ; \
    python scripts/download_hf_assets.py --assets tokenizer --repo_id \$REPO_ID --hf_token=\$HF_TOKEN ; \
    export NCCL_PXN_DISABLE=0 ; \
    export NCCL_P2P_NET_CHUNKSIZE=262144 ; \
    CONFIG_FILE=\$CONFIG_FILE SAVE_TRACES_FOLDER=\$SAVE_TRACES_FOLDER bash run_multinode_train.sh ; \
    ls /opt/venv/lib/python3.10/site-packages/aiter/jit/build/module_aiter_enum/build/ ; \
 echo \$(date) 
 '"