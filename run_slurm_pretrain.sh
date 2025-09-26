#!/bin/bash
#SBATCH --job-name=titan
#SBATCH --output=logs/slurm/llama4_17bx16e_pretraining.%j.out
#SBATCH --nodes=2                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=96                   # assign all CPUs to the job
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=01:00:00                      # Adjust as necessary
##SBATCH --nodelist=chi[2599-2600,2602-2603,2630-2631,2643-2644]  # modify based on your reservation settings
##SBATCH --reservation=vultr-mi325x-torch # modify based on your reservation settings

# Setup your keys for HF and WADNB
export HF_TOKEN=${HF_TOKEN:="your_hf_token"}    # please set your HF token here
# export WANDB_API_KEY=${WANDB_API_KEY:="your_wandb_token"}    # please set your WANDB token here
# Setup the mount points for the host and container
export HOST_MOUNT=${HOST_MOUNT:="/mnt/models/your_path"}     # change this path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${CONTAINER_MOUNT:="/workspace"}      # change this path to development workspace path inside the docker

MODEL_NAME=llama4-scout # llama4-scout, llama4-maverick, deepseek-16b, llama3, deepseek-236b, deepseek-671b
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
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b.toml"}  
  export REPO_ID=${REPO_ID:="deepseek-ai/DeepSeek-V3.1-Base"}
elif [ "$MODEL_NAME" == "llama3" ]; then
  export CONFIG_FILE=${CONFIG_FILE:="torchtitan/models/llama3/train_configs/llama3_8b.toml"}  
  export REPO_ID=${REPO_ID:="meta-llama/Llama-3.1-8B"}
else
  echo "Please add new mode confing in the run_slurm_pretrain.sh file"
  exit 1
fi
                           
# Setup the turbo wheel file and torch version
export TORCH_VERSION=${TORCH_VERSION:="2.9.0.dev20250825+rocm6.3"}                                   # torch version to install in the container
export PRIMUS_TURBO_WHEEL=${PRIMUS_TURBO_WHEEL:="3rdparty/primus_turbo-0.1.0+5fb2c17-cp310-cp310-linux_x86_64.whl"} # path to your local bulid turbo wheel file
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-"2"}

echo "get first node"
# Get the list of nodes and the first node (master node)
# node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo "node list: $(scontrol show hostnames $SLURM_JOB_NODELIST)" 
COORDINATOR_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
master_node=$COORDINATOR_IP
# node_array=(${node_list})
# master_node=${node_array[0]}

# Set environment variables for distributed training
export SLURM_MASTER_ADDR=${SLURM_MASTER_ADDR:="$master_node"}
export SLURM_MASTER_PORT=${SLURM_MASTER_PORT:-"29565"}

# Optional: Print out the values for debugging
echo "MASTER_ADDR=$SLURM_MASTER_ADDR"
echo "MASTER_PORT=$SLURM_MASTER_PORT"

# Define the Docker image
export NCCL_IB_HCA=${NCCL_IB_HCA:="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8"} # modify based on the GPU NiC settings
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:="enp49s0f0np0"}
echo $NCCL_IB_HCA

export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.5_py310"}
# Pull docker image
srun docker pull $DOCKER_IMAGE

export TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Current time: $TIME_STAMP"
# Define the mount points
export TITAN_DIR=${PWD}                                      # change this path to Megatron-LM inside the docker

# Setup the IB mount options
if [ -e "/etc/libibverbs.d/bnxt_re.driver" ]; then
  echo "/etc/libibverbs.d exists and using broadcom."
  export IB_MOUNT_OPTIONS="-v /usr/bin:/usr/bin -v /etc/libibverbs.d/:/etc/libibverbs.d -v /usr/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/ -v /usr/local/lib:/usr/local/lib"
  
else
  echo "/etc/libibverbs.d does not exist not using ."
  export IB_MOUNT_OPTIONS=""
fi

# setup the CPU governor to performance and disable numa balancing 
srun bash -c 'echo 0 | sudo tee /proc/sys/kernel/numa_balancing; \
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'


# Collect environment info for cache tagging, skip aiter jit by read from cache
OS_VER=$(grep ^PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '"' | tr ' ' '_' | tr -d '()')
PY_VER=$(python3 -c 'import platform; print(platform.python_version())')
ROCM_VER=$(/opt/rocm/bin/rocminfo | grep 'ROCm version' | head -1 | awk '{print $NF}' | tr -d '()')
if [[ -f /proc/driver/amdgpu/version ]]; then
    AMDGPU_VER=$(head -1 < /proc/driver/amdgpu/version | awk '{print $3}' | tr -d '()')
else
    AMDGPU_VER="unknown"
fi
KERNEL_VER=$(uname -r | tr '.' '_' | tr '-' '_')

export CACHE_TAG="${OS_VER}_py${PY_VER}_rocm${ROCM_VER}_amdgpu${AMDGPU_VER}_kernel${KERNEL_VER}"


# Run the Docker container with the script
srun bash -c "docker ps -aq | xargs -r docker rm -f ; \
docker run --rm \
 --env SLURM_MASTER_ADDR=\$SLURM_MASTER_ADDR \
 --env SLURM_MASTER_PORT=\$SLURM_MASTER_PORT \
 --env SLURM_PROCID=\$SLURM_PROCID \
 --env SLURM_NODEID=\$SLURM_NODEID \
 --env SLURM_NNODES=\$SLURM_NNODES \
 --env NCCL_IB_HCA=\$NCCL_IB_HCA \
 --env NCCL_SOCKET_IFNAME=\$NCCL_SOCKET_IFNAME \
 --env CACHE_TAG=\$CACHE_TAG \
 --env CONFIG_FILE=\$CONFIG_FILE \
 --env REPO_ID=\$REPO_ID \
 --env HF_TOKEN=\$HF_TOKEN \
 --env TORCH_VERSION=\$TORCH_VERSION \
 --env PRIMUS_TURBO_WHEEL=\$PRIMUS_TURBO_WHEEL \
 --env CONTAINER_MOUNT=\$CONTAINER_MOUNT \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 -v \$HOST_MOUNT:\$CONTAINER_MOUNT \
 \${IB_MOUNT_OPTIONS} \
 \$DOCKER_IMAGE /bin/bash -c \
 'echo \$(date) ; \
    cd \$CONTAINER_MOUNT/torchtitan-amd ; \
    HOST_NAME=\$(hostname) ; \
    export TMP_BUILD_DIR=\$CONTAINER_MOUNT/torchtitan-amd/3rdparty/build/\$HOST_NAME ; \
    mkdir -p \$TMP_BUILD_DIR ; \
    export AITER_JIT_DIR=\$TMP_BUILD_DIR/\${CACHE_TAG}_aiter_cache ; \
    echo AITER_JIT_DIR is \$AITER_JIT_DIR ; \
    pip3 install torch==\$TORCH_VERSION torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --force-reinstall ; \
    pip3 install -r requirements.txt ; \
    pip3 install -e . ; \
    pip3 install -qq hip-python --extra-index-url https://test.pypi.org/simple ; \
    pip3 install --extra-index-url https://test.pypi.org/simple \$PRIMUS_TURBO_WHEEL ; \
    pip install torchao ; \
    pip uninstall numpy -y && pip install numpy==1.26.4; \ 
    python scripts/download_hf_assets.py --assets tokenizer --repo_id \$REPO_ID --hf_token=\$HF_TOKEN ; \
    CONFIG_FILE=\$CONFIG_FILE bash run_multinode_train.sh ; \
 echo \$(date) 
 '"