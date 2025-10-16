#!/bin/bash
#SBATCH --job-name=john-titan
#SBATCH --output=logs/llama3-8B/llama-test.%j.out
#SBATCH --nodes=8                            # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=96                   # assign all CPUs to the job
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=01:00:00                      # Adjust as necessary
##SBATCH --nodelist=chi[2612,2631,2643-2646,2649,2661,2672]

# Setup your keys for HF and WADNB
export HF_TOKEN=${HF_TOKEN:="your_hf_token"}    # please set your HF token here or via environment variable
# export WANDB_API_KEY=${WANDB_API_KEY:="your_wandb_token"}    # please set your WANDB token here
# Setup the mount points for the host and container
export HOST_MOUNT=${HOST_MOUNT:="/mnt/models/john"}     # change this path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${CONTAINER_MOUNT:="/workspace/john"}      # change this path to development workspace path inside the docker

MODEL_NAME=llama3-8b # llama4-scout, llama4-maverick, deepseek-16b, llama3, deepseek-236b, deepseek-671b
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
export TORCH_VERSION=${TORCH_VERSION:="2.9.0.dev20250825+rocm6.3"}                                   # torch version to install in the container

export PRIMUS_TURBO_WHEEL=${PRIMUS_TURBO_WHEEL:="3rdparty/primus_turbo-0.1.0+dbeaf79-cp310-cp310-linux_x86_64.whl"} # path to your local bulid turbo wheel file

export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-"4"}

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

# export OMPI_MCA_btl_tcp_if_include=enp193s0f1np1; export OMPI_MCA_btl_tcp6=0 ; mpirun --allow-run-as-root -np 16 -N 8 -H chi2740:8,chi2742:8 --mca pml ob1 --mca oob_tcp_if_include "enp193s0f1np1"  -x UCX_IB_SRQ_DISABLE=1  -x NCCL_DEBUG=WARN -x NCCL_IB_ROCE_VERSION_NUM=2  \
# -x NCCL_NET_GDR_READ=1 -x NCCL_SHM_DISABLE=1 -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1 -x NCCL_IGNORE_CPU_AFFINITY=1 -x NCCL_MIN_NCHANNELS=64 -x NCCL_MAX_NCHANNELS=64\
#  -x NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 -x NCCL_SOCKET_IFNAME=enp193s0f1np1 -x NCCL_PXN_DISABLE=0 -x HSA_NO_SCRATCH_RECLAIM=1 /mnt/models/pras/rccl-tests/build/all_reduce_perf -b 8 -e 32g -f 2 -g 1

srun docker login -u username -p password 

# export ANP_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/amd-anp-1.1.0-5"
# export RCCL_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/rccl-drop-2025-08"

export USING_AINIC=${USING_AINIC:="1"}  # set to 1 if using AINIC, otherwise 0
export NCCL_DEBUG=${NCCL_DEBUG:="INFO"}

if [ "$USING_AINIC" == "1" ]; then
    # Define the Docker image
    export NCCL_IB_HCA=${NCCL_IB_HCA:="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"} # modify based on the GPU NiC settings
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:="enp193s0f1np1"}
    export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/pytorch-private:titan-mi355-10.16"}
    # export ANP_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/amd-anp-1.1.0-5"
    # export RCCL_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/rccl-drop-2025-08"
else
    # Define the Docker image
    export NCCL_IB_HCA=${NCCL_IB_HCA:="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8"} # modify based on the GPU NiC settings
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:="enp49s0f0np0"}
    export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/pytorch-private:titan-mi355-10.16"}
fi 
echo $NCCL_IB_HCA

# Pull docker image
srun docker pull $DOCKER_IMAGE

export TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Current time: $TIME_STAMP"
# Define the mount points
export TITAN_DIR=${PWD}                                      # change this path to Megatron-LM inside the docker

# Setup the IB mount options
if [ -e "/etc/libibverbs.d/bnxt_re.driver" ]; then
  echo "/etc/libibverbs.d exists and using broadcom or aininc."
  export IB_MOUNT_OPTIONS=" -v /etc/libibverbs.d/:/etc/libibverbs.d \
  "
else
  echo "/etc/libibverbs.d does not exist not using ."
  export IB_MOUNT_OPTIONS=""
fi
echo $IB_MOUNT_OPTIONS

export OMPI_MCA_btl_tcp_if_include=enp193s0f1np1; 
export OMPI_MCA_btl_tcp6=0 ; 

# export IB_MOUNT_OPTIONS=""
# -v /usr/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/
  # -v /usr/local/lib/librccl.so.1:/usr/local/lib/librccl.so.1 \
  # -v /usr/local/lib/librccl-net.so:/usr/local/lib/librccl-net.so \
  # -v /usr/local/lib/librccl.so.1.0:/usr/local/lib/librccl.so.1.0 \
# setup the CPU governor to performance and disable numa balancing 
srun bash -c 'echo 0 | sudo tee /proc/sys/kernel/numa_balancing; '


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
 --env PRIMUS_TURBO_WHEEL=\$PRIMUS_TURBO_WHEEL \
 --env CONTAINER_MOUNT=\$CONTAINER_MOUNT \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 -v \$HOST_MOUNT:\$CONTAINER_MOUNT \
 \${IB_MOUNT_OPTIONS} \
 \$DOCKER_IMAGE /bin/bash -c \
 'echo \$(date) ; \
    unset AITER_ASM_DIR ; \
    cd \$CONTAINER_MOUNT/torchtitan-amd ; \
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
    ls /opt/venv/lib/python3.10/site-packages/aiter/jit/ ; \
    CONFIG_FILE=\$CONFIG_FILE bash run_multinode_train.sh ; \
    ls /opt/venv/lib/python3.10/site-packages/aiter/jit/build/module_aiter_enum/build/ ; \
 echo \$(date) 
 '"

    # cd /workspace ; \
    # git clone https://github.com/AMD-AIG-AIMA/Primus-Turbo.git --recursive ; \
    # cd Primus-Turbo ; \
    # git checkout dev/fix_layout ; \
    # pip3 install -r requirements.txt ; \
    # pip uninstall numpy -y && pip install numpy==1.26.4 ; \
    # pip3 install --no-build-isolation -e . -v ; \
    # pip3 install -qq hip-python --extra-index-url https://test.pypi.org/simple ; \
    # pip3 install --extra-index-url https://test.pypi.org/simple \$PRIMUS_TURBO_WHEEL ; \

    # pip3 install -qq hip-python --extra-index-url https://test.pypi.org/simple ; \
    # pip3 install --extra-index-url https://test.pypi.org/simple \$PRIMUS_TURBO_WHEEL ; \
    # pip3 uninstall aiter -y ; \
    # cd /workspace ; \
    # git clone --recursive https://github.com/ROCm/aiter.git ; \
    # cd aiter ; \
    # python3 setup.py develop  ; \
    # echo AITER VERSION: ; \
    # cd \$CONTAINER_MOUNT/torchtitan-amd ; \