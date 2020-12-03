#!/bin/bash

function prune_111 {
    echo "Pruning CuDNN"

    export CUDNN_DIR="cuda/lib64"
    export NVPRUNE="/usr/local/cuda-11.2/bin/nvprune"
    export GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"

    $NVPRUNE $GENCODE_CUDNN $CUDNN_DIR/libcudnn_static.a -o $CUDNN_DIR/libcudnn_static.a
}

wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.1-linux-x64-v8.0.5.39.tgz -O cudnn-8.0.tgz
tar xvf cudnn-8.0.tgz
prune_111
tar cvzf cudnn-11.1-linux-x64-v8.0.5.39.tgz cuda
